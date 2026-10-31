# inference

A small LLM serving engine for Qwen2.5-0.5B-Instruct, written from scratch as a learning
project. It covers Triton attention kernels, a paged KV cache, CUDA-graph decode, chunked
prefill and a continuous-batching scheduler with preemption. A vLLM baseline runs the same
model, and a serving benchmark with Poisson arrivals compares the two.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, fp16 (24 layers, 14 q heads / 2 kv heads, head_dim 64)
- Sampling: greedy (`argmax`)
- Tested on: RTX 4070 Laptop (8 GiB), WSL2 Ubuntu, Python 3.12, torch 2.13, vLLM 0.31.0

## Layout

| Path | Contents |
|---|---|
| `kernels/decode_attention.py` | Split-KV GQA decode attention (load + merge kernels) |
| `kernels/prefill_attention.py` | Varlen GQA mixed attention: chunked prefill and decode in one kernel |
| `attn/core.py` | Kernel host functions, `triton_op` wrappers, `TritonQwen2Attention` replacing HF `Qwen2Attention` |
| `attn/metadata.py` | Attention metadata passed to the forward (slot ids, `cu_seqlens`, pages/offsets) |
| `piecewise/` | Piecewise `torch.compile` backend: splits the graph at attention, captures the other pieces as CUDA graphs |
| `kv_cache/storage.py` | Paged KV pool, block table, per-slot length and last generated token |
| `scheduler/scheduler.py` | Thread-based scheduler: queue, token budget, page allocation, preemption |
| `scheduler/models.py` | Per-request state kept by the scheduler (`Request`) |
| `scheduler/decode_module.py` | Pure decode forward captured as a CUDA graph |
| `scheduler/mixed_module.py` | Varlen (prefill + decode) forward: `FullMixedModule` (whole-forward CUDA graph) and `PiecewiseMixedModule` (piecewise CUDA graphs) |
| `scheduler/graph_backend.py` | `FullGraphBackend` (batch size x token count grid) and `PiecewiseGraphBackend` (token count buckets) |
| `benchmark/` | Poisson-arrival serving benchmark (`python -m benchmark`) with an HF greedy accuracy check |
| `vllm_baseline/` | vLLM engine for the same model and benchmark |
| `results/` | Benchmark outputs (timestamped JSON) |
| `main.py`, `common.py` | Single-prompt measurement helpers (currently only run vLLM) |

## Architecture

### KV cache (`kv_cache/storage.py`)

KV lives in a single pool of pages, shaped `(n_pages, H_kv, page_size, D)` per layer. Total
capacity is `Scheduler.KV_CACHE_CAPACITY = 32768` tokens.

- `block_table[slot, i]`: physical page id of the slot's `i`-th logical page.
- `lens[slot]`: number of tokens of the slot in KV (GPU side).
- `last_generated[slot]`: the last generated token, which is the decode input.
- Slot 0 and page 0 are unused (`unused_idx`). Padding rows that fill a batch up to its
  bucket size read from and write to them.

Token `t` is stored at page `block_table[slot, t // P]`, offset `t % P`.

### Attention

There are two kernels, both reading paged KV through the block table.

- **Decode** (`decode_gqa_attention`): one query per sequence. The KV length is split into
  `num_programs` parts processed in parallel, and a separate merge kernel combines them with
  online softmax (split-KV / flash-decoding).
- **Mixed** (`mixed_gqa_attention`): varlen. Each sequence knows its token range from
  `cu_seqlens` and the length already in KV from `kv_lens`. Queries start at position
  `kv_len` and read `[0, kv_len + L)` under a causal mask. Decode is the special case
  `L = 1`, so chunked prefill and decode can run in the same forward.

On both paths, K/V of the new tokens are written to KV before the kernel runs
(`decode_update` / `mixed_update`). The Q/K/V projections are fused into one linear.

### Forward paths (`scheduler/`)

Every forward replays CUDA graphs captured at init. Two backends hold them:

- **`FullGraphBackend`**: for each batch-size bucket (`DECODE_BATCH_BUCKETS = 1, 2, 4, 8, 16, 32`)
  - a `DecodeModule` (one token per sequence),
  - and a `FullMixedModule` for each token-count bucket up to `FULL_GRAPH_TOKEN_THRESHOLD = 128`
    (16, 32, 64, 128).

  The whole forward, attention included, is one CUDA graph. The batch is padded to the bucket
  size, so `slot_ids` and `cu_seqlens` have fixed shapes.
- **`PiecewiseGraphBackend`**: one `PiecewiseMixedModule` per token-count bucket above the
  threshold, in steps of `PIECEWISE_TOKEN_STEP = 32` up to `token_budget` (160, 192, ...). The
  forward is compiled with the piecewise backend in `piecewise/backend.py`:
  - the graph is split at the attention ops,
  - the pieces between them are compiled with inductor and captured as CUDA graphs on first call,
  - the gather + attention groups run eagerly and copy their output into a buffer shared by
    all buckets.

  Attention runs outside the graphs, so there is no batch padding. `slot_ids` and `cu_seqlens`
  get the real batch size each step and are marked dynamic. Each bucket compiles at most twice:
  once for `B >= 2` and once for `B = 1`.

Decode-only steps use `DecodeModule`. Steps with a prefill use the full graph backend when the
token count is at most 128, the piecewise backend otherwise. After the replay, KV lengths
are advanced and `lm_head` is applied to the last token of each sequence outside the graph.

### Scheduler (`scheduler/scheduler.py`)

`Scheduler.call(prompts)` returns one `Future` per prompt; `loop()` runs in a separate
thread. The result is `(new_tokens, t_arrival, t_first, t_end)`.

Each step:

1. **Budget for running requests** (`_allocate_pages_pre_request`): `token_budget` is
   handed out in arrival order. Decoding requests get 1 token; prefilling requests get
   `min(remaining budget, remaining prompt)` tokens. The pages these tokens need are
   computed, and if the pool is short, the most recent requests are preempted.
2. **Admit new requests** (`_accept_requests`): if budget and free slots/pages remain,
   preempted requests are taken first, then queued ones, and pages are allocated for their
   first chunk.
3. **Forward** (`_process_slots`): slots that received budget go through the decode or the
   mixed module.
4. **State update** (`_refresh_slots`): `kv_len` grows by the budget. The token produced by
   an unfinished prefill chunk is discarded. Requests that hit EOS or `max_new_tokens` have
   their Future resolved and their pages returned to the pool.

**Preemption** uses recompute: the victim's pages are freed and the request is pushed onto
a stack to be prefilled again with `prompt + generated tokens`. The stack is drained before
new requests, in arrival order. The victim is always the most recently admitted request.

**Shutdown**: `stop()` puts a sentinel on the queue. After the sentinel, no new requests are
admitted; running and preempted requests are finished, then the loop exits. On an error,
the Futures of all running, queued and preempted requests are failed with it.

## Running

GPU runs are done under WSL:

```bash
python -m benchmark --impl ours --n-slots 17
python -m benchmark --impl vllm --max-num-seqs 16
```

Main arguments: `--rates` (default `2 4 8 inf` requests/s), `--n-requests` (64),
`--max-new-tokens` (128), `--n-slots` (5), `--page-size` (16), `--token-budget` (256),
`--max-num-seqs` (vLLM only), `--prompt-tokens MIN MAX` (prepends a context text of a random
length in that range to each question). vLLM gets `max_num_batched_tokens = token_budget`.
Each run is saved to `results/{timestamp}_{impl}.json` with the table, arguments,
environment, git commit and per-request records.

`n_slots` can be at most one more than the largest bucket (32); slot 0 is the padding slot,
so at most 32 requests run at once. `token_budget` must be above 128 and a multiple of 32.

**Accuracy**: before serving, each prompt is generated once with HF (batch 1, greedy,
`repetition_penalty=1.0`) and cached in `results/reference/`. The table reports `exact%`
(requests identical to the reference) and `prefix%` (mean length of the common prefix
relative to the longer output); trailing EOS tokens are ignored.

## Results

64 requests, `max_new_tokens=128`, prompts of 30–50 tokens. Times are in ms; the `inf` row
is all requests arriving at once.

**Against vLLM at the same concurrency (16 requests)**: ours with `n_slots=17` and
`token_budget=256` (current version), vLLM with `max_num_seqs=16`:

| rate | tok/s (ours / vLLM) | ttft_mean | tpot_mean | e2e_mean |
|---|---|---|---|---|
| 2 | 146.6 / 145.5 | 9.7 / 38.4 | 5.85 / 7.19 | 497 / 643 |
| 4 | 288.2 / 284.9 | 9.8 / 39.7 | 5.75 / 7.55 | 489 / 670 |
| 8 | 554.8 / 540.6 | 11.0 / 42.0 | 5.97 / 8.00 | 509 / 715 |
| inf | 2037.1 / 1474.8 | 758 / 1113 | 6.50 / 8.67 | 1295 / 1854 |

**Across versions, at `inf` rate:**

| version | n_slots | tok/s | ttft_mean | tpot_mean |
|---|---|---|---|---|
| fixed per-slot KV | 5 | 644 | – | ~5.8 |
| paged KV | 5 | 586.0 | 4064 | 6.50 |
| paged KV | 17 | 1885.8 | 830 | 7.12 |
| paged KV + chunked prefill (`token_budget=256`) | 17 | 1833.4 | 863 | 7.48 |
| + incrementally updated page/slot sets | 17 | 1968.6 | 809 | 6.98 |
| + mixed forward as CUDA graphs (batch x token grid) | 17 | 2101.6 | 731 | 6.34 |
| + piecewise CUDA graphs above 128 tokens | 17 | 2037.1 | 758 | 6.50 |
| vLLM default (no `max_num_seqs` limit) | – | 4763 | – | – |

Notes:

- Default vLLM is ahead because of concurrency: it decodes all 64 requests at once, while
  our limit is 16. Decode is memory-bandwidth bound, so step time grows little with batch
  size and throughput grows almost proportionally.
- The prompts in this benchmark are shorter than the 256-token budget, so chunking never
  kicks in. The case chunked prefill helps with (a long prefill blocking decodes) does not
  occur here.
- Most of the slowdown that came with chunked prefill was host-side: free pages and slots
  were rebuilt from every slot on each access. Keeping them as incrementally updated sets
  recovered it and put the current version ahead of the one without chunked prefill.
- With these short prompts a step rarely exceeds 128 tokens: at `inf` rate the piecewise
  backend ran 5 of 404 steps (1 of 1500 at rate 8). The difference between the last two rows
  is within run-to-run variation, so this benchmark does not measure the piecewise path.

**Long prompts, higher concurrency**: 128 requests, prompts of 100-1045 tokens
(`--prompt-tokens 50 1000`), `token_budget=1024`, 32 concurrent requests. Ours
(`n_slots=33`) runs out of GPU memory during init while capturing the piecewise buckets:
10.4 GiB of the 12.9 GiB allocated sit in CUDA graph private pools (see Known gaps).
vLLM with `max_num_seqs=32`:

| rate | tok/s | ttft_mean | tpot_mean | e2e_mean | exact% | prefix% |
|---|---|---|---|---|---|---|
| 2 | 152.8 | 42.9 | 7.38 | 682 | 87.5 | 93.5 |
| 4 | 300.8 | 43.7 | 7.52 | 697 | 88.3 | 93.7 |
| 8 | 584.2 | 48.6 | 8.44 | 783 | 84.4 | 92.1 |
| inf | 1832.5 | 2427 | 15.12 | 3674 | 86.7 | 92.1 |

vLLM's fp16 outputs diverge from the HF greedy reference in 12-16% of the requests.

## Known gaps

- Every CUDA graph allocates its own private memory pool, and every piecewise `Graph` keeps
  its output alive. Memory grows with the number of graphs (about 13 per piecewise bucket), so
  `token_budget=1024` with 32 slots does not fit in 8 GiB. vLLM shares one pool across all
  captures, captures the largest sizes first and keeps only weak references to intermediate
  outputs.
- Init is slow: every bucket is a separate Dynamo trace and inductor compile, plus captures.
- The accuracy of our engine on the long-prompt workload is not measured yet (it does not get
  past init).
- The `ARCHITECTURE` title in `benchmark/report.py` still describes the old slot-KV
  design, so result files carry the wrong title.
- Concurrency is capped by the largest bucket (32).
- Greedy sampling only; no prefix caching.
