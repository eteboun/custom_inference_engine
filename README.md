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
| `kv_cache/storage.py` | Paged KV pool, block table, per-slot length and last generated token |
| `scheduler/scheduler.py` | Thread-based scheduler: queue, token budget, page allocation, preemption |
| `scheduler/models.py` | Per-request state kept by the scheduler (`Request`) |
| `scheduler/decode_module.py` | Pure decode forward captured as a CUDA graph |
| `scheduler/mixed_module.py` | Varlen (prefill + decode) forward compiled with `torch.compile` |
| `scheduler/bucket.py` | One decode module and one mixed module per batch size |
| `benchmark/` | Poisson-arrival serving benchmark (`python -m benchmark`) |
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

For each batch-size bucket (1, 2, 4, 8, 16), two modules are built and warmed up at init:

- `DecodeModule`: the fixed-shape decode forward is captured as a CUDA graph; each step
  only updates the input buffers and replays the graph.
- `MixedModule`: the token count varies, so there is no graph; it is compiled with
  `torch.compile` using dynamic shapes.

If even one request in a step is still prefilling, the mixed path is used; otherwise the
decode path.

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
`--max-num-seqs` (vLLM only). Each run is saved to `results/{timestamp}_{impl}.json` with
the table, arguments, environment, git commit and per-request records.

`n_slots` can be at most one more than the largest bucket (16); slot 0 is the padding slot,
so at most 16 requests run at once.

## Results

64 requests, `max_new_tokens=128`, prompts of 30–50 tokens. Times are in ms; the `inf` row
is all requests arriving at once.

**Against vLLM at the same concurrency (16 requests)**: ours with `n_slots=17` (paged,
before chunked prefill), vLLM with `max_num_seqs=16`:

| rate | tok/s (ours / vLLM) | ttft_mean | tpot_mean | e2e_mean |
|---|---|---|---|---|
| 2 | 146.4 / 146.0 | 16.0 / 38.4 | 5.93 / 7.24 | 512 / 645 |
| 4 | 288.4 / 284.8 | 16.2 / 39.7 | 5.99 / 7.38 | 519 / 657 |
| 8 | 551.9 / 543.0 | 16.4 / 40.1 | 6.35 / 7.79 | 550 / 693 |
| inf | 1885.8 / 1490.7 | 830 / 1083 | 7.12 / 8.74 | 1426 / 1818 |

**Across versions, at `inf` rate:**

| version | n_slots | tok/s | ttft_mean | tpot_mean |
|---|---|---|---|---|
| fixed per-slot KV | 5 | 644 | – | ~5.8 |
| paged KV | 5 | 586.0 | 4064 | 6.50 |
| paged KV | 17 | 1885.8 | 830 | 7.12 |
| paged KV + chunked prefill (`token_budget=256`) | 17 | 1833.4 | 863 | 7.48 |
| vLLM default (no `max_num_seqs` limit) | – | 4763 | – | – |

Notes:

- Default vLLM is ahead because of concurrency: it decodes all 64 requests at once, while
  our limit is 16. Decode is memory-bandwidth bound, so step time grows little with batch
  size and throughput grows almost proportionally.
- The prompts in this benchmark are shorter than the 256-token budget, so chunking never
  kicks in. The case chunked prefill helps with (a long prefill blocking decodes) does not
  occur here; what remains is the extra host-side work and the cost of decodes taking the
  graph-less path in mixed steps.

## Known gaps

- The `ARCHITECTURE` title in `benchmark/report.py` still describes the old slot-KV
  design, so result files carry the wrong title.
- Concurrency is capped by the largest bucket (16).
- The mixed forward does not use a CUDA graph; in every step with a prefill, decodes go
  through it too.
- No profiling yet on whether short prefills should take a separate prefill + decode path
  instead of mixed, or on reducing host-side cost.
- The benchmark has no long-prompt workload.
- `_empty_pages` rebuilds the page sets of all slots on every call.
- Greedy sampling only; no prefix caching.
