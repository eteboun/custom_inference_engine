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
| `scheduler/models.py` | Per-request state and results (`Request`, `RequestResult`), and init and run statistics (`Tracker`) |
| `scheduler/decode_module.py` | Pure decode forward captured as a CUDA graph |
| `scheduler/mixed_module.py` | Varlen (prefill + decode) forward: `FullMixedModule` (whole-forward CUDA graph) and `PiecewiseMixedModule` (piecewise CUDA graphs) |
| `scheduler/graph_backend.py` | `FullGraphBackend` (batch size x token count grid) and `PiecewiseGraphBackend` (token count buckets) |
| `benchmark/` | Serving benchmark against vLLM (`python -m benchmark`): workloads, clients, metrics, HF greedy reference, reports |
| `feature_bench/` | Benchmarks of individual features, separate from the vLLM comparison; results in `feature_bench/results/` as JSON and HTML |
| `vllm_baseline/` | vLLM engine for the same model and benchmark |
| `results/` | One timestamped folder per benchmark run; `results/reference/` caches HF outputs. The loose JSON files and `results/obsolete/` come from the previous benchmark |
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
    (16, 32, 64, 128). The threshold comes from the step-time measurement in Results.

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
  get the real batch size each step and are marked dynamic. Each bucket has two Dynamo
  entries, one for `B >= 2` and one for `B = 1` (0/1 specialization); both are created at init
  by warming up with several sequences and then with one. The `B = 1` entry reuses the
  captured piece graphs of the `B >= 2` entry: graphs are cached per bucket, keyed by piece
  name, generated code and input shapes/strides/dtypes, and all 25 pieces match.

Decode-only steps use `DecodeModule`. Steps with a prefill use the full graph backend when the
token count is at most 128, the piecewise backend otherwise. After the replay, KV lengths
are advanced and `lm_head` is applied to the last token of each sequence outside the graph.

**CUDA graph memory**: all captures (piecewise pieces, full mixed and decode graphs) share one
memory pool (`torch.cuda.graph_pool_handle()`). Only one forward runs at a time, so
intermediate buffers of different graphs can live at the same addresses. Piece, full mixed
and decode graphs keep only a non-owning reference (`weak_ref` in `piecewise/utils.py`) to their
output, so its block returns to the pool once the forward that captured it no longer needs it. Buckets are captured
largest first (piecewise, then full mixed, then decode), so smaller captures fit into blocks
freed by larger ones.

**Dynamo config is thread-local**: `Scheduler` raises `recompile_limit` in the thread that
builds it, while `loop()` runs in another thread with the default limit of 8. The code object
`_forward` already has ~50 entries after init, so any compile inside the loop thread fails
under `fullgraph=True`. This is why every entry, including `B = 1`, is compiled at init.

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

GPU runs are done under WSL. One command runs both engines, each in its own process:

```bash
python -m benchmark --title "<architecture of this run>"
bash benchmark/smoke_test.sh    # short check of the whole path
```

Each run creates a folder:

```text
results/<timestamp>/
  report.html        # settings, init, results of both engines side by side, per-run details
  gpu.csv            # nvidia-smi samples every 5 s: temperature, power, clocks, memory
  results_ours/      # one JSON per configuration, with per-request records
  results_vllm/
```

Main arguments: `--title` (required), `--engines` (`ours vllm`), `--kinds` (`short long mixed`),
`--rates` (`4 8 inf` requests/s), `--repeats` (3), `--n-requests` (500), `--trim` (0.05),
`--warmup-requests` (16), `--max-new-tokens` (128), `--n-slots` (33), `--page-size` (16),
`--token-budget` (1024). vLLM gets `max_num_seqs = n_slots - 1` and
`max_num_batched_tokens = token_budget`.

`n_slots` can be at most one more than the largest bucket (32); slot 0 is the padding slot,
so at most 32 requests run at once. `token_budget` must be above 128 and a multiple of 32.

**Protocol**:

- Workloads: `short` (30-50 token prompts), `long` (500-1000 tokens), `mixed` (half of each,
  shuffled). Long prompts are questions with a context built from a fixed set of paragraphs.
- Arrivals are open-loop Poisson. Prompts are fixed by the seed; arrivals change with each
  repeat. At `rate=inf` all requests arrive at once, so repeats see identical arrivals.
- Each engine is built once, warmed up once, and then serves every configuration in the same
  seeded random order.
- Latency and throughput skip the first and last 5% of requests by planned arrival time,
  except at `rate=inf`.
- Latency is measured from the planned arrival. `send_lag` shows how late the client sent a
  request, and `ttft_engine` excludes it.

**Accuracy**: before serving, each prompt is generated once with HF (batch 1, greedy,
`repetition_penalty=1.0`) and cached in `results/reference/`. Reports give the number of
requests identical to the reference (`exact`) and the mean share of tokens matching up to the
first difference (`match ratio`), using every request; trailing EOS tokens are ignored. The
reference is not ground truth: it is also fp16, so a difference is not necessarily an error.

### Feature benchmarks (`feature_bench/`)

Each benchmark writes a JSON report; `html_report` turns reports into HTML tables:

```bash
python -m feature_bench.piecewise_vs_full --title "<what this run tests>"
python -m feature_bench.step_time --title "<what this run tests>" --token-counts 64 128 256
python -m feature_bench.html_report feature_bench/results/<report>.json
```

- `piecewise_vs_full`: runs the same inputs through full graph and piecewise mixed modules and
  compares the logits.
- `step_time`: times `replay()` of both modules for each batch size and token count, with one
  process per token count to keep memory use low.

## Results

### Serving benchmark against vLLM

Run `results/20261102-142549`, commit `984f70d`: 500 requests per configuration,
`max_new_tokens=128`, 32 concurrent requests, `token_budget=1024`. Values are `ours | vLLM`,
the mean of 3 repeats; times are in ms.

| workload | rate | tok/s | ttft_overall | tpot | e2e mean | e2e p99 | exact (of 500) |
|---|---|---|---|---|---|---|---|
| short | 4 | 360 \| 360 | 11 \| 32 | 5.97 \| 7.37 | 540 \| 682 | 823 \| 1041 | 432 \| 437 |
| short | 8 | 715 \| 712 | 11 \| 33 | 6.06 \| 7.65 | 548 \| 710 | 833 \| 1126 | 428 \| 433 |
| short | inf | 4036 \| 2432 | 4867 \| 8201 | 7.53 \| 12.29 | 5533 \| 9295 | 10884 \| 18184 | 434 \| 419 |
| long | 4 | 331 \| 331 | 30 \| 48 | 6.72 \| 7.93 | 571 \| 692 | 981 \| 1200 | 441 \| 439 |
| long | 8 | 657 \| 653 | 32 \| 56 | 7.72 \| 9.35 | 654 \| 809 | 1193 \| 1514 | 442 \| 441 |
| long | inf | 1957 \| 1665 | 10353 \| 12129 | 15.71 \| 18.08 | 11638 \| 13618 | 21380 \| 25148 | 442 \| 441 |
| mixed | 4 | 351 \| 350 | 21 \| 39 | 6.31 \| 7.54 | 562 \| 686 | 931 \| 1113 | 428 \| 432 |
| mixed | 8 | 698 \| 695 | 22 \| 42 | 6.84 \| 8.20 | 608 \| 743 | 1043 \| 1258 | 427 \| 427 |
| mixed | inf | 2658 \| 2123 | 7670 \| 9698 | 11.59 \| 14.40 | 8646 \| 10908 | 16118 \| 20271 | 430 \| 427 |

- At `rate=inf`, ours has higher throughput: 66% on `short`, 25% on `mixed`, 18% on `long`. The
  ranges of the 3 repeats do not overlap.
- At rates 4 and 8, throughput equals the offered load on both engines. Ours has lower tpot
  (about 15-20%) and e2e (about 15-25%).
- Part of the TTFT difference comes from the client: vLLM's single-threaded loop can only submit
  between steps, so its `send_lag` p99 is 15-28 ms against about 2 ms for ours. The difference
  remains in `ttft_engine` (for example 10.2 | 28.1 ms on `short` at rate 4).
- Both engines match the HF reference on about 84-89% of requests, with no consistent winner
  except `short` at `inf`.
- There were no preemptions. The piecewise backend ran almost all mixed steps on `long`, about
  half on `mixed`, and few on `short`.
- Init: ours 462 s with a warm inductor cache, vLLM 44 s. PyTorch reserved 2180 MiB for ours:
  model 950 MiB, KV cache 385 MiB, graphs and the shared attention buffer 263 MiB.
- GPU: at most 77 C and 79 W. Under load the median SM clock was 2475 MHz, but some samples
  dropped to about 1080 MHz with a power-cap flag, and these are not matched to runs yet.

These numbers come from one laptop GPU under WSL, with ours always running first, and from
synthetic workloads on a small model. They compare the two engines under these conditions and
do not predict production behavior or other models.

### Earlier measurements (previous benchmark)

These come from the previous benchmark code, which is no longer in the repository; its result
files are in `results/` and `results/obsolete/`.

> These are not strict or carefully designed benchmarks. They are temporary measurements
> taken while developing individual features, to check that a change works and roughly how
> it affects performance. Each configuration is a single run on a laptop GPU under WSL, with
> no repeated runs, no confidence intervals and no control over background load, and the
> workloads are small and synthetic. Treat the numbers as rough indications, not as a
> performance claim.

64 requests, `max_new_tokens=128`, prompts of 30–50 tokens. Times are in ms; the `inf` row
is all requests arriving at once.

**Against vLLM at the same concurrency (16 requests)**: ours with `n_slots=17` and
`token_budget=256` (measured before the shared memory pool, not re-run with the current code),
vLLM with `max_num_seqs=16`:

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
  recovered it and put that version ahead of the one without chunked prefill.
- With these short prompts a step rarely exceeds 128 tokens: at `inf` rate the piecewise
  backend ran 5 of 404 steps (1 of 1500 at rate 8). The difference between the last two rows
  is within run-to-run variation, so this benchmark does not measure the piecewise path.

**Long prompts, higher concurrency**: 128 requests, prompts of 100-1045 tokens
(`--prompt-tokens 50 1000`), `token_budget=1024`, 32 concurrent requests: ours with
`n_slots=33`, vLLM with `max_num_seqs=32` and `max_num_batched_tokens=1024`:

| rate | tok/s (ours / vLLM) | ttft_mean | tpot_mean | e2e_mean | exact% | prefix% |
|---|---|---|---|---|---|---|
| 2 | 151.7 / 152.8 | 24.4 / 42.9 | 6.26 / 7.38 | 560 / 682 | 86.7 / 87.5 | 92.9 / 93.5 |
| 4 | 301.8 / 300.8 | 29.3 / 43.7 | 6.79 / 7.52 | 620 / 697 | 88.3 / 88.3 | 93.8 / 93.7 |
| 8 | 590.3 / 584.2 | 26.1 / 48.6 | 6.96 / 8.44 | 629 / 783 | 88.3 / 84.4 | 93.5 / 92.1 |
| inf | 2117.0 / 1832.5 | 2069 / 2427 | 13.20 / 15.12 | 3154 / 3674 | 88.3 / 86.7 | 92.9 / 92.1 |

Notes:

- The whole run of ours (init, warmup and the four rates) took 724 s.
- Both engines match the HF greedy reference exactly in 84-88% of the requests; the rest
  diverge at some point because of fp16 numerics in batched execution.
- The vLLM numbers come from a run a few hours earlier with the same settings.
- Before the shared memory pool, ours ran out of GPU memory during init with this
  configuration: 10.4 GiB of the 12.9 GiB allocated were in per-graph private pools.

**Full graph vs piecewise step time** (for the token threshold): for each batch bucket
(1-32) and token count (16-512, `T >= B`), `FullMixedModule` and `PiecewiseMixedModule`
replay the same inputs: `B` sequences sharing `T` tokens, each with 256 tokens already in
KV. Only `replay()` is timed (median of 7 x 50 iterations); each token count runs in its own
process. Ranges are min-max over the batch buckets; rows marked (median) give the median overhead:

| T | full (ms) | piecewise (ms) | piecewise overhead |
|---|---|---|---|
| 16-64 | 3.8-4.5 | 5.9-6.9 | 47-77% |
| 128 | 5.3-5.7 | 6.3-6.7 | 13-28% |
| 132-156 | 6.3-7.1 | 6.5-7.6 | ~4-5% (median) |
| 160 | 6.7-7.4 | 6.9-7.5 | ~3% (median) |
| 192 | 6.9-7.2 | 7.2-7.5 | 3-4% |
| 256-512 | 9.0-14.1 | 9.0-14.1 | mostly < 2% |

- Piecewise has a floor of about 6.5 ms up to 128 tokens, independent of the token count:
  the step is bound by host-side submission (24 eager attention groups, buffer copies,
  Python), and the GPU waits for it.
- Full graph time jumps by about 1 ms between 128 and 132 tokens and is flat from 132 to
  148. Past 128 its GPU time exceeds the piecewise floor, the host work is hidden behind
  GPU work, and the overhead drops to 4-5%.
- The batch size has little effect on the overhead above 128 tokens.
- This is why steps of up to 128 tokens use the full graph and larger ones the piecewise
  backend. Raw numbers and tables are in `feature_bench/results/`.

## Known gaps

- The token threshold was measured with a fixed KV context of 256 tokens and timing only
  the replay; the 0.1-0.3 ms that remains above 128 tokens is not broken down.
- `PIECEWISE_TOKEN_STEP = 32` is not based on a measurement.
- Any compile inside the scheduler thread fails with `FailOnRecompileLimitHit`, so an
  unexpected guard failure during serving stops the loop.
- Ours always runs before vLLM, so an order effect is not measured. `gpu.csv` is not yet matched
  to individual runs, and vLLM's own memory use is only visible there.
- Reports compare repeats row by row; there is no summary across repeats.
- Init is slow: every bucket is a separate Dynamo trace and inductor compile, plus captures.
  `torch.cuda.graph` also runs `torch.cuda.synchronize()` and `empty_cache()` on every capture
  (`gc.collect()` only if `torch.compiler.config.force_cudagraph_gc` is set, off by default);
  their share of init time is not measured.
- Concurrency is capped by the largest bucket (32).
- Greedy sampling only; no prefix caching.
