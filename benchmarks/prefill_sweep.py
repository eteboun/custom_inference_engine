# prefill GQA attention kernel'i için BLOCK_M x BLOCK_N x num_warps taraması, sdpa'ya oranla.
# proje kökünden: python -m benchmarks.prefill_sweep
import itertools
import math

import torch
import torch.nn.functional as F
import triton

from attn import prefill_gqa_attention

H_Q, H_KV, D = 14, 2, 64
TS = [1024, 4096, 16384]
BLOCK_MS = [8, 16, 32, 64]
BLOCK_NS = [16, 32, 64, 128]
WARPS = [4, 8]


def make(T):
    q = torch.randn((T, H_Q, D), device="cuda", dtype=torch.float16)
    k = torch.randn((H_KV, T + 64, D), device="cuda", dtype=torch.float16)
    v = torch.randn((H_KV, T + 64, D), device="cuda", dtype=torch.float16)
    return q, k, v


def reference(q, k, v):
    T = q.shape[0]
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k[None, :, :T], v[None, :, :T],
        is_causal=True, enable_gqa=True)
    return o[0].transpose(0, 1)


torch.manual_seed(0)
data = {T: make(T) for T in TS}
# sdpa referansı: ölçümler arasında oynamasın diye birkaç kez, median
sdpa = {T: sorted(triton.testing.do_bench(lambda: reference(*data[T])) for _ in range(3))[1] for T in TS}
print("sdpa (flash) ms:", {T: round(t, 3) for T, t in sdpa.items()})

q_chk, k_chk, v_chk = make(1000)
ref_chk = reference(q_chk, k_chk, v_chk)

results = []
for BM, BN, W in itertools.product(BLOCK_MS, BLOCK_NS, WARPS):
    cfg = dict(BLOCK_M=BM, BLOCK_N=BN, num_warps=W)
    try:
        err = (prefill_gqa_attention(q_chk, k_chk, v_chk, **cfg).float() - ref_chk.float()).abs().max().item()
        times = {T: triton.testing.do_bench(lambda: prefill_gqa_attention(*data[T], **cfg)) for T in TS}
    except Exception as e:
        print(f"BM={BM:>3} BN={BN:>3} W={W}: HATA {type(e).__name__}: {str(e).splitlines()[0][:80]}")
        continue
    results.append((cfg, err, times))
    print(f"BM={BM:>3} BN={BN:>3} W={W}: err {err:.1e}  " +
          "  ".join(f"T={T} {times[T]:7.3f} ({times[T] / sdpa[T]:.2f}x)" for T in TS), flush=True)

print()
for T in TS:
    best = min(results, key=lambda r: r[2][T])
    print(f"T={T:>5} en iyi: {best[0]}  {best[2][T]:.3f} ms  sdpa {sdpa[T]:.3f} ms  ({best[2][T] / sdpa[T]:.2f}x)")
# bütün T'lerde birlikte en iyi tek config: sdpa'ya oranların geometrik ortalaması
best_all = min(results, key=lambda r: math.prod(r[2][T] / sdpa[T] for T in TS))
print("tek config olarak en iyi:", best_all[0],
      {T: f"{best_all[2][T]:.3f} ({best_all[2][T] / sdpa[T]:.2f}x)" for T in TS})
