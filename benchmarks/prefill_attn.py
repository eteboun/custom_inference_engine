# prefill GQA attention: Triton kernel'i vs sdpa (causal, enable_gqa), doğruluk + süre.
# K/V KV cache düzeninde (H_kv, max_len, D), T'den sonraki satırlar çöp: kernel L'nin
# ötesini okursa sonuç bozulur. proje kökünden: python -m benchmarks.prefill_attn
import torch
import torch.nn.functional as F
import triton
from torch.profiler import profile, ProfilerActivity

from attn import prefill_gqa_attention

H_Q, H_KV, D = 14, 2, 64
EXTRA = 64   # cache'in T'den sonraki boş kısmı
CHECK_TS = [1, 7, 16, 17, 100, 1000, 4099]
TIME_TS = [1024, 4096, 16384]
BLOCK_MS = [16, 32]


def make(T):
    q = torch.randn((T, H_Q, D), device="cuda", dtype=torch.float16)
    k = torch.full((H_KV, T + EXTRA, D), 100.0, device="cuda", dtype=torch.float16)
    v = torch.full((H_KV, T + EXTRA, D), 100.0, device="cuda", dtype=torch.float16)
    k[:, :T] = torch.randn((H_KV, T, D), device="cuda", dtype=torch.float16)
    v[:, :T] = torch.randn((H_KV, T, D), device="cuda", dtype=torch.float16)
    return q, k, v


def reference(q, k, v):
    T = q.shape[0]
    o = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k[None, :, :T], v[None, :, :T],
        is_causal=True, enable_gqa=True)
    return o[0].transpose(0, 1)


if __name__ == "__main__":
    torch.manual_seed(0)

    # sdpa'nın seçtiği backend (Windows: mem-efficient cutlass, Linux/WSL: flash)
    q, k, v = make(4096)
    reference(q, k, v)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        reference(q, k, v)
        torch.cuda.synchronize()
    print("sdpa kernel'leri:", sorted({e.name[:70] for e in prof.events() if e.device_type.name == "CUDA"}))

    print("doğruluk (max abs fark):")
    for T in CHECK_TS:
        q, k, v = make(T)
        ref = reference(q, k, v)
        row = f"  T={T:>5}"
        for BM in BLOCK_MS:
            err = (prefill_gqa_attention(q, k, v, BLOCK_M=BM).float() - ref.float()).abs().max().item()
            row += f"  BM={BM}: {err:.1e}"
        print(row)

    print()
    print("süre (ms, warm, triton.testing.do_bench):")
    for T in TIME_TS:
        q, k, v = make(T)
        row = f"  T={T:>5}  sdpa {triton.testing.do_bench(lambda: reference(q, k, v)):7.3f}"
        for BM in BLOCK_MS:
            t = triton.testing.do_bench(lambda: prefill_gqa_attention(q, k, v, BLOCK_M=BM))
            row += f"  BM={BM} {t:7.3f}"
        print(row)
