# Nsight Compute ile profillemek için: GQA decode attention, tek konfigürasyon.
# proje kökünden: ncu ... python -m benchmarks.ncu_gqa
# gqa_sweep.py'de L=16384 için en iyi cold: BLOCK_N=16, num_warps=4, P=16.
# num_warps verilmiyor, triton'un varsayılanı 4.
# ilk WARMUP çağrı (triton derleme dahil) profillenmez, ncu'ya
# --launch-skip 2*WARMUP --launch-count 2 verilir (her çağrı load + merge = 2 kernel).
# ncu varsayılan olarak her kernel'den önce cache'leri boşaltır (--cache-control all),
# yani ölçüm cold.
import torch

from attn import decode_gqa_attention

L = 16384
H_Q, H_KV, D = 14, 2, 64
BLOCK_N = 16
P = 16
WARMUP = 3

if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn((H_Q, D), device="cuda", dtype=torch.float16)
    k = torch.randn((H_KV, L, D), device="cuda", dtype=torch.float16)
    v = torch.randn((H_KV, L, D), device="cuda", dtype=torch.float16)

    seq_len = torch.tensor([L], device="cuda", dtype=torch.int32)

    for _ in range(WARMUP + 1):
        o = decode_gqa_attention(q[None, None], k[None], v[None], seq_len, BLOCK_N=BLOCK_N, num_programs=P)
    torch.cuda.synchronize()
    print(f"L={L} BLOCK_N={BLOCK_N} P={P} out={tuple(o.shape)}")
