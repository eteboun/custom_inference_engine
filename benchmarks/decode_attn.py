# decode GQA attention mikro benchmark'ı: Triton kernel'i vs SDPA (varsayılan backend
# ve cuDNN), warm/cold, CUDA graph içinde. proje kökünden: python -m benchmarks.decode_attn
import contextlib
import random
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import triton

from attn import decode_gqa_attention
from benchmarks.bench_utils import capture_graph, time_graph, COLD_FOOTPRINT_L2_MULTIPLE


TORCH_BACKENDS = {
    # varsayılan: PyTorch'un seçtiği backend (bu ortamda mem-efficient cutlass),
    # model sdpa ile çalıştığı için modelde kullanılan bu
    "sdpa": None,
    # bu GPU'da decode için en hızlı torch backend'i
    "cudnn": SDPBackend.CUDNN_ATTENTION,
}


def benchmark(L, rng, H_q=14, H_kv=2, D=64, BLOCK_N=64, rounds=5):
    # Qwen2.5-0.5B: 14 q head, 2 kv head (7 q head aynı K/V'yi paylaşıyor), head_dim 64
    #
    # ölçülenler:
    # triton: decode_gqa_attention, num_programs = min(SM sayısı, blok sayısı), grid (P, H_kv)
    # sdpa: F.scaled_dot_product_attention(enable_gqa=True), varsayılan backend
    # cudnn: aynı çağrı, cuDNN backend'ine zorlanmış
    #
    # bench_utils.py'deki graph yardımcılarıyla:
    # cold: n_sets farklı (q, k, v), toplamı L2'nin birkaç katı, graph hepsini
    # random sırayla çalıştırır, her çağrı K/V'yi DRAM'den okur.
    # warm: aynı sayıda çağrı ama hep ilk set, K/V L2'de (L2'ye sığdığı sürece).
    # ikisi de CUDA graph, aradaki tek fark cache durumu
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    set_bytes = (2 * H_kv * L + H_q) * D * torch.finfo(torch.float16).bits // 8
    n_sets = triton.cdiv(COLD_FOOTPRINT_L2_MULTIPLE * l2_bytes, set_bytes)
    # her SM'e bir parça, blok sayısından fazla değil
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    num_programs = min(sm_count, triton.cdiv(L, BLOCK_N))

    qs = torch.randn((n_sets, H_q, D), device="cuda", dtype=torch.float16)
    ks = torch.randn((n_sets, H_kv, L, D), device="cuda", dtype=torch.float16)
    vs = torch.randn((n_sets, H_kv, L, D), device="cuda", dtype=torch.float16)

    # kernel geçerli uzunluğu GPU'dan okuyor, tensor bir kez ayrılıyor
    # (çağrı içinde ayrılsaydı her çağrıya bir fill kernel'i eklenirdi)
    seq_len = torch.tensor([L], device="cuda", dtype=torch.int32)

    def run_triton(i):
        # kernel batch'li: B = 1
        return decode_gqa_attention(
            qs[i][None], ks[i][None], vs[i][None], seq_len,
            BLOCK_N=BLOCK_N, num_programs=num_programs
        )[0]

    def make_torch(backend):
        def run(i):
            # (batch, heads, seq, D) şekline view, kopya yok
            ctx = sdpa_kernel(backend) if backend else contextlib.nullcontext()
            with ctx:
                return F.scaled_dot_product_attention(
                    qs[i].view(1, H_q, 1, D),
                    ks[i].view(1, H_kv, L, D),
                    vs[i].view(1, H_kv, L, D),
                    enable_gqa=True,
                ).view(H_q, D)
        return run

    impls = {"triton": run_triton}
    impls.update({name: make_torch(b) for name, b in TORCH_BACKENDS.items()})

    for _ in range(10):
        for run in impls.values():
            run(0)
    torch.cuda.synchronize()

    # Correctness: torch'a ve fp32 referansa göre. referansta her q head kendi
    # kv head'ini kullanır (kv head'ler GROUP_SIZE kez tekrarlanır)
    outs = {name: run(0) for name, run in impls.items()}
    group = H_q // H_kv
    q32 = qs[0].float()
    k32 = ks[0].float().repeat_interleave(group, dim=0)
    v32 = vs[0].float().repeat_interleave(group, dim=0)
    scores = torch.einsum("hld,hd->hl", k32, q32) * D ** -0.5
    ref = torch.einsum("hl,hld->hd", torch.softmax(scores, dim=-1), v32)
    for name in TORCH_BACKENDS:
        torch.testing.assert_close(outs["triton"], outs[name], rtol=1e-2, atol=1e-2)
    err = {name: (out.float() - ref).abs().max().item() for name, out in outs.items()}

    cold_order = list(range(n_sets))
    rng.shuffle(cold_order)
    orders = {"warm": [0] * n_sets, "cold": cold_order}

    graphs = {}
    for name, run in impls.items():
        for cache, order in orders.items():
            graphs[(name, cache)] = capture_graph(
                [lambda i=i, run=run: run(i) for i in order]
            )

    # her (kernel, cache) rounds kez ölçülür ve ölçüm sırası karıştırılır
    schedule = [key for key in graphs for _ in range(rounds)]
    rng.shuffle(schedule)

    samples = {key: [] for key in graphs}
    for key in schedule:
        samples[key].append(time_graph(graphs[key], n_sets))

    us = {key: statistics.median(s) * 1000 for key, s in samples.items()}

    # kv head'lerin K ve V'si okunur (q, o ve ara sonuçlar ihmal edilebilir).
    # aynı kv head'i okuyan q head'lerin tekrar okumaları sayılmıyor
    kv_bytes = 2 * H_kv * L * D * qs.element_size()

    def bandwidth(t_us):
        return kv_bytes / (t_us / 1e6) / 1e9

    return {
        "L": L,
        "num_programs": num_programs,
        "n_sets": n_sets,
        "kv_mb": kv_bytes / 2**20,
        "us": us,
        "gbps": {key: bandwidth(t) for key, t in us.items()},
        "err": err,
    }


if __name__ == "__main__":
    seed = 0
    rng = random.Random(seed)
    H_q, H_kv = 14, 2
    D = 64
    BLOCK_N = 64
    rounds = 5
    l2_mb = torch.cuda.get_device_properties(0).L2_cache_size / 2**20
    names = ["triton", *TORCH_BACKENDS]

    # main.py/test.py'deki prompt uzunluklarına yakın
    seq_lens = [64, 256, 1024, 2048, 4096, 8192, 16384]
    order = seq_lens[:]
    rng.shuffle(order)
    print(f"seed: {seed} | L order: {order}")
    print(f"decode GQA attention, {H_q} q / {H_kv} kv head, 1 query | D={D} | "
          f"BLOCK_N={BLOCK_N} | rounds: {rounds} (median) | cuda graph")

    results = {
        L: benchmark(L, rng, H_q=H_q, H_kv=H_kv, D=D, BLOCK_N=BLOCK_N, rounds=rounds)
        for L in order
    }

    print()
    print("triton: grid (P, H_kv), sdpa: varsayılan backend, cudnn: cuDNN backend, süreler us")
    print(f"K+V MB > L2 ({l2_mb:.0f} MB) olan satırlarda warm da DRAM'den okuyor (*)")
    print("-" * 113)
    print(
        f"{'L':>6}{'P':>4}{'K+V MB':>9}"
        + "".join(
            f"{cache + ' triton':>13}{'sdpa':>9}{'cudnn':>9}{'sdpa/tri':>10}{'cudnn/tri':>11}"
            for cache in ["warm", "cold"]
        )
    )
    print("-" * 113)
    for L in seq_lens:
        r = results[L]
        mark = "*" if r["kv_mb"] > l2_mb else " "
        row = f"{L:>6}{r['num_programs']:>4}{r['kv_mb']:>8.1f}{mark}"
        for cache in ["warm", "cold"]:
            tri, sdpa, cudnn = (r["us"][(name, cache)] for name in names)
            row += f"{tri:>13.2f}{sdpa:>9.2f}{cudnn:>9.2f}{sdpa / tri:>9.2f}x{cudnn / tri:>10.2f}x"
        print(row)
    print("-" * 113)
    print("oran > 1: triton daha hızlı")

    print()
    print("correctness: assert_close(triton, sdpa ve cudnn, rtol=1e-2, atol=1e-2) hepsi geçti")
    print(f"{'L':>6}{'cold GB/s':>12}{'':>16}{'max abs err (fp32 referansa göre)':>36}")
    print(f"{'':>6}{'triton':>12}{'sdpa':>8}{'cudnn':>8}{'triton':>12}{'sdpa':>12}{'cudnn':>12}")
    for L in seq_lens:
        r = results[L]
        gbps, err = r["gbps"], r["err"]
        print(
            f"{L:>6}"
            + "".join(f"{gbps[(name, 'cold')]:>{12 if name == 'triton' else 8}.0f}" for name in names)
            + "".join(f"{err[name]:>12.1e}" for name in names)
        )
