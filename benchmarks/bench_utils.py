# kernel benchmark'ları için ortak yardımcılar
import statistics

import torch

# cold ölçümde okunan toplam veri L2'nin bu katı kadar olur
COLD_FOOTPRINT_L2_MULTIPLE = 4

_cu_seqlens = {}


def single_seq_prefill(q, k, v, **cfg):
    # tek dizi: q (T, H_q, D), k/v (H_kv, cap, D). cu_seqlens T başına bir kez
    # ayrılıyor, ölçüme host->device kopyası girmesin
    from attn import prefill_gqa_attention
    T = q.shape[0]
    if T not in _cu_seqlens:
        _cu_seqlens[T] = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    return prefill_gqa_attention(q[None], k[None], v[None], T, _cu_seqlens[T], **cfg)[0]


def capture_graph(calls):
    # verilen çağrıları sırayla tek bir CUDA graph'a kaydeder
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        calls[0]()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def time_graph(graph, n_calls, replays=5):
    # graph'taki kernel başına süre (ms), replay'lerin median'ı
    times = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / n_calls)
    return statistics.median(times)
