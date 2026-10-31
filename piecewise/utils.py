import torch
from torch.fx.passes.split_module import split_module


def is_attention(node) -> bool:
    # op, attn.core import edilince kayıt ediliyor
    return node.op == "call_function" and node.target is torch.ops.inference.gqa_mixed_attention.default


def create_partition(gm) -> dict:
    # graph parçaları çift, kv_lens gather + attention grupları tek numaralı
    part = 0
    ids = {}

    for node in gm.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue

        if is_attention(node):
            gather = node.kwargs["kv_lens"]
            part += 1
            ids[gather] = part
            ids[node] = part
            part += 1

        else:
            ids[node] = part

    return ids


def split(gm, ids):
    return split_module(gm, None, lambda node: ids[node], keep_original_order=True)


def is_graph_piece(submod) -> bool:
    return not any(is_attention(n) for n in submod.graph.nodes)


def weak_ref(x):
    if isinstance(x, torch.Tensor):
        storage = x.untyped_storage()
        storage = torch._C._construct_storage_from_data_pointer(
            storage.data_ptr(), x.device, storage.nbytes()
        )
        return torch.empty(0, dtype=x.dtype, device=x.device).set_(
            storage, x.storage_offset(), x.size(), x.stride()
        )
    if isinstance(x, (tuple, list)):
        return type(x)(weak_ref(v) for v in x)
    return x


def fake_inputs(submod) -> list:
    values = []
    for node in submod.graph.nodes:
        if node.op != "placeholder":
            continue
        if "example_value" not in node.meta:
            raise RuntimeError(f"{node.name}: meta['example_value'] yok")
        values.append(node.meta["example_value"])
    return values
