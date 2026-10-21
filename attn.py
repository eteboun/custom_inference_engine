import torch
from torch.library import triton_op, wrap_triton
import triton

from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb

from kernels.decode_attention import (
    decode_gqa_attention_kernel_load, decode_attention_kernel_merge
)
from kernels.prefill_attention import prefill_gqa_attention_kernel


def decode_gqa_attention(q, k, v, lens, slot_ids, num_programs, BLOCK_N=64, num_warps=4,
                         wrap=lambda kernel: kernel):
    total_T, H_q, D = q.shape[1:]
    S, H_kv, cap, _ = k.shape
    B = slot_ids.shape[0]
    assert H_q % H_kv == 0
    assert lens.shape == (B,)
    assert total_T == B
    assert q.is_contiguous()
    assert k.stride() == v.stride() and k.stride(3) == 1 and k.stride(2) == D
    max_kv_offset = (S - 1) * k.stride(0) + (H_kv - 1) * k.stride(1) + cap * D
    assert max_kv_offset < 2**31, "K/V ofsetleri int32'ye sığmıyor"
    assert BLOCK_N >= 16, "tl.dot için BLOCK_N >= 16 olmalı"
    P = num_programs
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_G = max(16, triton.next_power_of_2(H_q // H_kv))

    partial_m = torch.empty((B, H_q, P), device=q.device, dtype=torch.float32)
    partial_l = torch.empty((B, H_q, P), device=q.device, dtype=torch.float32)
    partial_acc = torch.empty((B, H_q, P, D), device=q.device, dtype=torch.float32)
    assert partial_acc.numel() < 2**31, "ara sonuç ofsetleri int32'ye sığmıyor"
    o = torch.empty_like(q)

    wrap(decode_gqa_attention_kernel_load)[(P, H_kv, B)](
        q, k, v,
        lens,
        slot_ids,
        partial_m, partial_l, partial_acc,
        k.stride(0),
        k.stride(1),
        D=D,
        Q_HEADS=H_q,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_G=BLOCK_G,
        GROUP_SIZE=H_q // H_kv,
        num_warps=num_warps,
    )

    wrap(decode_attention_kernel_merge)[(H_q, B)](
        o,
        partial_m, partial_l, partial_acc,
        P=P,
        D=D,
        Q_HEADS=H_q,
        BLOCK_P=triton.next_power_of_2(P),
        BLOCK_D=BLOCK_D,
    )

    return o


def prefill_gqa_attention(q, k, v, max_seqlen: int, cu_seqlens, slot_ids, BLOCK_M=16, BLOCK_N=64, num_warps=4,
                          wrap=lambda kernel: kernel):
    total_T, H_q, D = q.shape[1:]
    _, H_kv, cap, _ = k.shape
    B = slot_ids.shape[0]
    assert H_q % H_kv == 0
    assert cu_seqlens.shape == (B + 1,)
    assert k.shape == v.shape and k.shape[3] == D and total_T <= B * cap
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    GROUP_SIZE = H_q // H_kv
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_G = triton.next_power_of_2(GROUP_SIZE)
    assert BLOCK_M * BLOCK_G >= 16 and BLOCK_N >= 16, "tl.dot için boyutlar >= 16 olmalı"

    o = torch.empty_like(q)

    wrap(prefill_gqa_attention_kernel)[(triton.cdiv(max_seqlen, BLOCK_M), H_kv, B)](
        q, k, v,
        o,
        CAP=cap,
        CU_SEQLENS=cu_seqlens,
        SLOT_IDS=slot_ids,
        D=D,
        KV_HEADS=H_kv,
        Q_HEADS=H_q,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_G=BLOCK_G,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=num_warps
    )

    return o


GQA_DECODE_CONFIG = {"BLOCK_N": 16, "num_warps": 4, "num_programs": 16}

GQA_PREFILL_CONFIG = {"BLOCK_M": 16, "BLOCK_N": 64, "num_warps": 4}


@triton_op("inference::gqa_decode_attention", mutates_args=())
def gqa_decode_attention_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lens: torch.Tensor, slot_ids: torch.Tensor
) -> torch.Tensor:
    return decode_gqa_attention(q, k, v,
                                lens=lens, slot_ids=slot_ids,
                                **GQA_DECODE_CONFIG, wrap=wrap_triton)

@triton_op("inference::gqa_prefill_attention", mutates_args=())
def gqa_prefill_attention_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, max_seqlen: int, cu_seqlens: torch.Tensor, slot_ids: torch.Tensor
) -> torch.Tensor:
    return prefill_gqa_attention(q, k, v, **GQA_PREFILL_CONFIG,
                                 max_seqlen=max_seqlen, cu_seqlens=cu_seqlens,
                                 slot_ids=slot_ids, wrap=wrap_triton)

class TritonQwen2Attention(torch.nn.Module):
    def __init__(self, original_attn):
        super().__init__()

        assert original_attn.sliding_window is None, "sliding window desteklenmiyor"
        self.original = original_attn
        self.layer_idx = original_attn.layer_idx
        self.num_heads = original_attn.config.num_attention_heads
        self.head_dim = original_attn.head_dim
        assert original_attn.scaling == self.head_dim ** -0.5

        projs = [original_attn.q_proj, original_attn.k_proj, original_attn.v_proj]
        assert all(p.bias is not None for p in projs)
        self.qkv_sizes = [p.out_features for p in projs]
        with torch.no_grad():
            self.register_buffer(
                "qkv_weight", torch.cat([p.weight for p in projs]), persistent=False
            )
            self.register_buffer(
                "qkv_bias", torch.cat([p.bias for p in projs]), persistent=False
            )
        self.fuse_qkv = True

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, kv_manager=None, **kwargs):
        if kv_manager is None:
            return self.original(
                hidden_states, position_embeddings, attention_mask,
                past_key_values=past_key_values, **kwargs
            )

        # hidden_states: (1, totalT, hidden_size)

        attn = self.original
        total_T = hidden_states.shape[1]
        hidden_shape = (1, total_T, -1, self.head_dim)
        if self.fuse_qkv:
            q, k, v = torch.nn.functional.linear(
                hidden_states, self.qkv_weight, self.qkv_bias
            ).split(self.qkv_sizes, dim=-1)
        else:
            q, k, v = attn.q_proj(hidden_states), attn.k_proj(hidden_states), attn.v_proj(hidden_states)

        # (1, total_T, H, D)
        query_states = q.view(hidden_shape)
        key_states = k.view(hidden_shape)
        value_states = v.view(hidden_shape)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states,
            cos, sin,
            unsqueeze_dim=2
        )

        kv_manager.update(
            self.layer_idx, key_states, value_states
        )

        if kv_manager.is_decode:
            o = gqa_decode_attention_op(
                query_states,
                kv_manager.keys[self.layer_idx],
                kv_manager.values[self.layer_idx],
                kv_manager.pos + 1,
                slot_ids=kv_manager.slot_ids
            )

        else:
            o = gqa_prefill_attention_op(
                query_states,
                kv_manager.keys[self.layer_idx],
                kv_manager.values[self.layer_idx],
                max_seqlen=kv_manager.max_seqlen, cu_seqlens=kv_manager.cu_seqlens,
                slot_ids=kv_manager.slot_ids
            )

        attn_output = attn.o_proj(o.reshape(1, total_T, self.num_heads * self.head_dim))
        return attn_output, None


def replace_attention(module):
    for name, child in module.named_children():
        if isinstance(child, Qwen2Attention):
            setattr(module, name, TritonQwen2Attention(child))
        elif not isinstance(child, TritonQwen2Attention):
            replace_attention(child)
