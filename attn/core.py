import torch
from torch.library import triton_op, wrap_triton
import triton

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, apply_rotary_pos_emb, eager_attention_forward
)

from kernels.decode_attention import (
    decode_gqa_attention_kernel_load, decode_attention_kernel_merge
)
from kernels.prefill_attention import mixed_gqa_attention_kernel
from quant.core import Fp8Linear


def decode_gqa_attention(q, k, v, lens, slot_ids, block_table, num_programs, BLOCK_N=64, num_warps=4,
                         wrap=lambda kernel: kernel):
    total_T, H_q, D = q.shape[1:]
    N_PAGES, H_kv, PAGE_SIZE, _ = k.shape
    B = slot_ids.shape[0]
    assert H_q % H_kv == 0
    assert lens.shape == (B,)
    assert N_PAGES == block_table.shape[1]
    assert total_T == B
    assert q.is_contiguous()
    assert k.stride() == v.stride() and k.stride(3) == 1 and k.stride(2) == D
    max_kv_offset = (N_PAGES - 1) * k.stride(0) + (H_kv - 1) * k.stride(1) + PAGE_SIZE * D
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
        block_table,
        partial_m, partial_l, partial_acc,
        k.stride(0),
        k.stride(1),
        PAGE_SIZE=PAGE_SIZE,
        N_PAGES=N_PAGES,
        D=D,
        Q_HEADS=H_q,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_G=BLOCK_G,
        GROUP_SIZE=H_q // H_kv,
        num_warps=num_warps
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


def mixed_gqa_attention(q, k, v,
                        max_seqlen: int, cu_seqlens, kv_lens,
                        slot_ids, block_table,
                        BLOCK_M=16, BLOCK_N=64, num_warps=4,
                        wrap=lambda kernel: kernel):
    total_T, H_q, D = q.shape[1:]
    N_PAGES, H_kv, PAGE_SIZE, _ = k.shape
    B = slot_ids.shape[0]
    assert H_q % H_kv == 0
    assert kv_lens.shape == (B,)
    assert cu_seqlens.shape == (B + 1,)
    assert N_PAGES == block_table.shape[1]
    assert k.shape == v.shape and k.shape[3] == D
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    GROUP_SIZE = H_q // H_kv
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_G = triton.next_power_of_2(GROUP_SIZE)
    assert BLOCK_M * BLOCK_G >= 16 and BLOCK_N >= 16, "tl.dot için boyutlar >= 16 olmalı"

    o = torch.empty_like(q)

    wrap(mixed_gqa_attention_kernel)[(triton.cdiv(max_seqlen, BLOCK_M), H_kv, B)](
        q, k, v,
        o,
        CU_SEQLENS=cu_seqlens,
        SLOT_IDS=slot_ids,
        KV_LENS=kv_lens,
        BLOCK_TABLE=block_table,
        kv_page_stride=k.stride(0),
        kv_head_stride=k.stride(1),
        PAGE_SIZE=PAGE_SIZE,
        N_PAGES=N_PAGES,
        D=D,
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
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        lens: torch.Tensor, slot_ids: torch.Tensor, block_table: torch.Tensor
) -> torch.Tensor:
    return decode_gqa_attention(q, k, v,
                                lens=lens, slot_ids=slot_ids, block_table=block_table,
                                **GQA_DECODE_CONFIG, wrap=wrap_triton)

@triton_op("inference::gqa_mixed_attention", mutates_args=())
def gqa_mixed_attention_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        max_seqlen: int, cu_seqlens: torch.Tensor, kv_lens: torch.Tensor,
        slot_ids: torch.Tensor, block_table: torch.Tensor
) -> torch.Tensor:
    return mixed_gqa_attention(q, k, v, **GQA_PREFILL_CONFIG,
                               max_seqlen=max_seqlen, cu_seqlens=cu_seqlens,
                               block_table=block_table, kv_lens=kv_lens,
                               slot_ids=slot_ids, wrap=wrap_triton)

def merge_projections(projs: list[torch.nn.Linear]) -> torch.nn.Linear:
    weight = torch.cat([p.weight for p in projs]).detach()
    bias = torch.cat([p.bias for p in projs]).detach()

    merged = torch.nn.Linear(
        weight.shape[1], weight.shape[0], dtype=weight.dtype, device=weight.device
    )
    merged.weight = torch.nn.Parameter(weight, requires_grad=False)
    merged.bias = torch.nn.Parameter(bias, requires_grad=False)
    return merged


def project_qkv(attention, hidden_states):
    if hasattr(attention, "qkv"):
        return attention.qkv(hidden_states).split(attention.qkv_sizes, dim=-1)
    if isinstance(attention.q_proj, Fp8Linear):
        x_fp8, scale = attention.q_proj.quantize_input(hidden_states)
        return (
            attention.q_proj.forward_quantized(x_fp8, scale),
            attention.k_proj.forward_quantized(x_fp8, scale),
            attention.v_proj.forward_quantized(x_fp8, scale),
        )
    return (
        attention.q_proj(hidden_states),
        attention.k_proj(hidden_states),
        attention.v_proj(hidden_states),
    )


def replace_attention(module):
    for name, child in module.named_children():
        if type(child) in (Qwen2Attention, _Qwen2Attention):
            setattr(module, name, TritonQwen2Attention(child))
        else:
            replace_attention(child)


def wrap_qwen2_attention(module, *, merge_qkv: bool = False) -> None:
    for name, child in module.named_children():
        if type(child) is Qwen2Attention:
            setattr(module, name, _Qwen2Attention(child, merge_qkv=merge_qkv))
        else:
            wrap_qwen2_attention(child, merge_qkv=merge_qkv)


def use_triton_attention(model):
    if not getattr(model, "_uses_triton_attention", False):
        replace_attention(model)
        model._uses_triton_attention = True


class _Qwen2Attention(Qwen2Attention):

    def __init__(self, original: Qwen2Attention, *, merge_qkv: bool = False) -> None:
        with torch.device("meta"):
            super().__init__(original.config, original.layer_idx)
        self.eval()

        if hasattr(original, "qkv"):
            self.qkv = original.qkv
            self.qkv_sizes = original.qkv_sizes
        elif merge_qkv:
            projs = [original.q_proj, original.k_proj, original.v_proj]
            assert all(p.bias is not None for p in projs)
            self.qkv = merge_projections(projs)
            self.qkv_sizes = [p.out_features for p in projs]
        else:
            self.q_proj = original.q_proj
            self.k_proj = original.k_proj
            self.v_proj = original.v_proj
        self.o_proj = original.o_proj
        if hasattr(self, "qkv"):
            del self.q_proj, self.k_proj, self.v_proj

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q, k, v = project_qkv(self, hidden_states)
        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class TritonQwen2Attention(_Qwen2Attention):

    def __init__(self, original: Qwen2Attention) -> None:
        super().__init__(original, merge_qkv=True)
        assert self.sliding_window is None, "sliding window desteklenmiyor"
        assert self.scaling == self.head_dim ** -0.5
        self.num_heads = self.config.num_attention_heads

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_values=None, kv_manager=None, **kwargs):
        if kv_manager is None:
            raise ValueError("Triton attention requires a kv_manager.")

        metadata = kwargs.get("metadata")
        if metadata is None:
            raise ValueError(
                "'metadata' is required with custom kv_manager."
            )

        total_T = hidden_states.shape[1]
        hidden_shape = (1, total_T, -1, self.head_dim)
        q, k, v = project_qkv(self, hidden_states)

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

        if metadata.full_decode:
            kv_manager.decode_update(
                self.layer_idx,
                metadata.slot_ids,
                key_states,
                value_states
            )

        else:
            kv_manager.mixed_update(
                self.layer_idx,
                metadata.pages,
                metadata.offsets,
                key_states,
                value_states
            )

        o = self.paged_attention(query_states, kv_manager, metadata)
        attn_output = self.o_proj(o.reshape(1, total_T, self.num_heads * self.head_dim))
        return attn_output, None

    def paged_attention(self, query_states, kv_manager, metadata):
        if metadata.full_decode:
            return gqa_decode_attention_op(
                query_states,
                kv_manager.keys[self.layer_idx],
                kv_manager.values[self.layer_idx],
                kv_manager.lens[metadata.slot_ids] + 1,
                slot_ids=metadata.slot_ids,
                block_table=kv_manager.block_table
            )

        return gqa_mixed_attention_op(
            query_states,
            kv_manager.keys[self.layer_idx],
            kv_manager.values[self.layer_idx],
            kv_lens=kv_manager.lens[metadata.slot_ids],
            max_seqlen=metadata.max_seqlen, cu_seqlens=metadata.cu_seqlens,
            slot_ids=metadata.slot_ids, block_table=kv_manager.block_table
        )
