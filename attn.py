# GQA attention kernel'lerinin host tarafı (grid, ara buffer'lar, config) ve modele
# bağlanması: torch.compile için custom op, Qwen2Attention yerine geçen modül.
# kernel'ler kernels/ altında, mikro benchmark'lar benchmarks/ altında
import torch
from torch.library import triton_op, wrap_triton
import triton

from transformers.cache_utils import Cache, StaticLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, apply_rotary_pos_emb

from kernels.decode_attention import (
    decode_gqa_attention_kernel_load, decode_attention_kernel_merge
)
from kernels.prefill_attention import prefill_gqa_attention_kernel


def default_num_programs(L, BLOCK_N):
    # her SM'e bir program, ama blok sayısından fazla program açmanın anlamı yok
    # (fazlası hiç blok işlemeden m=-inf, l=0 yazar)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    return min(sm_count, triton.cdiv(L, BLOCK_N))


def decode_gqa_attention(q, k, v, seq_len, BLOCK_N=64, num_programs=None, num_warps=4,
                         wrap=lambda kernel: kernel):
    # GQA, tek query token: q (H_q, D), k/v (H_kv, max_len, D), son boyut contiguous
    # seq_len: GPU'da tek elemanlı int tensor, geçerli uzunluk (<= max_len). kernel bunu
    # runtime'da okur, yani K/V buffer'ının sadece ilk seq_len satırı kullanılır
    # q head h, kv head h // (H_q // H_kv)'yi okur
    # load grid (P, H_kv): her program bir kv head'in K/V bloklarını bir kez okur,
    # o kv head'i paylaşan GROUP_SIZE q head'i birlikte işler (BLOCK_G satıra pad'li)
    # merge grid (H_q,), ara sonuçlar q head başına
    # wrap: custom op içinde wrap_triton, torch.compile kernel çağrılarını böyle görüyor
    H_q, D = q.shape
    H_kv, max_len, _ = k.shape
    assert H_q % H_kv == 0
    assert k.stride() == v.stride() and k.stride(2) == 1 and k.stride(1) == D
    # tl.dot her boyutun en az 16 olmasını istiyor: q (BLOCK_G, D) @ k^T (D, BLOCK_N)
    # ve p (BLOCK_G, BLOCK_N) @ v (BLOCK_N, D)
    assert BLOCK_N >= 16, "tl.dot için BLOCK_N >= 16 olmalı"
    P = num_programs or default_num_programs(max_len, BLOCK_N)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_G = max(16, triton.next_power_of_2(H_q // H_kv))

    partial_m = torch.empty((H_q, P), device=q.device, dtype=torch.float32)
    partial_l = torch.empty((H_q, P), device=q.device, dtype=torch.float32)
    partial_acc = torch.empty((H_q, P, D), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    wrap(decode_gqa_attention_kernel_load)[(P, H_kv)](
        q, k, v,
        seq_len,
        partial_m, partial_l, partial_acc,
        k.stride(0),
        D=D,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_G=BLOCK_G,
        GROUP_SIZE=H_q // H_kv,
        num_warps=num_warps,
    )

    wrap(decode_attention_kernel_merge)[(H_q,)](
        o,
        partial_m, partial_l, partial_acc,
        P=P,
        D=D,
        BLOCK_P=triton.next_power_of_2(P),
        BLOCK_D=BLOCK_D,
    )

    return o


def prefill_gqa_attention(q, k, v, BLOCK_M=16, BLOCK_N=64, num_warps=4,
                          wrap=lambda kernel: kernel):
    # GQA, causal prefill (boş cache'ten, query pozisyonları 0..T-1):
    # q (T, H_q, D) token-major ve contiguous (RoPE sonrası q da bu düzende,
    # q[0].transpose(0, 1) kopyasız verir). k/v (H_kv, max_len, D): StaticCache buffer'ı,
    # update'ten sonra ilk T satırı dolu. o (T, H_q, D): view(T, H_q * D) kopyasız o_proj'a gider
    # kernel adresleri bu düzenlere göre sabit hesaplıyor, stride almıyor
    # grid (cdiv(T, BLOCK_M), H_kv): her program BLOCK_M token x GROUP_SIZE q head'i
    # birlikte işler, K/V'yi causal sınıra kadar gezip çıkışı doğrudan yazar.
    # decode'dan farklı olarak K bölünmüyor: tile'lar zaten GPU'yu dolduruyor, bölmek
    # ara sonuç buffer'ı + merge kernel'i demekti ve ölçümde yavaştı
    T, H_q, D = q.shape
    H_kv, max_len, _ = k.shape
    assert H_q % H_kv == 0
    assert k.shape == v.shape and k.shape[2] == D and T <= max_len
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    GROUP_SIZE = H_q // H_kv
    BLOCK_D = triton.next_power_of_2(D)
    # q tile'ı (BLOCK_M * BLOCK_G, D) olarak tl.dot'a giriyor, 16 sınırı satır toplamı için
    BLOCK_G = triton.next_power_of_2(GROUP_SIZE)
    assert BLOCK_M * BLOCK_G >= 16 and BLOCK_N >= 16, "tl.dot için boyutlar >= 16 olmalı"

    o = torch.empty_like(q)

    wrap(prefill_gqa_attention_kernel)[(triton.cdiv(T, BLOCK_M), H_kv)](
        q, k, v,
        o,
        T,
        max_len,
        D=D,
        KV_HEADS=H_kv,
        Q_HEADS=H_q,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_G=BLOCK_G,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=num_warps,
    )

    return o


# GQA decode kernel config'i, benchmarks/gqa_sweep.py sonuçlarından (results/gqa_sweep.html: 14 q / 2 kv head,
# D=64, RTX 4070 Laptop, L = 1024..32768). cold'a göre seçildi: bütün L'lerde tek
# bir config olarak en iyisi, her L'de o L'nin en iyi cold süresinden en fazla %0.8 yavaş.
# L'ye bağlı değil: P blok sayısından büyük olursa (kısa L) fazla programlar
# hiç blok işlemeden m=-inf, l=0 yazar, merge bunları yok sayar
GQA_DECODE_CONFIG = {"BLOCK_N": 16, "num_warps": 4, "num_programs": 16}

# prefill kernel config'i (modelde kullanılmıyor, bkz. TritonQwen2Attention).
# Windows'ta sdpa'dan (mem-efficient cutlass) %11-12 hızlı. WSL'de sdpa FlashAttention 2
# kullanıyor: BLOCK_M x BLOCK_N x num_warps taramasında T=1024'te flash'tan hızlı,
# T=4096/16384'te en iyi config bile %8/%19 yavaş, bu config %10/%25 yavaş
GQA_PREFILL_CONFIG = {"BLOCK_M": 16, "BLOCK_N": 64, "num_warps": 4}


# torch.compile için custom op. triton_op + wrap_triton ile inductor kernel'leri
# opak bir çağrı olarak değil, kendi ürettiği kodun içinde doğrudan çalıştırır
# (etrafındaki RMSNorm/RoPE füzyonları bozulmaz). eager'da da normal fonksiyon gibi çalışır.
# k/v StaticCache buffer'ı (değiştirilmiyor), seq_len geçerli uzunluk (GPU, tek eleman)
@triton_op("inference::gqa_decode_attention", mutates_args=())
def gqa_decode_attention_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: torch.Tensor
) -> torch.Tensor:
    return decode_gqa_attention(q, k, v, seq_len, **GQA_DECODE_CONFIG, wrap=wrap_triton)


class TritonQwen2Attention(torch.nn.Module):
    # orijinal Qwen2Attention modülünün yerine geçer (replace_attention).
    # decode adımında (tek token + StaticCache) q/k/v projeksiyonu, RoPE ve cache
    # güncellemesi Qwen2Attention.forward ile aynı, attention Triton GQA kernel'iyle.
    # geçerli uzunluk StaticCache katmanının GPU'daki cumulative_length tensor'ından
    # okunur (update'ten sonra yeni token dahil), CUDA graph'ta da her adımda günceldir.
    # attention_mask kullanılmıyor: tek query için causal mask "ilk seq_len pozisyon"
    # demek, bu yüzden prompt'ta pad olmamalı (batch 1, pad'siz).
    # prefill ve diğer durumlar orijinal modüle (sdpa) gider. prefill için Triton kernel'i
    # var (prefill_gqa_attention) ama WSL'de sdpa'nın FlashAttention 2'sinden uzun
    # prompt'larda yavaş (bkz. GQA_PREFILL_CONFIG)
    def __init__(self, original_attn):
        super().__init__()

        assert original_attn.sliding_window is None, "sliding window desteklenmiyor"
        self.original = original_attn
        self.layer_idx = original_attn.layer_idx
        self.num_heads = original_attn.config.num_attention_heads
        self.head_dim = original_attn.head_dim
        # kernel D ** -0.5 ile ölçekliyor
        assert original_attn.scaling == self.head_dim ** -0.5

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        layer = (
            past_key_values.layers[self.layer_idx]
            if isinstance(past_key_values, Cache) else None
        )
        if hidden_states.shape[:2] != (1, 1) or not isinstance(layer, StaticLayer):
            return self.original(
                hidden_states, position_embeddings, attention_mask,
                past_key_values=past_key_values, **kwargs
            )

        attn = self.original
        hidden_shape = (1, 1, -1, self.head_dim)
        query_states = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # keys/values: (1, H_kv, max_len, D), bütün StaticCache buffer'ı
        keys, values = past_key_values.update(key_states, value_states, self.layer_idx)

        o = gqa_decode_attention_op(
            query_states.reshape(self.num_heads, self.head_dim),
            keys[0],
            values[0],
            layer.cumulative_length,
        )

        attn_output = attn.o_proj(o.view(1, 1, self.num_heads * self.head_dim))
        return attn_output, None


def replace_attention(module):
    for name, child in module.named_children():
        if isinstance(child, Qwen2Attention):
            setattr(module, name, TritonQwen2Attention(child))
        elif not isinstance(child, TritonQwen2Attention):
            replace_attention(child)
