# GQA attention kernel'lerinin host tarafı (grid, ara buffer'lar, config) ve modele
# bağlanması: torch.compile için custom op, Qwen2Attention yerine geçen modül.
# kernel'ler kernels/ altında, mikro benchmark'lar benchmarks/ altında
import torch
import torch.nn.functional as F
from torch.library import triton_op, wrap_triton
import triton

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
    # GQA, batch'teki her dizi için tek query token:
    # q (B, H_q, D) contiguous, k/v (B, H_kv, max_len, D), head içinde satırlar bitişik
    # (batch ve head adımı stride olarak kernel'e gider)
    # seq_len: GPU'da (B,) int tensor, dizi başına geçerli uzunluk (<= max_len, cache
    # kapasitesi). kernel bunu runtime'da okur, K/V'nin sadece ilk seq_len satırı kullanılır
    # q head h, kv head h // (H_q // H_kv)'yi okur
    # load grid (P, H_kv, B): her program bir dizinin bir kv head'inin K/V bloklarını bir
    # kez okur, o kv head'i paylaşan GROUP_SIZE q head'i birlikte işler (BLOCK_G satıra pad'li)
    # merge grid (H_q, B): her program bir dizinin bir q head'inin P parçasını birleştirir,
    # ara sonuçlar (B, H_q, P[, D]) düzeninde
    # wrap: custom op içinde wrap_triton, torch.compile kernel çağrılarını böyle görüyor
    B, H_q, D = q.shape
    _, H_kv, max_len, _ = k.shape
    assert H_q % H_kv == 0
    assert k.shape[0] == B and seq_len.shape == (B,)
    assert q.is_contiguous()
    assert k.stride() == v.stride() and k.stride(3) == 1 and k.stride(2) == D
    # kernel ofsetleri int32 hesaplıyor: K/V'de erişilen en büyük ofset sığmalı
    max_kv_offset = (B - 1) * k.stride(0) + (H_kv - 1) * k.stride(1) + max_len * D
    assert max_kv_offset < 2**31, "K/V ofsetleri int32'ye sığmıyor"
    # tl.dot her boyutun en az 16 olmasını istiyor: q (BLOCK_G, D) @ k^T (D, BLOCK_N)
    # ve p (BLOCK_G, BLOCK_N) @ v (BLOCK_N, D)
    assert BLOCK_N >= 16, "tl.dot için BLOCK_N >= 16 olmalı"
    P = num_programs or default_num_programs(max_len, BLOCK_N)
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_G = max(16, triton.next_power_of_2(H_q // H_kv))

    partial_m = torch.empty((B, H_q, P), device=q.device, dtype=torch.float32)
    partial_l = torch.empty((B, H_q, P), device=q.device, dtype=torch.float32)
    partial_acc = torch.empty((B, H_q, P, D), device=q.device, dtype=torch.float32)
    assert partial_acc.numel() < 2**31, "ara sonuç ofsetleri int32'ye sığmıyor"
    o = torch.empty_like(q)

    wrap(decode_gqa_attention_kernel_load)[(P, H_kv, B)](
        q, k, v,
        seq_len,
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


def prefill_gqa_attention(q, k, v, BLOCK_M=16, BLOCK_N=64, num_warps=4,
                          wrap=lambda kernel: kernel):
    # GQA, causal prefill (boş cache'ten, query pozisyonları 0..T-1):
    # q (T, H_q, D) token-major ve contiguous (RoPE sonrası q da bu düzende,
    # q[0].transpose(0, 1) kopyasız verir). k/v (H_kv, max_len, D): KV cache buffer'ı,
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
# k/v KV cache buffer'ı (değiştirilmiyor), seq_len dizi başına geçerli uzunluk (GPU, (B,))
@triton_op("inference::gqa_decode_attention", mutates_args=())
def gqa_decode_attention_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_len: torch.Tensor
) -> torch.Tensor:
    return decode_gqa_attention(q, k, v, seq_len, **GQA_DECODE_CONFIG, wrap=wrap_triton)


class TritonQwen2Attention(torch.nn.Module):
    # orijinal Qwen2Attention modülünün yerine geçer (replace_attention).
    # model forward'una kv_manager (kv_cache.KVCacheManager) verilince, batch boyutu
    # manager'ın batch_size'ı: q/k/v projeksiyonu (tek matmul) + RoPE, K/V manager'a
    # yazılır (update, dizi başına pozisyon manager.start'tan), sonra attention:
    # - decode (tek token): Triton GQA kernel'i, dizi başına geçerli uzunluk
    #   manager.start + 1 (GPU'da, yeni token dahil). CUDA graph'ta her replay'de günceldir
    # - prefill (T > 1): sdpa, causal, sadece bu T token'ın K/V'si. cache boş olmalı
    #   (start = 0, query pozisyonları 0..T-1); generate prefill'den önce reset ediyor.
    #   farklı uzunluklu prompt'lar right padding'le verilebilir, advance gerçek uzunluklarla
    #   prefill için Triton kernel'i de var (prefill_gqa_attention) ama WSL'de sdpa'nın
    #   FlashAttention 2'sinden uzun prompt'larda yavaş (bkz. GQA_PREFILL_CONFIG)
    # manager.advance çağırana ait (adım başına bir kez, bütün katmanlardan sonra).
    # attention_mask kullanılmıyor (causal; pad sadece sağda olabilir).
    # kv_manager verilmezse orijinal modüle (sdpa) gider
    def __init__(self, original_attn):
        super().__init__()

        assert original_attn.sliding_window is None, "sliding window desteklenmiyor"
        self.original = original_attn
        self.layer_idx = original_attn.layer_idx
        self.num_heads = original_attn.config.num_attention_heads
        self.head_dim = original_attn.head_dim
        # kernel D ** -0.5 ile ölçekliyor
        assert original_attn.scaling == self.head_dim ** -0.5

        # q/k/v projeksiyonu tek matmul: ağırlıklar ve bias'lar alt alta
        # (896 + 128 + 128 = 1152 satır). çıktı split ile kopyasız q/k/v view'larına
        # bölünür. decode'da 3 gemv yerine 1: okunan ağırlık aynı, kernel sayısı az.
        # orijinal q/k/v_proj kv_manager'sız yol için duruyor, ağırlıklar iki kopya
        # (~2 MB / katman). fuse_qkv=False ayrı projeksiyonlarla çalışır (karşılaştırma için)
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

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, kv_manager=None, **kwargs):
        if kv_manager is None:
            return self.original(
                hidden_states, position_embeddings, attention_mask,
                past_key_values=past_key_values, **kwargs
            )

        attn = self.original
        B, T = hidden_states.shape[:2]
        hidden_shape = (B, T, -1, self.head_dim)
        if self.fuse_qkv:
            q, k, v = torch.nn.functional.linear(
                hidden_states, self.qkv_weight, self.qkv_bias
            ).split(self.qkv_sizes, dim=-1)
        else:
            q, k, v = attn.q_proj(hidden_states), attn.k_proj(hidden_states), attn.v_proj(hidden_states)
        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # manager token sırasında (B, T, H_kv, D) bekliyor: key_states (B, H_kv, T, D)'nin
        # transpose'u, kopyasız view
        kv_manager.update(
            self.layer_idx, key_states.transpose(1, 2), value_states.transpose(1, 2)
        )

        if T == 1:
            # update start'ı ilerletmiyor: her dizinin yeni token'ı kendi start
            # pozisyonunda, uzunluk start + 1 (B,)
            o = gqa_decode_attention_op(
                query_states.reshape(B, self.num_heads, self.head_dim),
                kv_manager.keys[self.layer_idx],
                kv_manager.values[self.layer_idx],
                kv_manager.start + 1,
            )
        else:
            # boş cache'ten prefill: geçmiş yok, attention sadece bu T token'ın K/V'si
            # (cache'ten okumak yerine elimizdeki tensörler). (B, H_q, T, D) -> (B, T, H_q, D).
            # right padding'de pad'ler gerçek token'lardan sonra: causal mask yüzünden
            # gerçek token'lar onları görmüyor, pad'lerin çıktısı kullanılmıyor
            o = F.scaled_dot_product_attention(
                query_states, key_states, value_states, is_causal=True, enable_gqa=True
            ).transpose(1, 2)

        attn_output = attn.o_proj(o.reshape(B, T, self.num_heads * self.head_dim))
        return attn_output, None


def replace_attention(module):
    for name, child in module.named_children():
        if isinstance(child, Qwen2Attention):
            setattr(module, name, TritonQwen2Attention(child))
        elif not isinstance(child, TritonQwen2Attention):
            replace_attention(child)
