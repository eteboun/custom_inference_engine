# GQA decode attention (tek query token): K/V P parçaya bölünür (split-KV),
# load kernel'i parça başına kısmi softmax üretir, merge kernel'i birleştirir.
# host: attn.decode_gqa_attention
import triton
import triton.language as tl


# GQA load ve merge kernel'lerinin gövdesi ASCII olmalı: torch.compile (inductor)
# kernel kaynağını ayrı bir dosyaya kopyalıyor, Windows'ta o dosyayı sistem kodlamasıyla
# (cp1254) yazıp UTF-8 okumaya çalışıyor, Türkçe karakter olunca kernel yüklenemiyor
# ("@jit functions should be defined in a Python file")
@triton.jit
def decode_gqa_attention_kernel_load(
    Q, K, V,
    SEQ_LEN,
    partial_m, partial_l, partial_acc,
    stride_kv_head,
    D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    partition_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    P = tl.num_programs(0)

    # gecerli uzunluk runtime'da GPU'dan okunuyor: her L icin yeniden derleme yok,
    # CUDA graph replay'inde de her adimda guncel deger okunur.
    # K/V buffer'i L'den uzun olabilir (StaticCache: max_len), head adimi ayri veriliyor
    L = tl.load(SEQ_LEN).to(tl.int32)

    cols = tl.arange(0, BLOCK_D)
    row_offsets = tl.arange(0, BLOCK_N)

    k_ptr = K + kv_head_idx * stride_kv_head
    v_ptr = V + kv_head_idx * stride_kv_head

    q_rows = tl.arange(0, BLOCK_G)
    q = tl.load(
        Q + (kv_head_idx * GROUP_SIZE + q_rows[:, None]) * D + cols[None, :],
        mask=(q_rows[:, None] < GROUP_SIZE) & (cols[None, :] < D),
        other=0
    )

    ms_old = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    ls_old = tl.full((BLOCK_G,), 0.0, tl.float32)
    accs_old = tl.full((BLOCK_G, BLOCK_D), 0.0, tl.float32)

    for start in range(
        partition_idx * BLOCK_N,
        L,
        P * BLOCK_N
    ):
        rows = start + row_offsets

        offsets = rows[:, None] * D + cols[None, :]
        mask = (rows[:, None] < L) & (cols[None, :] < D)

        k = tl.load(
            k_ptr + offsets,
            mask=mask,
            other=0
        )

        v = tl.load(
            v_ptr + offsets,
            mask=mask,
            other=0
        )

        scores = tl.dot(
            q,
            tl.trans(k)
        )

        norm_scores = tl.where(
            (rows[None, :] < L)
            & (q_rows[:, None] < GROUP_SIZE),
            scores * (D ** -0.5),
            -float("inf")
        )

        ms = tl.max(norm_scores, axis=1)
        ms_new = tl.maximum(ms, ms_old)

        ps = tl.exp(norm_scores - ms_new[:, None])
        ls = tl.sum(ps, axis=1)

        alphas = tl.exp(ms_old - ms_new)

        ls_old = ls_old * alphas + ls
        accs_old = (
                accs_old * alphas[:, None]
                + tl.dot(ps.to(tl.float16), v)
        )

        ms_old = ms_new

    indices = (kv_head_idx * GROUP_SIZE + q_rows) * P
    offset = partition_idx

    tl.store(
        partial_m + indices + offset,
        ms_old, mask=q_rows < GROUP_SIZE
    )

    tl.store(
        partial_l + indices + offset,
        ls_old, mask=q_rows < GROUP_SIZE
    )

    indices = kv_head_idx * GROUP_SIZE * P * D + q_rows * P * D
    offset = partition_idx * D

    tl.store(
        partial_acc + indices[:, None] + offset + cols[None, :],
        accs_old,
        mask=(q_rows[:, None] < GROUP_SIZE) & (cols[None, :] < D)
    )


@triton.jit
def decode_attention_kernel_merge(
    O,
    partial_m, partial_l, partial_acc,
    P: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    head_idx = tl.program_id(0)

    ps = tl.arange(0, BLOCK_P)
    cols = tl.arange(0, BLOCK_D)

    # Head baslangic adresleri
    partial_idx = head_idx * P

    m_ptr = partial_m + partial_idx
    l_ptr = partial_l + partial_idx
    acc_ptr = partial_acc + partial_idx * D
    o_ptr = O + head_idx * D

    # Partial sonuclari yukle
    valid_ps = ps < P
    valid_cols = cols < D

    ms = tl.load(
        m_ptr + ps,
        mask=valid_ps,
        other=-float("inf")
    )

    ls = tl.load(
        l_ptr + ps,
        mask=valid_ps,
        other=0
    )

    offsets = ps[:, None] * D + cols[None, :]
    mask = valid_ps[:, None] & valid_cols[None, :]

    accs = tl.load(
        acc_ptr + offsets,
        mask=mask,
        other=0
    )

    # Partial softmax sonuclarini birlestir
    max_m = tl.max(ms, axis=0)

    alpha = tl.exp(ms - max_m)
    denominator = tl.sum(ls * alpha, axis=0)

    weighted_accs = accs * alpha[:, None]
    numerator = tl.sum(weighted_accs, axis=0)

    # Nihai attention ciktisi
    tl.store(
        o_ptr + cols,
        numerator / denominator,
        mask=valid_cols
    )
