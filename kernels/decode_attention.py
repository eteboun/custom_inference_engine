# GQA decode attention (tek query token): K/V P parçaya bölünür (split-KV),
# load kernel'i parça başına kısmi softmax üretir, merge kernel'i birleştirir.
# host: attn.core.decode_gqa_attention
import triton
import triton.language as tl


# GQA load ve merge kernel'lerinin gövdesi ASCII olmalı: torch.compile (inductor)
# kernel kaynağını ayrı bir dosyaya kopyalıyor, Windows'ta o dosyayı sistem kodlamasıyla
# (cp1254) yazıp UTF-8 okumaya çalışıyor, Türkçe karakter olunca kernel yüklenemiyor
# ("@jit functions should be defined in a Python file")
@triton.jit
def decode_gqa_attention_kernel_load(
    Q, K, V,
    LENS,
    SLOT_IDS,
    BLOCK_TABLE,
    partial_m, partial_l, partial_acc,
    kv_page_stride,
    kv_head_stride,
    PAGE_SIZE: tl.constexpr,
    N_PAGES: tl.constexpr,
    D: tl.constexpr,
    Q_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Q: (total_T, H_q, D)
    # KV: (n_pages, H_kv, page_size, D)

    partition_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    seq_idx = tl.program_id(2)
    P = tl.num_programs(0)

    slot_id = tl.load(
        SLOT_IDS + seq_idx
    )

    L = tl.load(
        LENS + seq_idx
    ).to(tl.int32)

    qkv_cols = tl.arange(0, BLOCK_D)
    qkv_col_offsets = qkv_cols[None, :]
    qkv_col_mask = qkv_cols[None, :] < D

    q_heads = tl.arange(0, BLOCK_G)
    q_batch_offsets = seq_idx * Q_HEADS * D
    q_head_offsets = (kv_head_idx * GROUP_SIZE + q_heads)[:, None] * D
    q_head_mask = q_heads[:, None] < GROUP_SIZE

    # (BLOCK_G, BLOCK_D)
    q = tl.load(
        Q + q_batch_offsets
        + q_head_offsets
        + qkv_col_offsets,
        mask=q_head_mask
        & qkv_col_mask,
        other=0
    )

    ms_old = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    ls_old = tl.full((BLOCK_G,), 0.0, tl.float32)
    accs_old = tl.full((BLOCK_G, BLOCK_D), 0.0, tl.float32)

    kv_rows = tl.arange(0, BLOCK_N)

    for start in range(
        partition_idx * BLOCK_N,
        L,
        P * BLOCK_N
    ):
        rows = start + kv_rows

        logical_pages = rows // PAGE_SIZE
        page_ids = tl.load(
            BLOCK_TABLE + slot_id * N_PAGES + logical_pages,
            mask=rows < L,
            other=0
        )
        offsets = rows % PAGE_SIZE

        kv_page_offsets = page_ids[:, None] * kv_page_stride
        kv_head_offsets = kv_head_idx * kv_head_stride
        kv_row_offsets = offsets[:, None] * D
        kv_row_mask = rows[:, None] < L

        # (BLOCK_N, BLOCK_D)
        k = tl.load(
            K + kv_page_offsets
            + kv_head_offsets
            + kv_row_offsets
            + qkv_col_offsets,
            mask=kv_row_mask
            & qkv_col_mask,
            other=0
        )

        # (BLOCK_N, BLOCK_D)
        v = tl.load(
            V + kv_page_offsets
            + kv_head_offsets
            + kv_row_offsets
            + qkv_col_offsets,
            mask=kv_row_mask
            & qkv_col_mask,
            other=0
        )

        # (BLOCK_G, BLOCK_N)
        scores = tl.dot(
            q,
            tl.trans(k)
        )

        # (BLOCK_G, BLOCK_N)
        norm_scores = tl.where(
            (rows[None, :] < L)
            & q_head_mask,
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

    # partial_m, partial_l: (B, H_q, P)
    # partial_acc: (B, H_q, P, D)

    ml_batch_offsets = seq_idx * Q_HEADS * P
    ml_head_offsets = (kv_head_idx * GROUP_SIZE + q_heads) * P
    ml_partition_offsets = partition_idx
    ml_head_mask = q_heads < GROUP_SIZE

    tl.store(
        partial_m + ml_batch_offsets
        + ml_head_offsets
        + ml_partition_offsets,
        ms_old,
        mask=ml_head_mask
    )

    tl.store(
        partial_l + ml_batch_offsets
        + ml_head_offsets
        + ml_partition_offsets,
        ls_old,
        mask=ml_head_mask
    )

    acc_batch_offsets = seq_idx * Q_HEADS * P * D
    acc_head_offsets = (kv_head_idx * GROUP_SIZE + q_heads)[:, None] * P * D
    acc_partition_offsets = partition_idx * D

    tl.store(
        partial_acc + acc_batch_offsets
        + acc_head_offsets
        + acc_partition_offsets
        + qkv_col_offsets,
        accs_old,
        mask=q_head_mask
        & qkv_col_mask
    )


@triton.jit
def decode_attention_kernel_merge(
    O,
    partial_m, partial_l, partial_acc,
    P: tl.constexpr,
    D: tl.constexpr,
    Q_HEADS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # O: (total_T, H_q, D)
    # partial_m, partial_l: (B, H_q, P)
    # partial_acc: (B, H_q, P, D)

    head_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    partitions = tl.arange(0, BLOCK_P)

    cols = tl.arange(0, BLOCK_D)

    ml_batch_offsets = batch_idx * Q_HEADS * P
    ml_head_offsets = head_idx * P
    ml_partition_mask = partitions < P

    # (BLOCK_P,)
    ms = tl.load(
        partial_m + ml_batch_offsets
        + ml_head_offsets
        + partitions,
        mask=ml_partition_mask,
        other=-float("inf")
    )

    # (BLOCK_P,)
    ls = tl.load(
        partial_l + ml_batch_offsets
        + ml_head_offsets
        + partitions,
        mask=ml_partition_mask,
        other=0.0
    )

    acc_batch_offsets = batch_idx * Q_HEADS * P * D
    acc_head_offsets = head_idx * P * D
    acc_partition_offsets = partitions[:, None] * D
    acc_col_offsets = cols[None, :]
    acc_partition_mask = partitions[:, None] < P
    acc_col_mask = cols[None, :] < D

    # (BLOCK_P, BLOCK_D)
    accs = tl.load(
        partial_acc + acc_batch_offsets
        + acc_head_offsets
        + acc_partition_offsets
        + acc_col_offsets,
        mask=acc_partition_mask
        & acc_col_mask,
        other=0.0
    )

    max_m = tl.max(ms, axis=0)

    # (BLOCK_P,)
    alpha = tl.exp(ms - max_m)

    denominator = tl.sum(ls * alpha, axis=0)

    # (BLOCK_P, BLOCK_D)
    weighted_accs = accs * alpha[:, None]

    # (BLOCK_D,)
    numerator = tl.sum(weighted_accs, axis=0)

    o_batch_offset = batch_idx * Q_HEADS * D
    o_head_offsets = head_idx * D
    o_cols_mask = cols < D

    output = tl.where(
        max_m > -float("inf"),
        numerator / denominator,
        0.0
    )

    tl.store(
        O + o_batch_offset
        + o_head_offsets
        + cols,
        output,
        mask=o_cols_mask
    )
