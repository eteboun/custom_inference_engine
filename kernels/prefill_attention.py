# GQA causal prefill attention (boş cache'ten, flash-attention tarzı): her program
# BLOCK_M token x bir kv head'in GROUP_SIZE q head'ini işler, K/V'yi causal sınıra kadar
# blok blok gezer (online softmax), çıkışı doğrudan yazar.
# host: attn.prefill_gqa_attention. modelde kullanılmıyor (bkz. attn.TritonQwen2Attention)
# gövde ASCII olmalı, bkz. kernels/decode_attention.py'deki not
import triton
import triton.language as tl


@triton.jit
def prefill_gqa_attention_kernel(
    Q, K, V,
    O,
    L,
    MAX_LEN,
    D: tl.constexpr,
    KV_HEADS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    # Q: (T, H_q, D)
    # KV: (H_kv, max_len, D)

    tile_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    cols = tl.arange(0, BLOCK_D)

    q_rows = tl.arange(0, BLOCK_M)
    q_heads = tl.arange(0, BLOCK_G)

    q_row_offsets = (tile_idx * BLOCK_M * Q_HEADS * D
        + q_rows[:, None, None] * Q_HEADS * D)
    q_head_offsets = (kv_head_idx * GROUP_SIZE * D
        + q_heads[None, :, None] * D)
    q_col_offsets = cols[None, None, :]

    q_row_mask = (tile_idx * BLOCK_M + q_rows)[:, None, None] < L
    q_head_mask = q_heads[None, :, None] < GROUP_SIZE
    q_col_mask = cols[None, None, :] < D

    # (BLOCK_M, BLOCK_G, BLOCK_D)
    qs = tl.load(
        Q + q_head_offsets
        + q_row_offsets
        + q_col_offsets,
        mask=q_head_mask
        & q_row_mask
        & q_col_mask,
        other=0
    )
    qs = tl.reshape(
        qs, (BLOCK_M * BLOCK_G, BLOCK_D)
    )

    kv_rows = tl.arange(0, BLOCK_N)
    head_offsets = kv_head_idx * MAX_LEN * D
    col_offsets = cols[None, :]
    kv_col_mask = cols[None, :] < D

    end = tl.minimum(L, (tile_idx + 1) * BLOCK_M)

    ms_old = tl.full((BLOCK_M, BLOCK_G), -float("inf"), dtype=tl.float32)
    ls_old = tl.full((BLOCK_M, BLOCK_G), 0.0, dtype=tl.float32)
    accs_old = tl.full((BLOCK_M, BLOCK_G, BLOCK_D), 0.0, dtype=tl.float32)

    for start in range(
        0, end, BLOCK_N
    ):
        rows = start + kv_rows

        row_offsets = rows[:, None] * D
        kv_row_mask = rows[:, None] < L

        # (BLOCK_N, BLOCK_D)
        ks = tl.load(
            K + head_offsets
            + row_offsets
            + col_offsets,
            mask=kv_row_mask
                 & kv_col_mask,
            other=0
        )

        # (BLOCK_N, BLOCK_D)
        vs = tl.load(
            V + head_offsets
            + row_offsets
            + col_offsets,
            mask=kv_row_mask
                 & kv_col_mask,
            other=0
        )

        # (BLOCK_M * BLOCK_G, BLOCK_N)
        scores = tl.dot(
            qs,
            tl.trans(ks)
        )

        # (BLOCK_M, BLOCK_G, BLOCK_N)
        scores = tl.reshape(
            scores, (BLOCK_M, BLOCK_G, BLOCK_N)
        )

        # (BLOCK_M, 1, BLOCK_N)
        causal_mask = ((tile_idx * BLOCK_M + q_rows)[:, None, None]
                       >= rows[None, None, :])

        # (BLOCK_M, BLOCK_G, BLOCK_N)
        norm_scores = tl.where(
            q_row_mask
            & q_head_mask
            & causal_mask,
            scores * (D ** -0.5),
            -float("inf")
        )

        ms = tl.max(norm_scores, axis=2)
        ms_new = tl.maximum(ms_old, ms)

        alphas = tl.exp(ms_old - ms_new)
        ps = tl.exp(norm_scores - ms_new[:, :, None])

        ls = tl.sum(
            ps,
            axis=2
        )
        ls_new = ls_old * alphas + ls

        # (BLOCK_M * BLOCK_G, BLOCK_D)
        output = tl.dot(
            tl.reshape(
                ps, (BLOCK_M * BLOCK_G, BLOCK_N)
            ).to(tl.float16),
            vs
        )

        accs = tl.reshape(
            output, (BLOCK_M, BLOCK_G, BLOCK_D)
        )
        accs_new = accs_old * alphas[:, :, None] + accs

        ms_old = ms_new
        ls_old = ls_new
        accs_old = accs_new

    # (BLOCK_M, BLOCK_G, BLOCK_D)
    attn = accs_old / ls_old[:, :, None]

    tl.store(
        O + q_row_offsets
        + q_head_offsets
        + q_col_offsets,
        value=attn,
        mask=q_row_mask
        & q_head_mask
        & q_col_mask
    )
