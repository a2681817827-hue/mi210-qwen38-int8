# SPDX-License-Identifier: Apache-2.0
"""64-row gfx90a grouped-int8 split-K decode core for verify batches."""

from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def g128_core_cdna2(
    Q,
    K,
    V,
    SK,
    SV,
    BT,
    SEQLENS,
    CUQ,
    PARTIAL,
    PMAX,
    PSUM,
    SCALE: gl.constexpr,
    NUM_SEQS: gl.constexpr,
    NUM_QHEADS: gl.constexpr,
    NQ_PER_KV: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    SPLITS: gl.constexpr,
    BT_STRIDE: gl.constexpr,
    Q_STRIDE0: gl.constexpr,
    Q_STRIDE1: gl.constexpr,
    K_STRIDE0: gl.constexpr,
    K_STRIDE1: gl.constexpr,
    K_STRIDE2: gl.constexpr,
    V_STRIDE0: gl.constexpr,
    V_STRIDE1: gl.constexpr,
    V_STRIDE2: gl.constexpr,
    S_STRIDE0: gl.constexpr,
    S_STRIDE1: gl.constexpr,
    S_STRIDE2: gl.constexpr,
    MMA_DT: gl.constexpr,
    MMA_FP16: gl.constexpr,
    KV_TILE: gl.constexpr = 32,
    Q_ROWS: gl.constexpr = 64,
    Q_TOKENS: gl.constexpr = 10,
    QK_INT8: gl.constexpr = False,
    NUM_WARPS: gl.constexpr = 4,
    MMA_WARPS_N: gl.constexpr = 1,
):
    gl.static_assert(Q_TOKENS * NQ_PER_KV <= Q_ROWS)
    gl.static_assert(NUM_WARPS == 2 or NUM_WARPS == 4)
    gl.static_assert(NUM_WARPS % MMA_WARPS_N == 0)
    block_id = gl.program_id(0)
    kv_head = gl.program_id(1)
    seg = gl.program_id(2)
    seq = 0
    for s in gl.static_range(NUM_SEQS):
        start = gl.load(CUQ + s)
        first_block = start // Q_TOKENS + s
        seq = gl.where(block_id >= first_block, s, seq)
    q_start = gl.load(CUQ + seq)
    q_end = gl.load(CUQ + seq + 1)
    q_len = q_end - q_start
    local_block = block_id - (q_start // Q_TOKENS + seq)
    if local_block * Q_TOKENS >= q_len:
        return
    seq_len = gl.load(SEQLENS + seq)
    context = seq_len - q_len
    tiles_per_seg = (seq_len + SPLITS * KV_TILE - 1) // (SPLITS * KV_TILE)
    lo = seg * tiles_per_seg
    if lo * KV_TILE >= seq_len:
        return
    # The causal mask below excludes the conservative extra key in this bound.
    max_prefix = gl.minimum(context + local_block * Q_TOKENS + Q_TOKENS + 1, seq_len)
    hi = gl.minimum((seg + 1) * tiles_per_seg, (max_prefix + KV_TILE - 1) // KV_TILE)

    q_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0]
    )
    kv_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[4, 16],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0]
    )
    if MMA_FP16:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=2, instr_shape=[16, 16, 16],
            transposed=True, warps_per_cta=[NUM_WARPS // MMA_WARPS_N, MMA_WARPS_N]
        )
    else:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=2, instr_shape=[16, 16, 16],
            transposed=True, warps_per_cta=[NUM_WARPS // MMA_WARPS_N, MMA_WARPS_N]
        )
    out_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 2], threads_per_warp=[16, 4],
        warps_per_cta=[NUM_WARPS, 1], order=[1, 0]
    )
    qm = gl.arange(0, Q_ROWS, layout=gl.SliceLayout(1, q_layout))
    qd = gl.arange(0, 128, layout=gl.SliceLayout(0, q_layout))
    qpos = local_block * Q_TOKENS + qm // NQ_PER_KV
    qhead = kv_head * NQ_PER_KV + qm % NQ_PER_KV
    qvalid = (qpos < q_len) & (qhead < NUM_QHEADS) & (qm < Q_TOKENS * NQ_PER_KV)
    qptr = Q + (q_start + qpos[:, None]) * Q_STRIDE0 + qhead[:, None] * Q_STRIDE1 + qd[None, :]
    q0 = gl.load(qptr, mask=qvalid[:, None], other=0.0).to(MMA_DT)
    q1 = gl.load(qptr + 128, mask=qvalid[:, None], other=0.0).to(MMA_DT)
    if QK_INT8:
        # Separate query scales match the existing two 128-element KV groups.
        # PV keeps its BF16/FP16 MFMA path and FP32 softmax accumulation.
        qs0 = gl.maximum(gl.max(gl.abs(q0.to(gl.float32)), axis=1) / 127.0, 1.0e-6)
        qs1 = gl.maximum(gl.max(gl.abs(q1.to(gl.float32)), axis=1) / 127.0, 1.0e-6)
        qr0 = q0.to(gl.float32) / qs0[:, None]
        qr1 = q1.to(gl.float32) / qs1[:, None]
        q0 = (qr0 + gl.where(qr0 >= 0, 0.5, -0.5)).to(gl.int8)
        q1 = (qr1 + gl.where(qr1 >= 0, 0.5, -0.5)).to(gl.int8)
        qs0 = gl.convert_layout(qs0, gl.SliceLayout(1, mma))
        qs1 = gl.convert_layout(qs1, gl.SliceLayout(1, mma))
    QK_WIDTH: gl.constexpr = 4 if QK_INT8 else 2
    K_DTYPE: gl.constexpr = gl.int8 if QK_INT8 else MMA_DT
    q0 = gl.convert_layout(q0, gl.DotOperandLayout(0, mma, k_width=QK_WIDTH))
    q1 = gl.convert_layout(q1, gl.DotOperandLayout(0, mma, k_width=QK_WIDTH))

    score_rows = gl.arange(0, Q_ROWS, layout=gl.SliceLayout(1, mma))
    score_cols = gl.arange(0, KV_TILE, layout=gl.SliceLayout(0, mma))
    score_qpos = local_block * Q_TOKENS + score_rows // NQ_PER_KV
    score_qhead = kv_head * NQ_PER_KV + score_rows % NQ_PER_KV
    score_qvalid = ((score_qpos < q_len) & (score_qhead < NUM_QHEADS)
                    & (score_rows < Q_TOKENS * NQ_PER_KV))
    m = gl.full((Q_ROWS,), float('-inf'), gl.float32, layout=gl.SliceLayout(1, mma))
    l = gl.full((Q_ROWS,), 1.0, gl.float32, layout=gl.SliceLayout(1, mma))
    acc0 = gl.full((Q_ROWS, 128), 0.0, gl.float32, layout=mma)
    acc1 = gl.full((Q_ROWS, 128), 0.0, gl.float32, layout=mma)

    tn = gl.arange(0, KV_TILE, layout=gl.SliceLayout(1, kv_layout))
    kd = gl.arange(0, 128, layout=gl.SliceLayout(0, kv_layout))
    for j in range(lo, hi):
        slot = (j * KV_TILE) % BLOCK_SIZE
        # Widen before multiplying: the KV arena can exceed 2 GiB even
        # though its block IDs fit in int32.
        physical = gl.load(
            BT + seq * BT_STRIDE + (j * KV_TILE) // BLOCK_SIZE
        ).to(gl.int64)
        kv_valid = gl.minimum(KV_TILE, max_prefix - j * KV_TILE)
        valid_t = tn < kv_valid
        kb = K + physical * K_STRIDE0 + kv_head * K_STRIDE2 + slot * K_STRIDE1
        off_k = tn[:, None] * K_STRIDE1 + kd[None, :]
        if kv_valid == KV_TILE:
            k0 = gl.load(kb + off_k).to(K_DTYPE)
            k1 = gl.load(kb + 128 + off_k).to(K_DTYPE)
        else:
            k0 = gl.load(kb + off_k, mask=valid_t[:, None], other=0).to(K_DTYPE)
            k1 = gl.load(kb + 128 + off_k, mask=valid_t[:, None], other=0).to(K_DTYPE)
        k0 = gl.convert_layout(gl.permute(k0, (1, 0)), gl.DotOperandLayout(1, mma, k_width=QK_WIDTH))
        k1 = gl.convert_layout(gl.permute(k1, (1, 0)), gl.DotOperandLayout(1, mma, k_width=QK_WIDTH))
        sptr_k = SK + physical * S_STRIDE0 + kv_head * S_STRIDE2 + (slot + tn) * S_STRIDE1
        if kv_valid == KV_TILE:
            sk0 = gl.load(sptr_k).to(gl.float32)
            sk1 = gl.load(sptr_k + 1).to(gl.float32)
        else:
            sk0 = gl.load(sptr_k, mask=valid_t, other=1.0).to(gl.float32)
            sk1 = gl.load(sptr_k + 1, mask=valid_t, other=1.0).to(gl.float32)
        sk0 = gl.convert_layout(sk0, gl.SliceLayout(0, mma))
        sk1 = gl.convert_layout(sk1, gl.SliceLayout(0, mma))
        SCORE_DTYPE: gl.constexpr = gl.int32 if QK_INT8 else gl.float32
        score0 = gl.full((Q_ROWS, KV_TILE), 0, SCORE_DTYPE, layout=mma)
        score1 = gl.full((Q_ROWS, KV_TILE), 0, SCORE_DTYPE, layout=mma)
        score0 = gl.amd.cdna3.mfma(q0, k0, score0)
        score1 = gl.amd.cdna3.mfma(q1, k1, score1)
        if QK_INT8:
            scores = (score0.to(gl.float32) * (SCALE * sk0[None, :]) * qs0[:, None]
                      + score1.to(gl.float32) * (SCALE * sk1[None, :]) * qs1[:, None])
        else:
            scores = score0 * (SCALE * sk0[None, :]) + score1 * (SCALE * sk1[None, :])
        kvpos = j * KV_TILE + score_cols
        mask = score_qvalid[:, None] & (kvpos[None, :] < max_prefix) & (kvpos[None, :] <= context + score_qpos[:, None])
        scores = gl.where(mask, scores, float('-inf'))
        m_new = gl.maximum(m, gl.max(scores, axis=1))
        m_new = gl.where(m_new > float('-inf'), m_new, 0.0)
        p = gl.exp(scores - m_new[:, None])
        alpha = gl.exp(m - m_new)
        l = l * alpha + gl.sum(p, axis=1)
        m = m_new
        acc0 = acc0 * alpha[:, None]
        acc1 = acc1 * alpha[:, None]
        vb = V + physical * V_STRIDE0 + kv_head * V_STRIDE2 + slot * V_STRIDE1
        off_v = tn[:, None] * V_STRIDE1 + kd[None, :]
        if kv_valid == KV_TILE:
            v0 = gl.load(vb + off_v).to(MMA_DT)
            v1 = gl.load(vb + 128 + off_v).to(MMA_DT)
        else:
            v0 = gl.load(vb + off_v, mask=valid_t[:, None], other=0).to(MMA_DT)
            v1 = gl.load(vb + 128 + off_v, mask=valid_t[:, None], other=0).to(MMA_DT)
        v0 = gl.convert_layout(v0, gl.DotOperandLayout(1, mma, k_width=4))
        v1 = gl.convert_layout(v1, gl.DotOperandLayout(1, mma, k_width=4))
        sptr_v = SV + physical * S_STRIDE0 + kv_head * S_STRIDE2 + (slot + tn) * S_STRIDE1
        if kv_valid == KV_TILE:
            sv0 = gl.load(sptr_v).to(gl.float32)
            sv1 = gl.load(sptr_v + 1).to(gl.float32)
        else:
            sv0 = gl.load(sptr_v, mask=valid_t, other=1.0).to(gl.float32)
            sv1 = gl.load(sptr_v + 1, mask=valid_t, other=1.0).to(gl.float32)
        sv0 = gl.convert_layout(sv0, gl.SliceLayout(0, mma))
        sv1 = gl.convert_layout(sv1, gl.SliceLayout(0, mma))
        p0 = gl.convert_layout((p * sv0[None, :]).to(MMA_DT), gl.DotOperandLayout(0, mma, k_width=4))
        p1 = gl.convert_layout((p * sv1[None, :]).to(MMA_DT), gl.DotOperandLayout(0, mma, k_width=4))
        acc0 = gl.amd.cdna3.mfma(p0, v0, acc0)
        acc1 = gl.amd.cdna3.mfma(p1, v1, acc1)

    acc0 = gl.convert_layout(acc0, out_layout)
    acc1 = gl.convert_layout(acc1, out_layout)
    om = gl.arange(0, Q_ROWS, layout=gl.SliceLayout(1, out_layout))
    od = gl.arange(0, 128, layout=gl.SliceLayout(0, out_layout))
    opos = local_block * Q_TOKENS + om // NQ_PER_KV
    ohead = kv_head * NQ_PER_KV + om % NQ_PER_KV
    ovalid = (opos < q_len) & (ohead < NUM_QHEADS) & (om < Q_TOKENS * NQ_PER_KV)
    base = PARTIAL + (q_start + opos[:, None]) * (NUM_QHEADS * SPLITS * 256) + ohead[:, None] * (SPLITS * 256) + seg * 256 + od[None, :]
    gl.store(base, acc0, mask=ovalid[:, None])
    gl.store(base + 128, acc1, mask=ovalid[:, None])
    moff = (q_start + score_qpos) * (NUM_QHEADS * SPLITS) + score_qhead * SPLITS + seg
    gl.store(PMAX + moff, m, mask=score_qvalid)
    gl.store(PSUM + moff, l, mask=score_qvalid)
