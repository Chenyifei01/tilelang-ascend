"""lightning_indexer kernel implementation (attempt 2: cast pattern fix).

Key fix over attempt 1 (cast pattern, per user instruction):
  Weights now enter as input_dtype (fp16/bf16) and are cast to float32
  INSIDE the kernel via T.tile.cast(weight_ub, w_raw_ub, "CAST_NONE", G).
  This matches the AscendC arch22 reference (vector.h:88 Cast(weightsUb,
  weightsTUb, CAST_NONE, groupInner)) and li_0720.py (line 782).
  Host-side weights.float() conversion removed (saves GM bandwidth, matches
  the AscendC cast path). Q/K remain input_dtype through L1/L0A/L0B;
  T.gemm_v0 auto-accumulates to float32 in L0C (no cast needed for QK).

Retained from attempt 1 (precision fixes, verified working):
  - Per-s1 cutoff for sparse_mode=3 (rightDownCausal mask):
    golden uses per-row cutoff = act_k - act_q + s1 + 1
  - fp32→bf16→fp32 rounding before topk to match golden's
    to_be_sort_ele = reduce_sum.clone().to(torch.bfloat16)

Retained from attempt 1 (v5 fallback, verified working):
  - Expert mode (T.Scope "C"/"V", alloc_L1/L0C/ub)
  - 4D Weights tensor + per-row T.tile.mul (row_expand_mul fallback)
  - T.barrier_all sync (pipe_barrier("v") fallback)
  - T.serial (Pipelined fallback)
  - G processed in one shot (reduce_sum clear=True fallback)
"""

import torch
import tilelang
import tilelang.language as T

NUM_CORES = int(torch.npu.get_device_properties("npu").cube_core_num)

BLOCK_N = 128  # S2 block size for GEMM
MAX_S2_UB = 16384  # Single topk UB limit


@tilelang.jit(
    out_idx=[8, 9],
    workspace_idx=[6, 7],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
def lightning_indexer(
    B, S1, S2, N1, D, G, SPARSE_COUNT, MAX_S2,
    layout_query, layout_key, block_size, max_block_num,
    sparse_mode, return_value,
    input_dtype="float16", calc_dtype="float",
):
    """Single kernel: QK BMM + ReLU + weight mul + group reduce + mask + topk.

    Uses Expert mode (T.Scope C/V) with 4D tensors, matching the proven
    example_lightning_indexer.py pattern.
    """
    s2_base_num = T.ceildiv(S2, BLOCK_N)
    N2 = 1
    VECTOR_BASEN = BLOCK_N
    VECTOR_BASEG = G  # Process all G at once (UB allows for G<=64, BLOCK_N=128)

    is_pa_key = layout_key == "PA_BSND"

    # Key shape depends on layout (Python-level, compile-time)
    if is_pa_key:
        k_shape = (max_block_num, block_size, N2, D)
    else:
        k_shape = (B, S2, N2, D)

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, D), input_dtype),
        Key: T.Tensor(k_shape, input_dtype),
        Weights: T.Tensor((B, S1, N2, G), input_dtype),
        actual_seq_q: T.Tensor((B,), "int32"),
        actual_seq_k: T.Tensor((B,), "int32"),
        block_table: T.Tensor((B, max_block_num), "int32"),
        QK_RES: T.Tensor((B, N2, S1, G, S2), calc_dtype),
        score_accum_gm: T.Tensor((MAX_S2,), calc_dtype),
        Output: T.Tensor((B, S1, N2, SPARSE_COUNT), "int32"),
        Output_value: T.Tensor((B, S1, N2, SPARSE_COUNT), calc_dtype),
    ):
        with T.Kernel(B * N2, is_npu=True) as (cid, vid):
            b = cid // N2

            # Read act_seq
            act_q = actual_seq_q[b]
            act_k = actual_seq_k[b]

            with T.Scope("C"):
                Q_L1 = T.alloc_L1((G, D), input_dtype)
                K_L1 = T.alloc_L1((BLOCK_N, D), input_dtype)
                C_L0 = T.alloc_L0C((G, BLOCK_N), calc_dtype)

                T.barrier_all()
                for s1 in T.serial(S1):
                    T.barrier_all()
                    T.copy(Query[b, s1, :, :], Q_L1)
                    T.barrier_all()
                    for s2_blk in T.serial(s2_base_num):
                        s2_start = s2_blk * BLOCK_N
                        if s2_start < act_k:
                            T.barrier_all()
                            if is_pa_key:
                                k_blk_id = block_table[b, s2_blk]
                                T.copy(Key[k_blk_id, :BLOCK_N, 0, :], K_L1)
                            else:
                                T.copy(Key[b, s2_start:s2_start + BLOCK_N, 0, :], K_L1)
                            T.barrier_all()
                            T.gemm_v0(Q_L1, K_L1, C_L0, transpose_B=True, init=True)
                            T.barrier_all()
                            T.copy(
                                C_L0,
                                QK_RES[b, 0, s1, :, s2_start:s2_start + BLOCK_N],
                                enable_relu=True,
                            )
                            T.barrier_all()
                T.set_cross_flag("FIX", 0)
                T.barrier_all()

            with T.Scope("V"):
                mm_res_ub = T.alloc_ub((VECTOR_BASEG, VECTOR_BASEN), calc_dtype)
                # Cast path (attempt 2): Weights enter as input_dtype (fp16/bf16),
                # cast to calc_dtype (float32) inside kernel via T.tile.cast.
                # Matches li_0720.py line 782 + AscendC arch22 vector.h line 88.
                w_raw_ub = T.alloc_ub(VECTOR_BASEG, input_dtype)
                weight_ub = T.alloc_ub(VECTOR_BASEG, calc_dtype)
                reduce_g_ub = T.alloc_ub(VECTOR_BASEN, calc_dtype)
                score_accum_ub = T.alloc_ub(MAX_S2, calc_dtype)
                # bf16 intermediate for fp32→bf16→fp32 rounding (matches golden's
                # to_be_sort_ele = reduce_sum.clone().to(torch.bfloat16))
                score_bf16_ub = T.alloc_ub(MAX_S2, "bfloat16")
                topk_dst_ub = T.alloc_ub(2 * SPARSE_COUNT, calc_dtype)
                topk_index_ub = T.alloc_ub(SPARSE_COUNT, calc_dtype)
                output_ub = T.alloc_ub(SPARSE_COUNT, "int32")
                output_value_ub = T.alloc_ub(SPARSE_COUNT, calc_dtype)

                T.wait_cross_flag(0)
                T.barrier_all()

                for s1 in T.serial(S1):
                    if s1 < act_q:
                        T.barrier_all()
                        T.tile.fill(score_accum_ub, -T.infinity(calc_dtype))
                        T.barrier_all()

                        for s2_blk in T.serial(s2_base_num):
                            s2_start = s2_blk * BLOCK_N
                            if s2_start < act_k:
                                T.barrier_all()
                                T.copy(
                                    QK_RES[b, 0, s1, :, s2_start:s2_start + BLOCK_N],
                                    mm_res_ub,
                                )
                                T.barrier_all()
                                # Cast path: copy input_dtype weights to w_raw_ub,
                                # then T.tile.cast (CAST_NONE, lossless bf16/fp16→fp32)
                                # to weight_ub (float32). Matches li_0720.py line 770-782.
                                T.copy(Weights[b, s1, 0, :], w_raw_ub)
                                T.barrier_all()
                                T.tile.cast(weight_ub, w_raw_ub, "CAST_NONE", VECTOR_BASEG)
                                T.barrier_all()
                                # Weight mul per row
                                for i in range(VECTOR_BASEG):
                                    T.barrier_all()
                                    T.tile.mul(mm_res_ub[i, :], mm_res_ub[i, :], weight_ub[i])
                                    T.barrier_all()
                                T.barrier_all()
                                # Reduce over G dimension directly (no add step needed)
                                T.reduce_sum(mm_res_ub, reduce_g_ub, 0)
                                T.barrier_all()
                                # Per-s1 cutoff for sparse_mode=3 (rightDownCausal):
                                # golden's create_mask gives cutoff[s1] = act_k - act_q + s1 + 1
                                # Valid positions: j < act_k - act_q + s1 + 1
                                # (inline expression — TVM parser doesn't allow Python var
                                # assignment inside TIR for loop depending on loop var)
                                if sparse_mode == 3:
                                    for i in T.serial(VECTOR_BASEN):
                                        j = s2_start + i
                                        if j < act_k - act_q + s1 + 1:
                                            score_accum_ub[j] = reduce_g_ub[i]
                                else:
                                    for i in T.serial(VECTOR_BASEN):
                                        j = s2_start + i
                                        if j < act_k:
                                            score_accum_ub[j] = reduce_g_ub[i]
                                T.barrier_all()

                        # Round fp32 scores to bf16 precision then back to fp32,
                        # matching golden's to_be_sort_ele = reduce_sum.clone().to(torch.bfloat16).
                        # vbitsort HW intrinsic doesn't support bf16 directly, so we
                        # round via cast chain and sort the bf16-rounded fp32 values.
                        T.barrier_all()
                        T.tile.cast(score_bf16_ub, score_accum_ub, "CAST_ROUND", MAX_S2)
                        T.barrier_all()
                        T.tile.cast(score_accum_ub, score_bf16_ub, "CAST_ROUND", MAX_S2)
                        T.barrier_all()

                        # TopK on bf16-rounded fp32 scores
                        T.barrier_all()
                        T.tile.topk(topk_dst_ub, score_accum_ub, SPARSE_COUNT, act_k)
                        T.barrier_all()
                        T.tile.gather_mask(topk_index_ub, topk_dst_ub, "P1010")
                        T.barrier_all()
                        T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", SPARSE_COUNT)
                        T.barrier_all()

                        # Extract values if needed (from bf16-rounded scores)
                        if return_value:
                            for i in T.serial(SPARSE_COUNT):
                                idx = output_ub[i]
                                if idx >= 0:
                                    output_value_ub[i] = score_accum_ub[idx]

                        T.barrier_all()
                        T.copy(output_ub, Output[b, s1, 0, :SPARSE_COUNT])
                        T.barrier_all()

                        if return_value:
                            T.copy(output_value_ub, Output_value[b, s1, 0, :SPARSE_COUNT])
                            T.barrier_all()
                    else:
                        T.barrier_all()
                        T.tile.fill(output_ub, T.cast(-1, "int32"))
                        T.barrier_all()
                        T.copy(output_ub, Output[b, s1, 0, :SPARSE_COUNT])
                        T.barrier_all()

    return main
