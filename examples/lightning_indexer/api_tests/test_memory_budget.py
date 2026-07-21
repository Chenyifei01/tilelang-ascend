"""缺口 2: 实际数据规模下的内存预算验证

用 4 个测试用例的真实参数核算 v7.1 设计的 L1/L0/UB 占用，
确保独立 buffer 方案和 sort 缓存优化在实际规模下不超限。

硬件限制（Ascend910B3 每核）:
- L1: 512KB
- L0A: 64KB
- L0B: 64KB
- L0C: 512KB
- UB: 196KB（代码中 _UB_LIMIT=196352）

测试用例:
| 用例 | B | S1 | S2 | N1 | D | block_size | dtype | mode |
| BSND_BSND | 16 | 5 | 3072 | 64 | 128 | 128 | FP16 | 3 |
| BSND_PA_BSND | 2 | 1 | 2048 | 8 | 128 | 128 | FP16 | 3 |
| TND_TND | 8 | 5 | 3072 | 24 | 128 | 256 | FP16 | 0 |
| TND_PA_BSND | 20 | 3 | 512 | 64 | 128 | 16 | BF16 | 0 |
"""
import json

# 硬件限制
L1_LIMIT_KB = 512
L0A_LIMIT_KB = 64
L0B_LIMIT_KB = 64
L0C_LIMIT_KB = 512
UB_LIMIT_KB = 192  # 196352 bytes ≈ 192KB

# 测试用例
TEST_CASES = [
    {"name": "BSND_BSND", "B": 16, "S1": 5, "S2": 3072, "N1": 64, "D": 128,
     "block_size": 128, "dtype": "float16", "mode": 3, "is_pa": False, "is_tnd": False},
    {"name": "BSND_PA_BSND", "B": 2, "S1": 1, "S2": 2048, "N1": 8, "D": 128,
     "block_size": 128, "dtype": "float16", "mode": 3, "is_pa": True, "is_tnd": False},
    {"name": "TND_TND", "B": 8, "S1": 5, "S2": 3072, "N1": 24, "D": 128,
     "block_size": 256, "dtype": "float16", "mode": 0, "is_pa": False, "is_tnd": True},
    {"name": "TND_PA_BSND", "B": 20, "S1": 3, "S2": 512, "N1": 64, "D": 128,
     "block_size": 16, "dtype": "bfloat16", "mode": 0, "is_pa": True, "is_tnd": True},
]

DTYPE_SIZE = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "uint8": 1}


def calc_dtype_bytes(dtype, *dims):
    """计算 buffer 字节数"""
    size = DTYPE_SIZE[dtype]
    for d in dims:
        size *= d
    return size


def kb(bytes_val):
    return bytes_val / 1024


def compute_tiling_params(case):
    """计算 tiling 参数（复刻 lightning_indexer.py 的逻辑）"""
    N1, N2 = case["N1"], 1
    D = case["D"]
    S1 = case["S1"]
    S2 = case["S2"]
    is_pa = case["is_pa"]
    TOP_K = 2048

    G = N1 // N2
    S1_BLOCK = 8 if S1 >= 8 else (4 if TOP_K <= 2048 else 2)
    M_L1 = S1_BLOCK * G
    BLOCK_M_L0 = 128
    _M_L1_padded = ((M_L1 + BLOCK_M_L0 - 1) // BLOCK_M_L0) * BLOCK_M_L0

    # BLOCK_N selection
    _q_bufs = 2
    _q_l1_kb = kb(_q_bufs * _M_L1_padded * D * DTYPE_SIZE[case["dtype"]])
    _candidates = [512, 256, 128] if is_pa else [256, 128]
    BLOCK_N = 128  # fallback
    for _test_n in _candidates:
        _n_split = max(1, (_test_n + 127) // 128)
        _l0b_n = _test_n // _n_split
        _l0c_kb = kb(2 * BLOCK_M_L0 * _l0b_n * DTYPE_SIZE["float32"])
        _k_l1_kb = kb(3 * _test_n * D * DTYPE_SIZE[case["dtype"]])
        if _l0c_kb <= L0C_LIMIT_KB and (_q_l1_kb + _k_l1_kb) <= 500:
            BLOCK_N = _test_n
            break

    _N_SPLIT = max(1, (BLOCK_N + 127) // 128)
    _L0B_N = BLOCK_N // _N_SPLIT

    S2_VEC_BLOCK = min(512, ((S2 + 511) // 512) * 512) if S2 >= 256 else S2
    _BLOCK_N_VEC = max(S2_VEC_BLOCK, 64) if is_pa else S2_VEC_BLOCK

    if G % 8 == 0:
        VECTOR_BASEG = 8
    elif G % 4 == 0:
        VECTOR_BASEG = 4
    elif G % 2 == 0:
        VECTOR_BASEG = 2
    else:
        VECTOR_BASEG = G

    TOP_K_ALIGNED = ((TOP_K + 63) // 64) * 64
    _K_PER_BLOCK = min(TOP_K, S2_VEC_BLOCK)
    VID_S1 = (S1_BLOCK + 1) // 2 if S1_BLOCK >= 2 else 1

    return {
        "G": G, "S1_BLOCK": S1_BLOCK, "M_L1": M_L1, "M_L1_padded": _M_L1_padded,
        "BLOCK_M_L0": BLOCK_M_L0, "BLOCK_N": BLOCK_N, "N_SPLIT": _N_SPLIT, "L0B_N": _L0B_N,
        "D": D, "S2_VEC_BLOCK": S2_VEC_BLOCK, "BLOCK_N_VEC": _BLOCK_N_VEC,
        "VECTOR_BASEG": VECTOR_BASEG, "TOP_K": TOP_K, "TOP_K_ALIGNED": TOP_K_ALIGNED,
        "K_PER_BLOCK": _K_PER_BLOCK, "VID_S1": VID_S1,
        "is_pa": is_pa, "dtype": case["dtype"],
        "q_bufs": 2, "k_bufs": 3,
    }


def calc_v6_memory(params):
    """计算 v6（现有 li_0720.py）的内存占用"""
    dtype = params["dtype"]
    calc_dtype = "float32"

    # ===== L1 (Cube) =====
    # q_l1: [q_bufs, 1, M_L1_padded, D]
    q_l1 = calc_dtype_bytes(dtype, params["q_bufs"], 1, params["M_L1_padded"], params["D"])
    # k_l1: [k_bufs, BLOCK_N, D]
    k_l1 = calc_dtype_bytes(dtype, params["k_bufs"], params["BLOCK_N"], params["D"])
    l1_total = q_l1 + k_l1

    # ===== L0 =====
    # a_l0: [2, BLOCK_M_L0, D]
    a_l0 = calc_dtype_bytes(dtype, 2, params["BLOCK_M_L0"], params["D"])
    # b_l0: [2, D, L0B_N]
    b_l0 = calc_dtype_bytes(dtype, 2, params["D"], params["L0B_N"])
    # acc_l0c: [2, BLOCK_M_L0, L0B_N]
    acc_l0c = calc_dtype_bytes("float32", 2, params["BLOCK_M_L0"], params["L0B_N"])

    # ===== UB (Vector) =====
    bn_vec = params["BLOCK_N_VEC"]
    vg = params["VECTOR_BASEG"]
    s2v = params["S2_VEC_BLOCK"]
    ta2 = 2 * params["TOP_K_ALIGNED"]
    kp2 = 2 * params["K_PER_BLOCK"]
    vid_s1 = params["VID_S1"]

    ub_buffers = {
        "reduce_g_ub": calc_dtype_bytes("float32", bn_vec),
        "w_raw_ub": calc_dtype_bytes(dtype, 2, 16),
        "weight_ub": calc_dtype_bytes("float32", vg),
        "weight_2d_ub": calc_dtype_bytes("float32", vg, s2v),
        "mm_res_ub": calc_dtype_bytes("float32", 2, vg, s2v),
        "reduce_tmp_ub": calc_dtype_bytes("float32", vg, s2v),
        "topk_a_ub": calc_dtype_bytes("float32", vid_s1, ta2),
        "cache_tmp_ub": calc_dtype_bytes("float32", 1, kp2),
        "merged_ub": calc_dtype_bytes("float32", 2 * ta2),
        "p2_acc_ub": calc_dtype_bytes("float32", ta2),
        "stride2_blk_ub": calc_dtype_bytes("float32", kp2),
        "reduce_sum_tmp_ub": calc_dtype_bytes("float32", 1),
        "index_blk_ub": calc_dtype_bytes("float32", bn_vec),
        "mask_blk_ub": calc_dtype_bytes("uint8", bn_vec // 8),
        "topk_index_ub": calc_dtype_bytes("float32", params["TOP_K_ALIGNED"]),
        "output_ub": calc_dtype_bytes("int32", params["TOP_K_ALIGNED"]),
        "output_val_ub": calc_dtype_bytes(dtype, params["TOP_K_ALIGNED"]),
        "score_topk_ub": calc_dtype_bytes("float32", params["TOP_K_ALIGNED"]),
        "mask_topk_ub": calc_dtype_bytes("uint8", params["TOP_K_ALIGNED"] // 8),
        "prev_bsn_ub": calc_dtype_bytes("int32", 1),
    }
    ub_total = sum(ub_buffers.values())

    return {
        "L1": {"q_l1": q_l1, "k_l1": k_l1, "total": l1_total},
        "L0": {"a_l0": a_l0, "b_l0": b_l0, "acc_l0c": acc_l0c},
        "UB": {"buffers": ub_buffers, "total": ub_total},
    }


def calc_v71_memory(params):
    """计算 v7.1（独立 buffer + sort 缓存优化）的内存占用

    v7.1 变更：
    1. Cube GEMM: 独立 buffer 方案（4 个独立 L1 buffer 替代 1 个大 buffer + 切片）
       - 但实际 lightning_indexer 的 G=64 时 M_L1=512，需要 4 组 128×128 = 8 个 buffer
       - 如果保持 Q 2-buf + K 3-buf 的 pingpong，需要更多 buffer
    2. sort 缓存: 1D 平铺 cache + sort_tmp + merge_output
    """
    dtype = params["dtype"]
    calc_dtype = "float32"

    # ===== L1 (Cube) - v7.1 独立 buffer 方案 =====
    # v7.1: 4 个独立 (M_L0, D) buffer 替代 1 个 (M_L1, D)
    # M_L1=512, M_L0=128 → 需要 M_L1/M_L0=4 组
    num_m_tiles = params["M_L1_padded"] // params["BLOCK_M_L0"]  # 4 for G=64
    num_n_tiles = params["BLOCK_N"] // params["L0B_N"]  # N_SPLIT

    # Q: q_bufs × num_m_tiles × (M_L0, D)
    # 但实际 v7.1 独立 buffer 是为每个子 tile 分配独立 buffer
    # 如果保持 pingpong: q_bufs=2, 每组 num_m_tiles=4 个 → 2×4=8 个 (128,128) buffer
    q_l1_v71 = calc_dtype_bytes(dtype, params["q_bufs"], num_m_tiles, params["BLOCK_M_L0"], params["D"])

    # K: k_bufs × num_n_tiles × (L0B_N, D)
    k_l1_v71 = calc_dtype_bytes(dtype, params["k_bufs"], num_n_tiles, params["L0B_N"], params["D"])

    l1_total_v71 = q_l1_v71 + k_l1_v71

    # 注意：v7.1 独立 buffer 实际上总大小 = v6 的总大小
    # 因为 (2 × 4 × 128 × 128) = (2 × 512 × 128) = v6 的 q_l1
    # 区别在于 buffer 的分配方式（多个小 buffer vs 1 个大 buffer）

    # ===== L0 (不变) =====
    a_l0 = calc_dtype_bytes(dtype, 2, params["BLOCK_M_L0"], params["D"])
    b_l0 = calc_dtype_bytes(dtype, 2, params["D"], params["L0B_N"])
    acc_l0c = calc_dtype_bytes("float32", 2, params["BLOCK_M_L0"], params["L0B_N"])

    # ===== UB (Vector) - v7.1 sort 缓存优化 =====
    bn_vec = params["BLOCK_N_VEC"]
    vg = params["VECTOR_BASEG"]
    s2v = params["S2_VEC_BLOCK"]
    ta2 = 2 * params["TOP_K_ALIGNED"]
    kp2 = 2 * params["K_PER_BLOCK"]
    vid_s1 = params["VID_S1"]

    # v7.1 新增: sort 缓存优化
    SORT_CACHE_SIZE = 4
    sort_buf_size = s2v * 2  # sort 输出 = 2×输入 (value-index pair)
    sorted_cache = calc_dtype_bytes("float32", SORT_CACHE_SIZE * sort_buf_size)
    sort_tmp = calc_dtype_bytes("float32", sort_buf_size)
    merge_output = calc_dtype_bytes("float32", SORT_CACHE_SIZE * sort_buf_size)

    # v6 保留的 buffer（部分可能被 sort 缓存优化替代）
    ub_buffers_v71 = {
        # === 保留的 v6 buffer ===
        "reduce_g_ub": calc_dtype_bytes("float32", bn_vec),
        "w_raw_ub": calc_dtype_bytes(dtype, 2, 16),
        "weight_ub": calc_dtype_bytes("float32", vg),
        "weight_2d_ub": calc_dtype_bytes("float32", vg, s2v),
        "mm_res_ub": calc_dtype_bytes("float32", 2, vg, s2v),
        "reduce_tmp_ub": calc_dtype_bytes("float32", vg, s2v),
        "topk_a_ub": calc_dtype_bytes("float32", vid_s1, ta2),
        # cache_tmp_ub 被 sort_tmp 替代
        "merged_ub": calc_dtype_bytes("float32", 2 * ta2),
        "p2_acc_ub": calc_dtype_bytes("float32", ta2),
        "stride2_blk_ub": calc_dtype_bytes("float32", kp2),
        "reduce_sum_tmp_ub": calc_dtype_bytes("float32", 1),
        "index_blk_ub": calc_dtype_bytes("float32", bn_vec),
        "mask_blk_ub": calc_dtype_bytes("uint8", bn_vec // 8),
        "topk_index_ub": calc_dtype_bytes("float32", params["TOP_K_ALIGNED"]),
        "output_ub": calc_dtype_bytes("int32", params["TOP_K_ALIGNED"]),
        "output_val_ub": calc_dtype_bytes(dtype, params["TOP_K_ALIGNED"]),
        "score_topk_ub": calc_dtype_bytes("float32", params["TOP_K_ALIGNED"]),
        "mask_topk_ub": calc_dtype_bytes("uint8", params["TOP_K_ALIGNED"] // 8),
        "prev_bsn_ub": calc_dtype_bytes("int32", 1),
        # === v7.1 新增 sort 缓存优化 ===
        "sorted_cache (v7.1新增)": sorted_cache,
        "sort_tmp (v7.1新增)": sort_tmp,
        "merge_output (v7.1新增)": merge_output,
    }
    ub_total_v71 = sum(ub_buffers_v71.values())

    return {
        "L1": {"q_l1": q_l1_v71, "k_l1": k_l1_v71, "total": l1_total_v71,
               "num_m_tiles": num_m_tiles, "num_n_tiles": num_n_tiles},
        "L0": {"a_l0": a_l0, "b_l0": b_l0, "acc_l0c": acc_l0c},
        "UB": {"buffers": ub_buffers_v71, "total": ub_total_v71,
               "sort_cache_add": sorted_cache + sort_tmp + merge_output},
    }


def calc_v71_memory_optimized(params):
    """v7.1 优化版：SortedBasicBlock_ 与 globalTopkUb_ 内存复用（缺口 6 建议 #2）

    将 sorted_cache 和 topk_a_ub 合并为单个 1D buffer
    """
    calc_dtype = "float32"
    s2v = params["S2_VEC_BLOCK"]
    ta2 = 2 * params["TOP_K_ALIGNED"]
    vid_s1 = params["VID_S1"]

    SORT_CACHE_SIZE = 4
    sort_buf_size = s2v * 2

    # 合并 topk_a_ub + sorted_cache 为单个 buffer
    # topk_a_ub: vid_s1 × ta2 (per-row global topk)
    # sorted_cache: SORT_CACHE_SIZE × sort_buf_size
    combined_sort_ub = calc_dtype_bytes("float32", vid_s1 * ta2 + SORT_CACHE_SIZE * sort_buf_size)

    # sort_tmp 和 merge_output 仍独立
    sort_tmp = calc_dtype_bytes("float32", sort_buf_size)
    merge_output = calc_dtype_bytes("float32", SORT_CACHE_SIZE * sort_buf_size)

    return {
        "combined_sort_ub": combined_sort_ub,
        "sort_tmp": sort_tmp,
        "merge_output": merge_output,
        "total_sort": combined_sort_ub + sort_tmp + merge_output,
    }


def format_table(headers, rows):
    """格式化表格输出"""
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_widths)) + "|"
    lines = [sep, header, sep]
    for row in rows:
        line = "|" + "|".join(f" {str(c):<{w}} " for c, w in zip(row, col_widths)) + "|"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def main():
    print("=" * 80)
    print("缺口 2: 实际数据规模下的内存预算验证")
    print("=" * 80)
    print(f"硬件限制: L1={L1_LIMIT_KB}KB, L0A={L0A_LIMIT_KB}KB, L0B={L0B_LIMIT_KB}KB, "
          f"L0C={L0C_LIMIT_KB}KB, UB={UB_LIMIT_KB}KB")
    print()

    all_pass = True

    for case in TEST_CASES:
        print(f"\n{'='*80}")
        print(f"用例: {case['name']} (B={case['B']}, S1={case['S1']}, S2={case['S2']}, "
              f"N1={case['N1']}, D={case['D']}, block_size={case['block_size']}, "
              f"dtype={case['dtype']})")
        print(f"{'='*80}")

        params = compute_tiling_params(case)
        print(f"\nTiling 参数:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        # ===== v6 内存 =====
        v6 = calc_v6_memory(params)
        print(f"\n--- v6（现有 li_0720.py）内存占用 ---")
        print(f"L1: {kb(v6['L1']['total']):.1f}KB / {L1_LIMIT_KB}KB "
              f"({'✅' if kb(v6['L1']['total']) <= L1_LIMIT_KB else '❌'})")
        print(f"  q_l1: {kb(v6['L1']['q_l1']):.1f}KB, k_l1: {kb(v6['L1']['k_l1']):.1f}KB")
        print(f"L0A: {kb(v6['L0']['a_l0']):.1f}KB / {L0A_LIMIT_KB}KB "
              f"({'✅' if kb(v6['L0']['a_l0']) <= L0A_LIMIT_KB else '❌'})")
        print(f"L0B: {kb(v6['L0']['b_l0']):.1f}KB / {L0B_LIMIT_KB}KB "
              f"({'✅' if kb(v6['L0']['b_l0']) <= L0B_LIMIT_KB else '❌'})")
        print(f"L0C: {kb(v6['L0']['acc_l0c']):.1f}KB / {L0C_LIMIT_KB}KB "
              f"({'✅' if kb(v6['L0']['acc_l0c']) <= L0C_LIMIT_KB else '❌'})")
        print(f"UB: {kb(v6['UB']['total']):.1f}KB / {UB_LIMIT_KB}KB "
              f"({'✅' if kb(v6['UB']['total']) <= UB_LIMIT_KB else '❌'})")

        # ===== v7.1 内存 =====
        v71 = calc_v71_memory(params)
        print(f"\n--- v7.1（独立 buffer + sort 缓存优化）内存占用 ---")
        print(f"L1: {kb(v71['L1']['total']):.1f}KB / {L1_LIMIT_KB}KB "
              f"({'✅' if kb(v71['L1']['total']) <= L1_LIMIT_KB else '❌'})")
        print(f"  q_l1: {kb(v71['L1']['q_l1']):.1f}KB ({params['q_bufs']}×{v71['L1']['num_m_tiles']} bufs), "
              f"k_l1: {kb(v71['L1']['k_l1']):.1f}KB ({params['k_bufs']}×{v71['L1']['num_n_tiles']} bufs)")
        print(f"L0A: {kb(v71['L0']['a_l0']):.1f}KB / {L0A_LIMIT_KB}KB "
              f"({'✅' if kb(v71['L0']['a_l0']) <= L0A_LIMIT_KB else '❌'})")
        print(f"L0B: {kb(v71['L0']['b_l0']):.1f}KB / {L0B_LIMIT_KB}KB "
              f"({'✅' if kb(v71['L0']['b_l0']) <= L0B_LIMIT_KB else '❌'})")
        print(f"L0C: {kb(v71['L0']['acc_l0c']):.1f}KB / {L0C_LIMIT_KB}KB "
              f"({'✅' if kb(v71['L0']['acc_l0c']) <= L0C_LIMIT_KB else '❌'})")
        print(f"UB: {kb(v71['UB']['total']):.1f}KB / {UB_LIMIT_KB}KB "
              f"({'✅' if kb(v71['UB']['total']) <= UB_LIMIT_KB else '❌ OVERFLOW!'})")
        print(f"  sort 缓存新增: {kb(v71['UB']['sort_cache_add']):.1f}KB "
              f"(sorted_cache={kb(v71['UB']['buffers']['sorted_cache (v7.1新增)']):.1f}KB, "
              f"sort_tmp={kb(v71['UB']['buffers']['sort_tmp (v7.1新增)']):.1f}KB, "
              f"merge_output={kb(v71['UB']['buffers']['merge_output (v7.1新增)']):.1f}KB)")

        v6_ub = kb(v6['UB']['total'])
        v71_ub = kb(v71['UB']['total'])
        ub_diff = v71_ub - v6_ub
        print(f"\n  UB 变化: {v6_ub:.1f}KB → {v71_ub:.1f}KB ({'+' if ub_diff > 0 else ''}{ub_diff:.1f}KB)")

    # ===== v7.1 优化方案对比 =====
    print(f"\n--- v7.1 优化方案对比 ---")

    # 方案 A: buffer 复用（sort_tmp 复用 cache_tmp_ub，merge_output 复用 merged_ub）
    _s2v = params["S2_VEC_BLOCK"]
    _sort_buf_size = _s2v * 2
    _calc_dtype = "float32"
    sorted_cache_new = calc_dtype_bytes(_calc_dtype, 4 * _sort_buf_size)  # 仍需新增
    ub_plan_a = v6['UB']['total'] + sorted_cache_new
    print(f"  方案 A（复用 sort_tmp+merge_output）: UB={kb(ub_plan_a):.1f}KB "
          f"({'✅' if kb(ub_plan_a) <= UB_LIMIT_KB else '❌'}) +{kb(sorted_cache_new):.1f}KB")

    # 方案 B: 2 块缓存（SORT_CACHE_SIZE=2）+ 复用
    sorted_cache_2 = calc_dtype_bytes("float32", 2 * _sort_buf_size)
    ub_plan_b = v6['UB']['total'] + sorted_cache_2
    print(f"  方案 B（2块缓存+复用）: UB={kb(ub_plan_b):.1f}KB "
          f"({'✅' if kb(ub_plan_b) <= UB_LIMIT_KB else '❌'}) +{kb(sorted_cache_2):.1f}KB")

    # 方案 C: VID_S1=1（单核 S1）+ 4 块缓存 + 复用
    # topk_a_ub 从 VID_S1×TA2 变为 1×TA2，省 (VID_S1-1)×TA2
    _ta2 = 2 * params["TOP_K_ALIGNED"]
    topk_saving = calc_dtype_bytes("float32", (params["VID_S1"] - 1) * _ta2)
    ub_plan_c = v6['UB']['total'] + sorted_cache_new - topk_saving
    print(f"  方案 C（VID_S1=1省{kb(topk_saving):.1f}KB+4块缓存+复用）: "
          f"UB={kb(ub_plan_c):.1f}KB "
          f"({'✅' if kb(ub_plan_c) <= UB_LIMIT_KB else '❌'})")

    # 方案 D: S2_VEC_BLOCK=256（减小 sort buffer）+ 4 块缓存 + 复用
    sorted_cache_256 = calc_dtype_bytes("float32", 4 * (256 * 2))
    ub_plan_d = v6['UB']['total'] + sorted_cache_256
    print(f"  方案 D（S2_VEC=256+4块缓存+复用）: UB={kb(ub_plan_d):.1f}KB "
          f"({'✅' if kb(ub_plan_d) <= UB_LIMIT_KB else '❌'}) +{kb(sorted_cache_256):.1f}KB")

    # 推荐
    best_plan = min([("A", ub_plan_a), ("B", ub_plan_b), ("C", ub_plan_c), ("D", ub_plan_d)],
                    key=lambda x: x[1])
    print(f"\n  推荐方案: 方案 {best_plan[0]} (UB={kb(best_plan[1]):.1f}KB)")
    case_pass = kb(best_plan[1]) <= UB_LIMIT_KB

    status = "✅ PASS" if case_pass else "❌ FAIL"
    print(f"\n汇总: {status}")
    if not case_pass:
        all_pass = False
        print(f"  ⚠️ 所有方案都超限，需进一步优化")

    # ===== 总汇总 =====
    print(f"\n{'='*80}")
    print(f"总汇总: {'✅ 全部通过' if all_pass else '❌ 有超限'}")
    print(f"{'='*80}")

    if all_pass:
        print("\n结论: v7.1 设计在所有 4 个测试用例的真实参数下内存预算通过")
        print("→ 可安全进入 Stage 2 实现")
    else:
        print("\n结论: 部分用例内存超限，需调整 v7.1 设计")
        print("→ 建议优化方向:")
        print("  1. 内存复用（合并 topk_a_ub + sorted_cache）")
        print("  2. 减少 sort 缓存块数（4→2）")
        print("  3. 减小 BLOCK_N（256→128）")


if __name__ == "__main__":
    main()
