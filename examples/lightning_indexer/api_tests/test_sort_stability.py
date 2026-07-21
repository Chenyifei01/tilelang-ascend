"""缺口 3: sort 稳定性问题的验证方案

验证 v7.1 sort 缓存优化路径下的 sort 稳定性。
之前定位的问题: 相同 score 时 kernel 的 Sort32+MrgSort 选了不同 indices，
与 golden 的 stable sort（index 小的排前面）不一致。

验证目标:
1. 构造相同 score 的输入，验证 sort 后 index 顺序
2. 对比 v6 标准路径和 v7.1 缓存路径的稳定性
3. 验证 4-way merge 是否引入新的稳定性差异

运行: python examples/lightning_indexer/api_tests/test_sort_stability.py
"""
import tilelang as tl
import tilelang.language as T
import torch
import heapq

tl.cache.clear_cache()

ELEMENT_SIZE = 2
VALUE_POS = 0
INDEX_POS = 1

S2_BLOCK = 64
CACHE_SLOTS = 4

pass_configs = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def sort_cache_4way_stability():
    """v7.1 缓存路径: sort → cache → 4-way merge

    输入: 4 个 score buffer（含相同 score）
    输出: 4-way merge 后的 value-index 对
    """
    sort_buf_size = S2_BLOCK * ELEMENT_SIZE
    cache_total = CACHE_SLOTS * sort_buf_size

    @T.prim_func
    def main(
        score0: T.Tensor([S2_BLOCK], "float32"),
        score1: T.Tensor([S2_BLOCK], "float32"),
        score2: T.Tensor([S2_BLOCK], "float32"),
        score3: T.Tensor([S2_BLOCK], "float32"),
        output: T.Tensor([CACHE_SLOTS * sort_buf_size], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _):
            s0_ub = T.alloc_ub([S2_BLOCK], "float32")
            s1_ub = T.alloc_ub([S2_BLOCK], "float32")
            s2_ub = T.alloc_ub([S2_BLOCK], "float32")
            s3_ub = T.alloc_ub([S2_BLOCK], "float32")

            sort_tmp = T.alloc_ub([sort_buf_size], "float32")
            sorted_cache = T.alloc_ub([cache_total], "float32")
            merge_output = T.alloc_ub([cache_total], "float32")

            T.copy(score0, s0_ub)
            T.copy(score1, s1_ub)
            T.copy(score2, s2_ub)
            T.copy(score3, s3_ub)

            # Block 0
            T.tile.sort(sort_tmp, s0_ub, S2_BLOCK)
            T.copy(sort_tmp, sorted_cache[0 * sort_buf_size : 1 * sort_buf_size])

            # Block 1
            T.tile.sort(sort_tmp, s1_ub, S2_BLOCK)
            T.copy(sort_tmp, sorted_cache[1 * sort_buf_size : 2 * sort_buf_size])

            # Block 2
            T.tile.sort(sort_tmp, s2_ub, S2_BLOCK)
            T.copy(sort_tmp, sorted_cache[2 * sort_buf_size : 3 * sort_buf_size])

            # Block 3
            T.tile.sort(sort_tmp, s3_ub, S2_BLOCK)
            T.copy(sort_tmp, sorted_cache[3 * sort_buf_size : 4 * sort_buf_size])

            # 4-way merge
            T.tile.merge_sort(
                merge_output,
                sorted_cache[0 * sort_buf_size : 1 * sort_buf_size],
                sorted_cache[1 * sort_buf_size : 2 * sort_buf_size],
                sorted_cache[2 * sort_buf_size : 3 * sort_buf_size],
                sorted_cache[3 * sort_buf_size : 4 * sort_buf_size],
            )

            T.copy(merge_output, output)

    return main


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def sort_merge_2way_stability():
    """v6 标准路径: sort → 2-way merge（逐块合并）"""
    sort_buf_size = S2_BLOCK * ELEMENT_SIZE
    merge_out_size = CACHE_SLOTS * sort_buf_size

    @T.prim_func
    def main(
        score0: T.Tensor([S2_BLOCK], "float32"),
        score1: T.Tensor([S2_BLOCK], "float32"),
        score2: T.Tensor([S2_BLOCK], "float32"),
        score3: T.Tensor([S2_BLOCK], "float32"),
        output: T.Tensor([merge_out_size], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _):
            s0_ub = T.alloc_ub([S2_BLOCK], "float32")
            s1_ub = T.alloc_ub([S2_BLOCK], "float32")
            s2_ub = T.alloc_ub([S2_BLOCK], "float32")
            s3_ub = T.alloc_ub([S2_BLOCK], "float32")

            topk_ub = T.alloc_ub([merge_out_size], "float32")
            sort_tmp = T.alloc_ub([sort_buf_size], "float32")
            merge_tmp = T.alloc_ub([merge_out_size], "float32")

            T.copy(score0, s0_ub)
            T.copy(score1, s1_ub)
            T.copy(score2, s2_ub)
            T.copy(score3, s3_ub)

            # Block 0: sort → topk
            T.tile.sort(topk_ub, s0_ub, S2_BLOCK)

            # Block 1: sort → 2-way merge
            T.tile.sort(sort_tmp, s1_ub, S2_BLOCK)
            T.tile.merge_sort(merge_tmp, topk_ub[0 : sort_buf_size], sort_tmp)
            T.copy(merge_tmp[0 : 2 * sort_buf_size], topk_ub[0 : 2 * sort_buf_size])

            # Block 2
            T.tile.sort(sort_tmp, s2_ub, S2_BLOCK)
            T.tile.merge_sort(merge_tmp, topk_ub[0 : 2 * sort_buf_size], sort_tmp)
            T.copy(merge_tmp[0 : 3 * sort_buf_size], topk_ub[0 : 3 * sort_buf_size])

            # Block 3
            T.tile.sort(sort_tmp, s3_ub, S2_BLOCK)
            T.tile.merge_sort(merge_tmp, topk_ub[0 : 3 * sort_buf_size], sort_tmp)
            T.copy(merge_tmp[0 : 4 * sort_buf_size], topk_ub[0 : 4 * sort_buf_size])

            T.copy(topk_ub[0 : 4 * sort_buf_size], output)

    return main


def ref_stable_sort_merge(scores):
    """Python stable sort + merge 参考

    stable sort: 相同 score 时 index 小的排前面
    """
    sorted_blocks = []
    for block_idx, score in enumerate(scores):
        # stable sort: 按 (-score, block_idx * S2_BLOCK + original_idx) 排序
        pairs = []
        for i in range(len(score)):
            # global index = block_idx * S2_BLOCK + i
            global_idx = block_idx * S2_BLOCK + i
            pairs.append((score[i].item(), global_idx))
        # stable sort: 降序 score，升序 index
        pairs.sort(key=lambda x: (-x[0], x[1]))
        sorted_blocks.append(pairs)

    # 4-way merge with stable ordering
    # 使用 heapq.merge，它保持 stable（相同 key 时保持输入顺序）
    neg_seqs = [[(-v, i) for v, i in seq] for seq in sorted_blocks]
    merged = list(heapq.merge(*neg_seqs))
    merged = [(-v, i) for v, i in merged]

    result = torch.zeros(len(merged) * ELEMENT_SIZE, dtype=torch.float32)
    for i, (v, idx) in enumerate(merged):
        result[i * ELEMENT_SIZE + VALUE_POS] = v
        result[i * ELEMENT_SIZE + INDEX_POS] = float(idx)
    return result


def check_stability(result, ref, name):
    """检查稳定性: 相同 score 时 index 顺序是否与 stable sort 一致"""
    result_cpu = result.cpu()
    n_pairs = len(result_cpu) // ELEMENT_SIZE

    # 提取 (value, index) 对
    kernel_pairs = []
    ref_pairs = []
    for i in range(n_pairs):
        kv = result_cpu[i * ELEMENT_SIZE + VALUE_POS].item()
        ki = result_cpu[i * ELEMENT_SIZE + INDEX_POS].item()
        rv = ref[i * ELEMENT_SIZE + VALUE_POS].item()
        ri = ref[i * ELEMENT_SIZE + INDEX_POS].item()
        kernel_pairs.append((kv, ki))
        ref_pairs.append((rv, ri))

    # 检查 values 是否降序
    values = [v for v, _ in kernel_pairs]
    is_sorted = all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    # 检查 index 顺序
    indices_match = kernel_pairs == ref_pairs

    # 如果不匹配，找差异
    diffs = []
    if not indices_match:
        for i, (kp, rp) in enumerate(zip(kernel_pairs, ref_pairs)):
            if kp != rp:
                # 检查是否是相同 score 下的 index 顺序差异
                if abs(kp[0] - rp[0]) < 1e-5:
                    diffs.append((i, kp, rp, "same_score_diff_index"))
                else:
                    diffs.append((i, kp, rp, "diff_score"))

    print(f"\n[{name}]")
    print(f"  Is sorted (descending): {is_sorted}")
    print(f"  Indices match stable sort: {indices_match}")
    if diffs:
        print(f"  差异数: {len(diffs)}")
        for i, kp, rp, reason in diffs[:5]:
            print(f"    [{i}] kernel=(v={kp[0]:.4f}, idx={kp[1]}), "
                  f"golden=(v={rp[0]:.4f}, idx={rp[1]}), reason={reason}")
    return is_sorted and indices_match


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("缺口 3: sort 稳定性问题的验证方案")
    print("=" * 70)
    print(f"参数: S2_BLOCK={S2_BLOCK}, CACHE_SLOTS={CACHE_SLOTS}")

    # 构造含相同 score 的输入
    # Block 0: 前 32 个 score 相同（=1.0），后 32 个递减
    scores = []
    for block_idx in range(4):
        score = torch.zeros(S2_BLOCK, dtype=torch.float32)
        for i in range(S2_BLOCK):
            if i < 32:
                score[i] = 1.0  # 相同 score
            else:
                score[i] = 0.5 - (i - 32) * 0.01  # 递减
        scores.append(score)

    scores_npu = [s.npu() for s in scores]
    ref = ref_stable_sort_merge(scores)

    results = []

    # 测试 1: v7.1 缓存路径
    print("\n--- 测试 1: v7.1 缓存路径（4-way merge） ---")
    try:
        kernel1 = sort_cache_4way_stability()
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result1 = kernel1(*scores_npu)
        torch.npu.synchronize()
        ok1 = check_stability(result1, ref, "v7.1 缓存路径")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok1 = False
    results.append(("v7.1 缓存路径", ok1))

    # 测试 2: v6 标准路径
    print("\n--- 测试 2: v6 标准路径（2-way merge） ---")
    try:
        kernel2 = sort_merge_2way_stability()
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result2 = kernel2(*scores_npu)
        torch.npu.synchronize()
        ok2 = check_stability(result2, ref, "v6 标准路径")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok2 = False
    results.append(("v6 标准路径", ok2))

    # 汇总
    print("\n" + "=" * 70)
    print("汇总:")
    for name, ok in results:
        print(f"  {name}: {'✅ 稳定' if ok else '❌ 不稳定'}")
    print("=" * 70)

    if all(ok for _, ok in results):
        print("\n结论: sort 稳定性问题已解决，两条路径都与 stable sort 一致")
    elif results[0][1] and not results[1][1]:
        print("\n结论: v7.1 缓存路径稳定，v6 标准路径不稳定（缓存路径反而更好）")
    elif not results[0][1] and results[1][1]:
        print("\n结论: v6 标准路径稳定，v7.1 缓存路径引入了新的不稳定")
    else:
        print("\n结论: 两条路径都不稳定，sort 稳定性问题仍存在")
        print("→ 需要在 sort 前/后对相同 score 的 index 做稳定化处理")


if __name__ == "__main__":
    main()
