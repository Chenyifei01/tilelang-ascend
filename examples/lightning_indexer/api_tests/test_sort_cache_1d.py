"""验证 2: sort 缓存优化替代方案（1D 平铺 buffer）

v7 设计中 sort 缓存优化存在 API 用法错误：
- T.tile.sort 不支持 BufferRegion 切片（签名仅 Buffer）
- 不存在 way 参数（路数由位置参数决定）

本验证测试替代方案：
1. sort 到临时 buffer，再 T.copy 到 1D 平铺 cache 的对应偏移
2. 满 4 块后用 4-way merge_sort 合并（切片作为 src）

模拟 v7 sort 缓存优化的完整流程：
  for s2_block in S2_blocks:
      sort(score_buf, sort_tmp)           # sort 到临时 buffer
      T.copy(sort_tmp, cache[offset])     # 搬运到 cache 偏移
      cache_idx += 1
      if cache_idx == 4:
          merge_sort(topk, cache[0:1], cache[1:2], cache[2:3], cache[3:4])
          cache_idx = 0

运行: python examples/lightning_indexer/api_tests/test_sort_cache_1d.py
"""
import tilelang as tl
import tilelang.language as T
import torch
import heapq

tl.cache.clear_cache()

ELEMENT_SIZE = 2
VALUE_POS = 0
INDEX_POS = 1

# 模拟 lightning_indexer 的 sort 场景
S2_BLOCK = 64        # 每个 S2 块的元素数（待排序）
CACHE_SLOTS = 4      # 缓存 4 个 S2 块
TOPK = 32            # 最终取 top-k

pass_configs = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def sort_cache_4way_merge():
    """模拟 v7 sort 缓存优化的完整流程

    输入: 4 个未排序的 score buffer（每个 S2_BLOCK 元素）
    输出: 4-way merge 后的 top-K（value-index 交错对）

    流程:
      1. 对每个 score buffer 调用 T.tile.sort → sort_tmp（完整 Buffer）
      2. T.copy(sort_tmp, sorted_cache[offset]) 搬运到 1D cache
      3. 4 个都完成后，用切片作为 src 调用 4-way merge_sort
    """
    sort_buf_size = S2_BLOCK * ELEMENT_SIZE  # sort 输出 = 2×输入
    cache_total = CACHE_SLOTS * sort_buf_size
    merge_out_size = CACHE_SLOTS * sort_buf_size  # merge 输出 = sum(src)

    @T.prim_func
    def main(
        score0: T.Tensor([S2_BLOCK], "float32"),
        score1: T.Tensor([S2_BLOCK], "float32"),
        score2: T.Tensor([S2_BLOCK], "float32"),
        score3: T.Tensor([S2_BLOCK], "float32"),
        output: T.Tensor([merge_out_size], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _):
            # 输入 score 搬到 UB
            s0_ub = T.alloc_ub([S2_BLOCK], "float32")
            s1_ub = T.alloc_ub([S2_BLOCK], "float32")
            s2_ub = T.alloc_ub([S2_BLOCK], "float32")
            s3_ub = T.alloc_ub([S2_BLOCK], "float32")

            # sort 临时 buffer（完整 Buffer，不能用切片）
            sort_tmp = T.alloc_ub([sort_buf_size], "float32")

            # 1D 平铺 cache buffer
            sorted_cache = T.alloc_ub([cache_total], "float32")
            merge_output = T.alloc_ub([merge_out_size], "float32")

            T.copy(score0, s0_ub)
            T.copy(score1, s1_ub)
            T.copy(score2, s2_ub)
            T.copy(score3, s3_ub)

            # Block 0: sort → tmp → cache[0]
            T.tile.sort(sort_tmp, s0_ub, S2_BLOCK)
            T.copy(sort_tmp, sorted_cache[0 * sort_buf_size : 1 * sort_buf_size])

            # Block 1: sort → tmp → cache[1]
            T.tile.sort(sort_tmp, s1_ub, S2_BLOCK)
            T.copy(sort_tmp, sorted_cache[1 * sort_buf_size : 2 * sort_buf_size])

            # Block 2: sort → tmp → cache[2]
            T.tile.sort(sort_tmp, s2_ub, S2_BLOCK)
            T.copy(sort_tmp, sorted_cache[2 * sort_buf_size : 3 * sort_buf_size])

            # Block 3: sort → tmp → cache[3]
            T.tile.sort(sort_tmp, s3_ub, S2_BLOCK)
            T.copy(sort_tmp, sorted_cache[3 * sort_buf_size : 4 * sort_buf_size])

            # 4-way merge 用切片作为 src
            T.tile.merge_sort(
                merge_output,
                sorted_cache[0 * sort_buf_size : 1 * sort_buf_size],
                sorted_cache[1 * sort_buf_size : 2 * sort_buf_size],
                sorted_cache[2 * sort_buf_size : 3 * sort_buf_size],
                sorted_cache[3 * sort_buf_size : 4 * sort_buf_size],
            )

            T.copy(merge_output, output)

    return main


# ========== 对比: v6 标准路径（每次 sort 后立即 2-way merge） ==========
@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def sort_merge_2way_standard():
    """v6 标准路径: 每块 sort 后立即 2-way merge 到 running topk

    流程:
      1. sort(score0) → topk
      2. sort(score1) → tmp; merge_sort(topk, topk, tmp) → topk
      3. sort(score2) → tmp; merge_sort(topk, topk, tmp) → topk
      4. sort(score3) → tmp; merge_sort(topk, topk, tmp) → topk
    """
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

            # running topk buffer（每次 merge 后增大）
            topk_ub = T.alloc_ub([merge_out_size], "float32")
            # sort 临时 buffer
            sort_tmp = T.alloc_ub([sort_buf_size], "float32")
            # merge 临时输出
            merge_tmp = T.alloc_ub([merge_out_size], "float32")

            T.copy(score0, s0_ub)
            T.copy(score1, s1_ub)
            T.copy(score2, s2_ub)
            T.copy(score3, s3_ub)

            # Block 0: sort → topk (size = sort_buf_size)
            T.tile.sort(topk_ub, s0_ub, S2_BLOCK)

            # Block 1: sort → tmp; 2-way merge(topk, tmp) → topk (size = 2*sort_buf_size)
            T.tile.sort(sort_tmp, s1_ub, S2_BLOCK)
            T.tile.merge_sort(merge_tmp, topk_ub[0 : sort_buf_size], sort_tmp)
            T.copy(merge_tmp[0 : 2 * sort_buf_size], topk_ub[0 : 2 * sort_buf_size])

            # Block 2: sort → tmp; 2-way merge(topk[0:2*sz], tmp) → topk (size = 3*sort_buf_size)
            T.tile.sort(sort_tmp, s2_ub, S2_BLOCK)
            T.tile.merge_sort(merge_tmp, topk_ub[0 : 2 * sort_buf_size], sort_tmp)
            T.copy(merge_tmp[0 : 3 * sort_buf_size], topk_ub[0 : 3 * sort_buf_size])

            # Block 3: sort → tmp; 2-way merge(topk[0:3*sz], tmp) → topk (size = 4*sort_buf_size)
            T.tile.sort(sort_tmp, s3_ub, S2_BLOCK)
            T.tile.merge_sort(merge_tmp, topk_ub[0 : 3 * sort_buf_size], sort_tmp)
            T.copy(merge_tmp[0 : 4 * sort_buf_size], topk_ub[0 : 4 * sort_buf_size])

            T.copy(topk_ub[0 : 4 * sort_buf_size], output)

    return main


# ========== Golden 参考 ==========
def ref_sort_cache_4way(scores):
    """Python 参考: 4 个 score 分别 sort 后 4-way merge"""
    sorted_blocks = []
    for score in scores:
        # stable sort, descending
        indices = list(range(len(score)))
        indices.sort(key=lambda i: (-score[i], i))
        block = torch.zeros(len(score) * ELEMENT_SIZE, dtype=torch.float32)
        for j, idx in enumerate(indices):
            block[j * ELEMENT_SIZE + VALUE_POS] = score[idx]
            block[j * ELEMENT_SIZE + INDEX_POS] = float(idx)
        sorted_blocks.append(block)

    # 4-way merge
    sequences = []
    for block in sorted_blocks:
        pairs = []
        for j in range(len(score := scores[0])):
            v = block[j * ELEMENT_SIZE + VALUE_POS].item()
            i = block[j * ELEMENT_SIZE + INDEX_POS].item()
            pairs.append((v, i))
        sequences.append(pairs)

    neg_seqs = [[(-v, i) for v, i in seq] for seq in sequences]
    merged = list(heapq.merge(*neg_seqs))
    merged = [(-v, i) for v, i in merged]

    result = torch.zeros(len(merged) * ELEMENT_SIZE, dtype=torch.float32)
    for i, (v, idx) in enumerate(merged):
        result[i * ELEMENT_SIZE + VALUE_POS] = v
        result[i * ELEMENT_SIZE + INDEX_POS] = idx
    return result


def check_result(result, ref, name):
    result_cpu = result.cpu()
    n_pairs = len(result_cpu) // ELEMENT_SIZE
    values = [result_cpu[i * ELEMENT_SIZE + VALUE_POS].item() for i in range(n_pairs)]
    ref_values = [ref[i * ELEMENT_SIZE + VALUE_POS].item() for i in range(n_pairs)]

    is_sorted = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    values_match = values == ref_values

    print(f"\n[{name}]")
    print(f"  Is sorted (descending): {is_sorted}")
    print(f"  Values match golden: {values_match}")
    if not values_match:
        correct = sum(1 for v, r in zip(values, ref_values) if abs(v - r) < 1e-5)
        print(f"  Correct: {correct}/{len(ref_values)}")
        # 显示前 10 个差异
        diffs = [(i, v, r) for i, (v, r) in enumerate(zip(values, ref_values)) if abs(v - r) >= 1e-5]
        for i, v, r in diffs[:5]:
            print(f"    diff[{i}]: kernel={v}, golden={r}")
    return is_sorted and values_match


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("验证 2: sort 缓存优化替代方案（1D 平铺 buffer）")
    print("=" * 70)
    print(f"参数: S2_BLOCK={S2_BLOCK}, CACHE_SLOTS={CACHE_SLOTS}, TOPK={TOPK}")

    # 创建 4 个 score buffer
    scores = [torch.randn(S2_BLOCK, dtype=torch.float32) for _ in range(4)]
    scores_npu = [s.npu() for s in scores]
    ref = ref_sort_cache_4way(scores)

    # 测试 1: 4-way cache 方案
    print("\n--- 测试 1: sort 缓存 + 4-way merge（v7 方案） ---")
    try:
        kernel1 = sort_cache_4way_merge()
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result1 = kernel1(*scores_npu)
        torch.npu.synchronize()
        ok1 = check_result(result1, ref, "4-way cache")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok1 = False

    # 测试 2: v6 标准 2-way 方案
    print("\n--- 测试 2: v6 标准 2-way merge（对比基线） ---")
    try:
        kernel2 = sort_merge_2way_standard()
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result2 = kernel2(*scores_npu)
        torch.npu.synchronize()
        ok2 = check_result(result2, ref, "2-way standard")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok2 = False

    # 汇总
    print("\n" + "=" * 70)
    print("汇总:")
    print(f"  4-way cache 方案:   {'✅ PASS' if ok1 else '❌ FAIL'}")
    print(f"  2-way standard:     {'✅ PASS' if ok2 else '❌ FAIL'}")
    print("=" * 70)

    if ok1:
        print("\n结论: sort 缓存优化 1D 平铺方案可行")
        print("→ 可应用到 v7 lightning_indexer 的 sort 优化")
    else:
        print("\n结论: 4-way cache 方案有问题，需排查或回退到 2-way standard")


if __name__ == "__main__":
    main()
