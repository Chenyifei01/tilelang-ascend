"""验证 1: T.tile.merge_sort 4-way merge

v7 设计中 sort 缓存优化依赖 4-way merge_sort。
调研结论：4-way 通过位置参数调用 merge_sort(dst, s0, s1, s2, s3)，无 way 参数。

验证目标：
1. 4-way merge_sort 编译通过
2. 4-way merge_sort 精度正确（与 Python heapq.merge 对比）
3. 验证 BufferRegion 切片作为 src 是否可行（v7 sort 缓存优化的关键）

运行: python examples/lightning_indexer/api_tests/test_merge_sort_4way.py
"""
import tilelang as tl
import tilelang.language as T
import torch
import heapq

tl.cache.clear_cache()

ELEMENT_SIZE = 2  # value-index pair
VALUE_POS = 0
INDEX_POS = 1
N = 64  # elements per block

pass_configs = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ========== 测试 1: 基础 4-way merge（独立 buffer） ==========
@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def merge_sort_4way_basic():
    """4 个独立 buffer → 1 个输出"""
    @T.prim_func
    def main(
        block0: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block1: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block2: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block3: T.Tensor([N * ELEMENT_SIZE], "float32"),
        output: T.Tensor([N * 4 * ELEMENT_SIZE], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _):
            src0 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src1 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src2 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src3 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            merge_output = T.alloc_shared([N * 4 * ELEMENT_SIZE], "float32")

            T.copy(block0, src0)
            T.copy(block1, src1)
            T.copy(block2, src2)
            T.copy(block3, src3)

            # 4-way: 位置参数，无 way 参数
            T.tile.merge_sort(merge_output, src0, src1, src2, src3)

            T.copy(merge_output, output)

    return main


# ========== 测试 2: 1D 平铺 buffer + 切片作为 src ==========
# v7 sort 缓存优化的关键：sorted_cache_ub 是 1D 平铺 buffer，
# sort 结果通过 T.copy 搬运到对应偏移，merge_sort 用切片作为 src。
CACHE_SLOTS = 4
TOTAL_CACHE_SIZE = N * CACHE_SLOTS * ELEMENT_SIZE


@tl.jit(out_idx=[-1], pass_configs=pass_configs)
def merge_sort_4way_sliced_src():
    """1D 平铺 buffer 的切片作为 merge_sort 的 src

    模拟 v7 sort 缓存优化：
    - sorted_cache_ub 是 1D 平铺 buffer [CACHE_SLOTS * N * ELEMENT_SIZE]
    - 用切片 sorted_cache_ub[i*N*2:(i+1)*N*2] 作为 merge_sort 的 src
    """
    @T.prim_func
    def main(
        block0: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block1: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block2: T.Tensor([N * ELEMENT_SIZE], "float32"),
        block3: T.Tensor([N * ELEMENT_SIZE], "float32"),
        output: T.Tensor([N * 4 * ELEMENT_SIZE], "float32"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _):
            # 4 个输入 buffer（模拟 sort 结果）
            src0 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src1 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src2 = T.alloc_shared([N * ELEMENT_SIZE], "float32")
            src3 = T.alloc_shared([N * ELEMENT_SIZE], "float32")

            # 1D 平铺 cache buffer
            sorted_cache = T.alloc_shared([TOTAL_CACHE_SIZE], "float32")
            merge_output = T.alloc_shared([N * 4 * ELEMENT_SIZE], "float32")

            # 搬运输入到 cache 的对应偏移
            T.copy(block0, src0)
            T.copy(block1, src1)
            T.copy(block2, src2)
            T.copy(block3, src3)

            # 模拟 sort 后 T.copy 到 cache 偏移
            T.copy(src0, sorted_cache[0 * N * ELEMENT_SIZE : 1 * N * ELEMENT_SIZE])
            T.copy(src1, sorted_cache[1 * N * ELEMENT_SIZE : 2 * N * ELEMENT_SIZE])
            T.copy(src2, sorted_cache[2 * N * ELEMENT_SIZE : 3 * N * ELEMENT_SIZE])
            T.copy(src3, sorted_cache[3 * N * ELEMENT_SIZE : 4 * N * ELEMENT_SIZE])

            # 4-way merge 用切片作为 src
            T.tile.merge_sort(
                merge_output,
                sorted_cache[0 * N * ELEMENT_SIZE : 1 * N * ELEMENT_SIZE],
                sorted_cache[1 * N * ELEMENT_SIZE : 2 * N * ELEMENT_SIZE],
                sorted_cache[2 * N * ELEMENT_SIZE : 3 * N * ELEMENT_SIZE],
                sorted_cache[3 * N * ELEMENT_SIZE : 4 * N * ELEMENT_SIZE],
            )

            T.copy(merge_output, output)

    return main


# ========== Golden 参考 ==========
def create_sorted_block(n):
    """创建已排序的 (value, index) 交错对"""
    values = torch.randn(n, dtype=torch.float32)
    sorted_indices = torch.argsort(values, descending=True)
    sorted_values = values[sorted_indices]

    block = torch.zeros(n * ELEMENT_SIZE, dtype=torch.float32)
    for i in range(n):
        block[i * ELEMENT_SIZE + VALUE_POS] = sorted_values[i]
        block[i * ELEMENT_SIZE + INDEX_POS] = float(sorted_indices[i].item())
    return block


def ref_merge(blocks):
    """Python heapq.merge 参考"""
    sequences = []
    for block in blocks:
        pairs = []
        for j in range(N):
            v = block[j * ELEMENT_SIZE + VALUE_POS].item()
            i = block[j * ELEMENT_SIZE + INDEX_POS].item()
            pairs.append((v, i))
        sequences.append(pairs)

    neg_seqs = [[(-v, i) for v, i in seq] for seq in sequences]
    merged = list(heapq.merge(*neg_seqs))
    merged = [(-v, i) for v, i in merged]

    result = torch.zeros(N * len(blocks) * ELEMENT_SIZE, dtype=torch.float32)
    for i, (v, idx) in enumerate(merged):
        result[i * ELEMENT_SIZE + VALUE_POS] = v
        result[i * ELEMENT_SIZE + INDEX_POS] = idx
    return result


def check_result(result, ref, name):
    """检查结果"""
    result_cpu = result.cpu()
    values = [result_cpu[i * ELEMENT_SIZE + VALUE_POS].item() for i in range(len(result_cpu) // ELEMENT_SIZE)]
    ref_values = [ref[i * ELEMENT_SIZE + VALUE_POS].item() for i in range(len(ref) // ELEMENT_SIZE)]

    is_sorted = all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    values_match = values == ref_values

    print(f"\n[{name}]")
    print(f"  Is sorted (descending): {is_sorted}")
    print(f"  Values match golden: {values_match}")
    if not values_match:
        correct = sum(1 for v, r in zip(values, ref_values) if abs(v - r) < 1e-5)
        print(f"  Correct: {correct}/{len(ref_values)}")
    return is_sorted and values_match


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("验证 1: T.tile.merge_sort 4-way merge")
    print("=" * 70)

    # 创建 4 个已排序的 block
    blocks = [create_sorted_block(N) for _ in range(4)]
    blocks_npu = [b.npu() for b in blocks]
    ref = ref_merge(blocks)

    # 测试 1: 基础 4-way
    print("\n--- 测试 1: 基础 4-way merge（独立 buffer） ---")
    try:
        kernel1 = merge_sort_4way_basic()
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result1 = kernel1(*blocks_npu)
        torch.npu.synchronize()
        ok1 = check_result(result1, ref, "基础 4-way")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        ok1 = False

    # 测试 2: 1D 平铺 + 切片 src
    print("\n--- 测试 2: 1D 平铺 buffer + 切片作为 src ---")
    try:
        kernel2 = merge_sort_4way_sliced_src()
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result2 = kernel2(*blocks_npu)
        torch.npu.synchronize()
        ok2 = check_result(result2, ref, "切片 src 4-way")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        ok2 = False

    # 汇总
    print("\n" + "=" * 70)
    print("汇总:")
    print(f"  基础 4-way merge: {'✅ PASS' if ok1 else '❌ FAIL'}")
    print(f"  切片 src 4-way:   {'✅ PASS' if ok2 else '❌ FAIL'}")
    print("=" * 70)

    if ok1 and ok2:
        print("\n结论: 4-way merge_sort 可行，切片作为 src 可行")
        print("→ v7 sort 缓存优化的 1D 平铺方案可应用")
    else:
        print("\n结论: 部分验证失败，需排查")


if __name__ == "__main__":
    main()
