"""验证 5: T.pipe_barrier("v") 替代 T.barrier_all

用 threads=2 + GEMM 结构（参考 matmul_add_developer.py），
在 Vector 侧加 pipe_barrier("v") 验证同步效果。
"""
import tilelang as tl
import tilelang.language as T
import torch

tl.cache.clear_cache()

M = 256
N = 256
K = 128
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128

pass_configs = {
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}


# ========== 测试 1: GEMM + add + pipe_barrier("v") ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_add_pipe_barrier():
    """GEMM (Cube) → add (Vector) + pipe_barrier("v")

    参考 matmul_add_developer.py 结构：
      1. GEMM (Cube) → C_L0
      2. copy C_L0 → c_ub
      3. copy D → d_ub
      4. add c_ub += d_ub (Vector)
      5. pipe_barrier("v") — 确保 add 完成
      6. copy c_ub → C
    """
    m_num = M // BLOCK_M
    n_num = N // BLOCK_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
        D: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_L1 = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            C_L0 = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            c_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")
            d_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")

            T.copy(A[bx * BLOCK_M, 0], A_L1)
            T.copy(B[0, by * BLOCK_N], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=True)

            T.copy(C_L0, c_ub)
            T.copy(D[bx * BLOCK_M, by * BLOCK_N], d_ub)

            # Vector add
            T.tile.add(c_ub, c_ub, d_ub)

            # pipe_barrier("v"): 确保 add 完成
            T.pipe_barrier("v")

            T.copy(c_ub, C[bx * BLOCK_M, by * BLOCK_N])

    return main


# ========== 测试 2: GEMM + add + barrier_all ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_add_barrier_all():
    """用 barrier_all 替代 pipe_barrier("v")"""
    m_num = M // BLOCK_M
    n_num = N // BLOCK_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
        D: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_L1 = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            C_L0 = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            c_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")
            d_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")

            T.copy(A[bx * BLOCK_M, 0], A_L1)
            T.copy(B[0, by * BLOCK_N], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=True)

            T.copy(C_L0, c_ub)
            T.copy(D[bx * BLOCK_M, by * BLOCK_N], d_ub)

            T.tile.add(c_ub, c_ub, d_ub)

            T.barrier_all()

            T.copy(c_ub, C[bx * BLOCK_M, by * BLOCK_N])

    return main


# ========== 测试 3: 无显式同步 ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_add_no_sync():
    """无显式同步，依赖 AUTO_SYNC"""
    m_num = M // BLOCK_M
    n_num = N // BLOCK_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
        D: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_L1 = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            C_L0 = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            c_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")
            d_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")

            T.copy(A[bx * BLOCK_M, 0], A_L1)
            T.copy(B[0, by * BLOCK_N], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, init=True)

            T.copy(C_L0, c_ub)
            T.copy(D[bx * BLOCK_M, by * BLOCK_N], d_ub)

            T.tile.add(c_ub, c_ub, d_ub)

            T.copy(c_ub, C[bx * BLOCK_M, by * BLOCK_N])

    return main


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("验证 5: T.pipe_barrier('v') 替代 T.barrier_all")
    print("=" * 70)
    print(f"参数: M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print(f"计算: C = A @ B + D")

    A = torch.randn(M, K, dtype=torch.float16).npu()
    B = torch.randn(K, N, dtype=torch.float16).npu()
    D = torch.randn(M, N, dtype=torch.float16).npu()
    ref = (A.float() @ B.float() + D.float()).half()

    results = []

    # 测试 1: pipe_barrier("v")
    print("\n--- 测试 1: GEMM + add + pipe_barrier('v') ---")
    try:
        kernel1 = gemm_add_pipe_barrier()
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result1 = kernel1(A, B, D)
        torch.npu.synchronize()
        diff1 = (result1.float() - ref.float()).abs().max().item()
        ok1 = diff1 < 2.0
        print(f"  精度: max_diff={diff1:.4f} ({'✅ PASS' if ok1 else '❌ FAIL'})")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok1 = False
        diff1 = float('inf')
    results.append(("pipe_barrier('v')", ok1, diff1))

    # 测试 2: barrier_all
    print("\n--- 测试 2: GEMM + add + barrier_all ---")
    try:
        kernel2 = gemm_add_barrier_all()
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result2 = kernel2(A, B, D)
        torch.npu.synchronize()
        diff2 = (result2.float() - ref.float()).abs().max().item()
        ok2 = diff2 < 2.0
        print(f"  精度: max_diff={diff2:.4f} ({'✅ PASS' if ok2 else '❌ FAIL'})")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok2 = False
        diff2 = float('inf')
    results.append(("barrier_all", ok2, diff2))

    # 测试 3: 无显式同步
    print("\n--- 测试 3: 无显式同步（AUTO_SYNC） ---")
    try:
        kernel3 = gemm_add_no_sync()
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result3 = kernel3(A, B, D)
        torch.npu.synchronize()
        diff3 = (result3.float() - ref.float()).abs().max().item()
        ok3 = diff3 < 2.0
        print(f"  精度: max_diff={diff3:.4f} ({'✅ PASS' if ok3 else '❌ FAIL'})")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok3 = False
        diff3 = float('inf')
    results.append(("no explicit sync", ok3, diff3))

    # 汇总
    print("\n" + "=" * 70)
    print("汇总:")
    for name, ok, diff in results:
        print(f"  {name}: {'✅ PASS' if ok else '❌ FAIL'} (diff={diff:.4f})")
    print("=" * 70)

    if ok1:
        print("\n结论: pipe_barrier('v') 可行，可替代 barrier_all 用于 Vector 管线同步")
        print("→ 可应用到 v7 lightning_indexer 的同步优化")
    else:
        print("\n结论: pipe_barrier('v') 验证失败，需排查")


if __name__ == "__main__":
    main()
