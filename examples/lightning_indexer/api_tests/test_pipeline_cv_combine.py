"""验证 4: num_stages=3 + threads=2 + AUTO_CV_COMBINE 组合

v7 设计: Cube/Vector 流水深度提升 + CV 融合
- num_stages=3（已验证示例，buffer 须在外部分配）
- threads=2（AIC:AIV=1:2 双核）
- AUTO_CV_COMBINE（CombineCV Pass 分裂 cube_code/vec_code）

验证目标:
1. 三者组合编译通过
2. 精度正确
3. 确认 buffer 在 Pipelined body 外部分配的正确写法

运行: python examples/lightning_indexer/api_tests/test_pipeline_cv_combine.py
"""
import tilelang as tl
import tilelang.language as T
import torch

tl.cache.clear_cache()

# GEMM 参数
M = 256
N = 256
K = 512
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64

pass_configs = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ========== 测试 1: num_stages=3 + threads=2 + CV_COMBINE ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_pipeline_cv(M: int, N: int, K_dim: int):
    """num_stages=3 + threads=2 + CV_COMBINE 组合

    关键: buffer 在 Pipelined body 外部分配（避免 ring-buffer 放大 UB）
    """
    m_num = M // BLOCK_M
    n_num = N // BLOCK_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K_dim), "float16"),
        B: T.Tensor((K_dim, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        # threads=2: AIC+AIV 双核
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num

            # buffer 在 Pipelined body 外部分配！
            A_L1 = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_L1 = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            C_L0 = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            c_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")

            loop_k = T.ceildiv(K_dim, BLOCK_K)

            # num_stages=3
            for k in T.Pipelined(loop_k, num_stages=3):
                T.copy(A[bx * BLOCK_M, k * BLOCK_K], A_L1)
                T.copy(B[k * BLOCK_K, by * BLOCK_N], B_L1)
                if k == 0:
                    T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                else:
                    T.gemm_v0(A_L1, B_L1, C_L0)

            T.copy(C_L0, c_ub)
            T.copy(c_ub, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, by * BLOCK_N : (by + 1) * BLOCK_N])

    return main


# ========== 测试 2: num_stages=4（验证更高流水深度） ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_pipeline_4stage(M: int, N: int, K_dim: int):
    """num_stages=4 验证（源码无上界，但需实测 UB 占用）"""
    m_num = M // BLOCK_M
    n_num = N // BLOCK_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K_dim), "float16"),
        B: T.Tensor((K_dim, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_L1 = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            C_L0 = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            c_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")

            loop_k = T.ceildiv(K_dim, BLOCK_K)

            for k in T.Pipelined(loop_k, num_stages=4):
                T.copy(A[bx * BLOCK_M, k * BLOCK_K], A_L1)
                T.copy(B[k * BLOCK_K, by * BLOCK_N], B_L1)
                if k == 0:
                    T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                else:
                    T.gemm_v0(A_L1, B_L1, C_L0)

            T.copy(C_L0, c_ub)
            T.copy(c_ub, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, by * BLOCK_N : (by + 1) * BLOCK_N])

    return main


# ========== 测试 3: num_stages=2 基线（对比） ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_pipeline_2stage_baseline(M: int, N: int, K_dim: int):
    """num_stages=2 基线"""
    m_num = M // BLOCK_M
    n_num = N // BLOCK_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K_dim), "float16"),
        B: T.Tensor((K_dim, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_L1 = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            C_L0 = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            c_ub = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")

            loop_k = T.ceildiv(K_dim, BLOCK_K)

            for k in T.Pipelined(loop_k, num_stages=2):
                T.copy(A[bx * BLOCK_M, k * BLOCK_K], A_L1)
                T.copy(B[k * BLOCK_K, by * BLOCK_N], B_L1)
                if k == 0:
                    T.gemm_v0(A_L1, B_L1, C_L0, init=True)
                else:
                    T.gemm_v0(A_L1, B_L1, C_L0)

            T.copy(C_L0, c_ub)
            T.copy(c_ub, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, by * BLOCK_N : (by + 1) * BLOCK_N])

    return main


def run_test(name, kernel_func, M, N, K_dim):
    """运行单个测试"""
    print(f"\n--- {name} ---")
    try:
        kernel = kernel_func(M, N, K_dim)
        torch.npu.synchronize()
        print("  编译: ✅ 通过")

        A = torch.randn(M, K_dim, dtype=torch.float16).npu()
        B = torch.randn(K_dim, N, dtype=torch.float16).npu()
        ref = (A.float() @ B.float()).half()

        result = kernel(A, B)
        torch.npu.synchronize()

        diff = (result.float() - ref.float()).abs().max().item()
        ok = diff < 2.0  # fp16 GEMM 容差
        print(f"  精度: max_diff={diff:.4f} ({'✅ PASS' if ok else '❌ FAIL'})")
        return ok, diff
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        return False, float('inf')


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("验证 4: num_stages + threads=2 + AUTO_CV_COMBINE 组合")
    print("=" * 70)
    print(f"参数: M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")

    results = []

    # 测试 1: num_stages=3
    ok1, diff1 = run_test(
        "num_stages=3 + threads=2 + CV_COMBINE",
        gemm_pipeline_cv, M, N, K
    )
    results.append(("num_stages=3", ok1, diff1))

    # 测试 2: num_stages=4
    ok2, diff2 = run_test(
        "num_stages=4 + threads=2 + CV_COMBINE",
        gemm_pipeline_4stage, M, N, K
    )
    results.append(("num_stages=4", ok2, diff2))

    # 测试 3: num_stages=2 基线
    ok3, diff3 = run_test(
        "num_stages=2 基线 + threads=2 + CV_COMBINE",
        gemm_pipeline_2stage_baseline, M, N, K
    )
    results.append(("num_stages=2 基线", ok3, diff3))

    # 汇总
    print("\n" + "=" * 70)
    print("汇总:")
    for name, ok, diff in results:
        print(f"  {name}: {'✅ PASS' if ok else '❌ FAIL'} (diff={diff:.4f})")
    print("=" * 70)

    if ok1 and ok2:
        print("\n结论: num_stages=3-4 + threads=2 + CV_COMBINE 组合可行")
        print("→ 可应用到 v7 lightning_indexer 的 Cube/Vector 流水深度提升")
    elif ok1:
        print("\n结论: num_stages=3 可行，num_stages=4 需进一步排查 UB 占用")
        print("→ 建议先用 num_stages=3")
    else:
        print("\n结论: 组合验证失败，需排查")


if __name__ == "__main__":
    main()
