"""验证 3: 3 层 tiling + L1 行切片传 gemm_v0

v7 设计核心: Cube GEMM 3 层 tiling 对齐 AscendC
- L1 层: M_L1=256, S2_L1=256
- L0 层: M_L0=128, N_L0=128
- 4 层嵌套循环: s2_l1 → m_l1 → s2_l0 → m_l0

调研结论:
- T.gemm_v0 支持 BufferRegion（行切片，内存连续）
- 4 层 T.serial 嵌套编译通过（lightning_indexer 已有先例）
- 未验证组合: M/N 在 L0 层切分 + 行切片

验证目标:
1. 3 层 tiling 4 层嵌套循环编译通过
2. L1 行切片传 gemm_v0 精度正确
3. 与单层 tiling 对比精度一致

运行: python examples/lightning_indexer/api_tests/test_3layer_tiling.py
"""
import tilelang as tl
import tilelang.language as T
import torch

tl.cache.clear_cache()

# Tiling 参数（对齐 AscendC arch22）
M_L1 = 256      # L1 层 M tile
S2_L1 = 256      # L1 层 S2 tile
M_L0 = 128       # L0 层 M tile
N_L0 = 128       # L0 层 N tile
K = 128          # K 维（D=128 固定）

pass_configs = {
    tl.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tl.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tl.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


# ========== 测试 1: 3 层 tiling（Developer 模式，行切片） ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_3layer_tiling_developer(M: int, N: int, K_dim: int = K):
    """Developer 模式 3 层 tiling

    结构:
      for s2_l1 in range(N // S2_L1):       # L1 层 S2 循环
        for m_l1 in range(M // M_L1):        # L1 层 M 循环
          copy A[m_l1*M_L1, 0] → A_L1        # L1 buffer (M_L1, K)
          copy B[s2_l1*S2_L1, 0] → B_L1      # L1 buffer (S2_L1, K)
          for s2_l0 in range(S2_L1 // N_L0): # L0 层 S2 循环
            for m_l0 in range(M_L1 // M_L0):  # L0 层 M 循环
              gemm_v0(A_L1[m_l0*M_L0:(m_l0+1)*M_L0, :],
                      B_L1[s2_l0*N_L0:(s2_l0+1)*N_L0, :],
                      C_L0, transpose_B=True, init=True)
              copy C_L0 → C[m_l1*M_L1+m_l0*M_L0, s2_l1*S2_L1+s2_l0*N_L0]
    """
    m_l1_num = M // M_L1
    s2_l1_num = N // S2_L1

    @T.prim_func
    def main(
        A: T.Tensor((M, K_dim), "float16"),  # Q: (M, K)
        B: T.Tensor((N, K_dim), "float16"),  # K: (N, K), transpose_B=True
        C: T.Tensor((M, N), "float16"),      # 输出
    ):
        with T.Kernel(m_l1_num * s2_l1_num, is_npu=True) as (cid, _):
            bx = cid // s2_l1_num
            by = cid % s2_l1_num

            # L1 buffers（Developer 模式用 alloc_shared）
            A_L1 = T.alloc_shared((M_L1, K_dim), "float16")
            B_L1 = T.alloc_shared((S2_L1, K_dim), "float16")
            # L0C（Developer 模式用 alloc_fragment）
            C_L0 = T.alloc_fragment((M_L0, N_L0), "float32")

            # 载入 L1
            T.copy(A[bx * M_L1, 0], A_L1)
            T.copy(B[by * S2_L1, 0], B_L1)

            # 4 层嵌套（这里外层 2 层用 kernel 并行，内层 2 层用 serial）
            for s2_l0 in T.serial(S2_L1 // N_L0):
                for m_l0 in T.serial(M_L1 // M_L0):
                    # 行切片传给 gemm_v0
                    T.gemm_v0(
                        A_L1[m_l0 * M_L0 : (m_l0 + 1) * M_L0, :],
                        B_L1[s2_l0 * N_L0 : (s2_l0 + 1) * N_L0, :],
                        C_L0,
                        transpose_B=True,
                        init=True,
                    )
                    # 写回 GM
                    T.copy(
                        C_L0,
                        C[
                            bx * M_L1 + m_l0 * M_L0 : bx * M_L1 + (m_l0 + 1) * M_L0,
                            by * S2_L1 + s2_l0 * N_L0 : by * S2_L1 + (s2_l0 + 1) * N_L0,
                        ],
                    )

    return main


# ========== 测试 2: 完整 4 层嵌套（M > M_L1 且 N > S2_L1） ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_4layer_nesting_developer(M: int, N: int, K_dim: int = K):
    """完整 4 层 serial 嵌套（kernel 只用 1 个 block）

    用于验证 4 层嵌套编译是否通过
    """
    @T.prim_func
    def main(
        A: T.Tensor((M, K_dim), "float16"),
        B: T.Tensor((N, K_dim), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, is_npu=True) as (_cid, _):
            A_L1 = T.alloc_shared((M_L1, K_dim), "float16")
            B_L1 = T.alloc_shared((S2_L1, K_dim), "float16")
            C_L0 = T.alloc_fragment((M_L0, N_L0), "float32")

            # 4 层 serial 嵌套
            for s2_l1 in T.serial(N // S2_L1):
                for m_l1 in T.serial(M // M_L1):
                    T.copy(A[m_l1 * M_L1, 0], A_L1)
                    T.copy(B[s2_l1 * S2_L1, 0], B_L1)
                    for s2_l0 in T.serial(S2_L1 // N_L0):
                        for m_l0 in T.serial(M_L1 // M_L0):
                            T.gemm_v0(
                                A_L1[m_l0 * M_L0 : (m_l0 + 1) * M_L0, :],
                                B_L1[s2_l0 * N_L0 : (s2_l0 + 1) * N_L0, :],
                                C_L0,
                                transpose_B=True,
                                init=True,
                            )
                            T.copy(
                                C_L0,
                                C[
                                    m_l1 * M_L1 + m_l0 * M_L0 : m_l1 * M_L1 + (m_l0 + 1) * M_L0,
                                    s2_l1 * S2_L1 + s2_l0 * N_L0 : s2_l1 * S2_L1 + (s2_l0 + 1) * N_L0,
                                ],
                            )

    return main


# ========== 测试 3: 独立 buffer（无切片，验证切片是否是问题根源） ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_independent_buffers(M: int, N: int, K_dim: int = K):
    """用 4 个独立 L1 buffer 代替 1 个大 buffer + 切片

    如果这个测试精度正确，说明问题是 transpose_B=True + L1 行切片的组合
    """
    m_num = M // M_L1
    n_num = N // S2_L1

    @T.prim_func
    def main(
        A: T.Tensor((M, K_dim), "float16"),
        B: T.Tensor((N, K_dim), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            # 4 个独立 L1 buffer（每个 128×128）
            A_L1_0 = T.alloc_shared((M_L0, K_dim), "float16")
            A_L1_1 = T.alloc_shared((M_L0, K_dim), "float16")
            B_L1_0 = T.alloc_shared((N_L0, K_dim), "float16")
            B_L1_1 = T.alloc_shared((N_L0, K_dim), "float16")
            C_L0 = T.alloc_fragment((M_L0, N_L0), "float32")

            # 载入 4 个子 tile
            T.copy(A[bx * M_L1, 0], A_L1_0)
            T.copy(A[bx * M_L1 + M_L0, 0], A_L1_1)
            T.copy(B[by * S2_L1, 0], B_L1_0)
            T.copy(B[by * S2_L1 + N_L0, 0], B_L1_1)

            # 4 个子 GEMM（无切片，每个用独立 buffer）
            T.gemm_v0(A_L1_0, B_L1_0, C_L0, transpose_B=True, init=True)
            T.copy(C_L0, C[bx * M_L1 : bx * M_L1 + M_L0, by * S2_L1 : by * S2_L1 + N_L0])

            T.gemm_v0(A_L1_1, B_L1_0, C_L0, transpose_B=True, init=True)
            T.copy(C_L0, C[bx * M_L1 + M_L0 : bx * M_L1 + 2 * M_L0, by * S2_L1 : by * S2_L1 + N_L0])

            T.gemm_v0(A_L1_0, B_L1_1, C_L0, transpose_B=True, init=True)
            T.copy(C_L0, C[bx * M_L1 : bx * M_L1 + M_L0, by * S2_L1 + N_L0 : by * S2_L1 + 2 * N_L0])

            T.gemm_v0(A_L1_1, B_L1_1, C_L0, transpose_B=True, init=True)
            T.copy(C_L0, C[bx * M_L1 + M_L0 : bx * M_L1 + 2 * M_L0, by * S2_L1 + N_L0 : by * S2_L1 + 2 * N_L0])

    return main


# ========== 测试 4: 单层 tiling（对比基线） ==========
@tl.jit(out_idx=[2], pass_configs=pass_configs)
def gemm_single_layer_baseline(M: int, N: int, K_dim: int = K):
    """单层 tiling（M_L0=N_L0=128），作为精度对比基线"""
    m_num = M // M_L0
    n_num = N // N_L0

    @T.prim_func
    def main(
        A: T.Tensor((M, K_dim), "float16"),
        B: T.Tensor((N, K_dim), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((M_L0, K_dim), "float16")
            B_L1 = T.alloc_shared((N_L0, K_dim), "float16")
            C_L0 = T.alloc_fragment((M_L0, N_L0), "float32")

            T.copy(A[bx * M_L0, 0], A_L1)
            T.copy(B[by * N_L0, 0], B_L1)
            T.gemm_v0(A_L1, B_L1, C_L0, transpose_B=True, init=True)
            T.copy(C_L0, C[bx * M_L0 : (bx + 1) * M_L0, by * N_L0 : (by + 1) * N_L0])

    return main


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("验证 3: 3 层 tiling + L1 行切片传 gemm_v0")
    print("=" * 70)
    print(f"参数: M_L1={M_L1}, S2_L1={S2_L1}, M_L0={M_L0}, N_L0={N_L0}, K={K}")

    # 测试规模（M=N=256, K=128）
    M = N = 256

    A = torch.randn(M, K, dtype=torch.float16).npu()
    B = torch.randn(N, K, dtype=torch.float16).npu()
    ref = (A.float() @ B.float().T).half()

    # 测试 1: 3 层 tiling（2 层 serial 嵌套）
    print(f"\n--- 测试 1: 3 层 tiling（M=N={M}, K={K}, 2 层 serial 嵌套） ---")
    try:
        kernel1 = gemm_3layer_tiling_developer(M, N, K)
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result1 = kernel1(A, B)
        torch.npu.synchronize()
        diff1 = (result1.float() - ref.float()).abs().max().item()
        ok1 = diff1 < 1.0  # fp16 GEMM 容差
        print(f"  精度: max_diff={diff1:.4f} ({'✅ PASS' if ok1 else '❌ FAIL'})")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok1 = False
        diff1 = float('inf')

    # 测试 2: 完整 4 层 serial 嵌套
    print(f"\n--- 测试 2: 完整 4 层 serial 嵌套（M=N={M}, K={K}） ---")
    try:
        kernel2 = gemm_4layer_nesting_developer(M, N, K)
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result2 = kernel2(A, B)
        torch.npu.synchronize()
        diff2 = (result2.float() - ref.float()).abs().max().item()
        ok2 = diff2 < 1.0
        print(f"  精度: max_diff={diff2:.4f} ({'✅ PASS' if ok2 else '❌ FAIL'})")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok2 = False
        diff2 = float('inf')

    # 测试 3: 独立 buffer（无切片）
    print(f"\n--- 测试 3: 独立 buffer（无切片，验证切片是否是问题根源） ---")
    try:
        kernel3 = gemm_independent_buffers(M, N, K)
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result3 = kernel3(A, B)
        torch.npu.synchronize()
        diff3 = (result3.float() - ref.float()).abs().max().item()
        ok3 = diff3 < 1.0
        print(f"  精度: max_diff={diff3:.4f} ({'✅ PASS' if ok3 else '❌ FAIL'})")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok3 = False
        diff3 = float('inf')

    # 测试 4: 单层 tiling 基线
    print(f"\n--- 测试 4: 单层 tiling 基线（M=N={M}, K={K}） ---")
    try:
        kernel4 = gemm_single_layer_baseline(M, N, K)
        torch.npu.synchronize()
        print("  编译: ✅ 通过")
        result4 = kernel4(A, B)
        torch.npu.synchronize()
        diff4 = (result4.float() - ref.float()).abs().max().item()
        ok4 = diff4 < 1.0
        print(f"  精度: max_diff={diff4:.4f} ({'✅ PASS' if ok4 else '❌ FAIL'})")
    except Exception as e:
        print(f"  编译/运行失败: ❌ {e}")
        import traceback
        traceback.print_exc()
        ok4 = False
        diff4 = float('inf')

    # 汇总
    print("\n" + "=" * 70)
    print("汇总:")
    print(f"  3 层 tiling (L1 切片):        {'✅ PASS' if ok1 else '❌ FAIL'} (diff={diff1:.4f})")
    print(f"  4 层 serial 嵌套 (L1 切片):   {'✅ PASS' if ok2 else '❌ FAIL'} (diff={diff2:.4f})")
    print(f"  独立 buffer (无切片):          {'✅ PASS' if ok3 else '❌ FAIL'} (diff={diff3:.4f})")
    print(f"  单层 tiling 基线:              {'✅ PASS' if ok4 else '❌ FAIL'} (diff={diff4:.4f})")
    print("=" * 70)

    if ok3 and not ok1:
        print("\n结论: 确认是 transpose_B=True + L1 行切片的 bug")
        print("→ v7 设计需改用独立 buffer 或 Expert 模式（T.mma + T.copy(transpose=True)）")
    elif ok1 and ok3:
        print("\n结论: 3 层 tiling 可行（切片和独立 buffer 都正确）")
        print("→ 可应用到 v7 lightning_indexer 的 Cube GEMM tiling")
    else:
        print("\n结论: 需进一步排查")


if __name__ == "__main__":
    main()
