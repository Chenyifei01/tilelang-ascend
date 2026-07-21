# lightning_indexer 算子设计文档

> **设计版本 v7.4（基于 v7.3 + Phase 1-3 验证结果: sort 前 -inf 填充 + 边界 Case tiling 修复 + 单 Kernel 架构确认 + 测试代码 torch.sort bug 修正）**
>
> **v7.4 关键修正（Phase 1-3 验证后, 见 §12.12）**：
> - **Phase 1 (精度根因)**: sort 前未将无效 K 元素 score 设为 -inf → PA_BSND 输出 -1 无效 index. **修复**: sort 前对 `cu_s2_len:S2_VEC_BLOCK` 区间填 -inf (score) + -1 (index), 对齐 AscendC `service_vector.h:357-368` → §8.8
> - **Phase 1 (测试代码)**: 测试中 `torch.sort(cpu_result/npu_result)` 破坏 check_result 期望的 score 降序顺序, 导致 value_bm 取错值 (1/8→7/8 PASS). **修复**: 去掉 torch.sort → §9
> - **Phase 2 (架构确认)**: `lightning_indexer.py` 已是单 kernel + `T.Scope("C")/"V")` + 手动 `set_flag/wait_flag` (参考 `sparse_flash_attn_pa_no_cv_pipeline.py`). v7.3 "双 Kernel" 描述修正为"单 Kernel + CV 分离" → §2.1, §4, §11
> - **Phase 3 (边界 Case)**: block_size<64 超时 (BLOCKS_PER_TILE=16 过多) → 减小 BLOCK_N; Over2K (sparse_count>2048) UB=586KB 超限 → 缩小 S1_BLOCK + virTopK=sparse_count; TOP_K>S2 → -inf 填充 → §5.2.1, §10.2 Q27
>
> **v7.4 保留 v7.3 的全部优化**（4×L0C + make_zn/make_nz_layout + DMA 重排 + MTE2∥V overlap + brcb+row_expand_mul + 方案 C VID_S1=1 + SparseTopK 截断），仅在此基础上叠加 Phase 1-3 验证后的精度/边界修复。
>
> ---
>
> **设计版本 v7.3（基于 v7.2 + 5 个参考算子设计模式整合: 4×L0C + make_zn/make_nz_layout + DMA 重排 + MTE2∥V overlap + brcb+row_expand_mul）**
>
> **v7.3 关键更新（参考算子模式整合, 见 §12.11）**：
> - **模式 1 (P0)**: 4×L0C 消除 MMA/copy 争用 (HISA) → §4.6, §8.5
> - **模式 2 (P0)**: make_zn_layout / make_nz_layout (fa_opt) → §8.5
> - **模式 3 (P0)**: DMA 重排 K[0]+Q 优先 (HISA Wave 0) → §8.5
> - **模式 5 (P1)**: cross_interval=2 减少跨核同步 (fa_opt, ✅ 已采纳) → §6.3, §8.5
> - **模式 6 (P1)**: MTE2∥V overlap (HISA) → §8.8
> - **模式 7 (P1)**: brcb_experiment + row_expand_mul_experiment 三步模式 (xattention) → §8.5
> - **模式 4/8/9/10/11**: 已整合或保留 v7.2 策略 → §8.8, §12.11
> - **模式 12 (P2)**: 3-slot PRE_LAUNCH 流水 (Stage 3 候选) → §12.11
>
> **v7.3 保留 v7.2 的全部修正**（L0C→GM→UB 路径 + 方案 C VID_S1=1 + SparseTopK 截断 + sort 稳定性澄清），仅在此基础上叠加参考算子优化模式。
>
> ---
>
> **设计版本 v7.2（基于 v7.1 + 5 个缺口验证结果修正: L0C→GM→UB 路径 + 方案 C VID_S1=1 + SparseTopK 截断 + sort 稳定性澄清）**
>
> **v7.2 关键修正（缺口验证后, 见 §12.10）**：
> - **缺口 1**: L0C 不能直接到 UB, 必须 L0C→GM→UB; T.Scope("V") 用 `as (cid, vid)` → §8.5
> - **缺口 2**: sort 缓存优化 UB 超限 (216.4KB), 方案 C (VID_S1=1) 降到 184.4KB ✓ → §5.3, §8.8
> - **缺口 3**: sort 稳定性不是问题 (check_result 已处理), 真正问题是 score 计算精度 → §9.3, §10.2 Q4
> - **缺口 6**: AscendC 6 个遗漏点, 最关键是 SparseTopK 显式截断 → §8.8, §12.6.6
>
> **v7.2 保留 v7.1 的全部 API 验证修正**（独立 L1 buffer + 1D 平铺 sort cache + 位置参数 merge_sort + num_stages=3-4 + pipe_barrier("v")），仅基于缺口验证结果补充内存路径约束、方案 C 内存优化、SparseTopK 截断和 sort 稳定性澄清。
>
> ---
>
> **设计版本 v7（基于 v6 + msprof op 实测性能数据 + AscendC arch22 深度对比分析的性能优化更新）**
>
> **v7 关键优化（基于 msprof op 实测数据驱动的 3 项 P0 性能优化 + 2 项 P1 工程 + 1 项 P2 数据范围特化）**：
>
> | 优化 | 优先级 | v6 现状（msprof 实测） | v7 优化内容 | 影响章节 |
> |------|--------|----------------------|------------|---------|
> | #1: Cube GEMM tiling 对齐 AscendC | 🔴 P0 | M_L1=S1_BLOCK×G=512（G=64时），Cube 时间 98.83us vs AscendC 60.14us（慢 64%） | M_L1=min(S1_BLOCK×G, 256) 对齐 AscendC M_BASIC_BLOCK=256；S2_L1=256 对齐 S2_BASIC_BLOCK=256；3 层 tiling (L1→L0→MMA) | §5.2, §8.5, §12.6 |
> | #2: sort 缓存优化（S1≤4 场景） | 🔴 P0 | 每次 S2 块都 sort+merge_sort，Vec0 利用率 48.4% vs AscendC 63.1% | 缓存 4 个 S2 块 sort 结果，满 4 块后 4-way MrgBasicBlock 精排，减少 75% merge_sort 调用 | §8.5, §8.8（新增）, §10.2 Q26 |
> | #3: Vector 同步优化 | 🟡 P1 | barrier_all + set_flag/wait_flag 粗粒度 | 减少 T.barrier_all 使用，Vector 侧 groupInner 循环内用 pipe_barrier("v")，优化 set_flag/wait_flag 粒度 | §7, §8.5 |
> | #4: 数据范围特化（G=64 优先） | 🟡 P1 | 通用路径 | N1=64 时 G=64，groupInner=16，VECTOR_BASEG=8，优先优化 G=64 场景（最常见） | §5.2, §8.5 |
> | #5: TND_PA_BSND block_size=16 修复 | 🔴 P0 | 超时（>120s） | 排查 block_size=16 场景的 tiling/buffer 问题，确保功能补齐 | §10.2 Q27, §12.6 |
> | #6: num_stages 提升 | 🟢 P2 | num_stages=2 保守 | v6 UB=34KB 余量充足，实测后尝试 num_stages=3-4 提升 Cube/Vector 流水深度 | §6, §12.6 |
>
> **v7 msprof op 实测数据（驱动本次优化的核心依据）**：
>
> | 用例 | 参数 | AscendC(us) | TileLang v6(us) | 比值 | 80%目标(us) | 达标 |
> |------|------|------------|-----------------|------|------------|------|
> | BSND_BSND | B=16 S1=5 S2=3072 N1=64 G=64 mode=3 | 74.26 | 111.64 | 1.50x | ≤92.83 | ❌ |
> | BSND_PA_BSND | B=2 S1=1 S2=2048 N1=8 G=8 mode=3 | 20.20 | **18.78** | 0.93x | ≤25.25 | ✅ |
> | TND_TND | B=8 S1=5 S2=3072 N1=24 G=24 mode=0 | 40.54 | 55.10 | 1.36x | ≤50.67 | ❌ |
> | TND_PA_BSND | B=20 S1=3 S2=512 N1=64 G=64 mode=0 bs=16 | 30.44 | 超时 | — | ≤38.05 | ⏳ |
>
> **v7 瓶颈分析（来自 COMPARISON.md + msprof ArithmeticUtilization 数据）**：
>
> | 用例 | 瓶颈 | AscendC Cube(us)/Vec0(us) | TileLang Cube(us)/Vec0(us) | 差距来源 |
> |------|------|--------------------------|---------------------------|---------|
> | BSND_BSND | Cube GEMM tiling | 60.14 / 69.26 | 98.83 / 102.49 | Cube 慢 64%（tiling 差异），Vec0 慢 48%（sort 缓存缺失） |
> | TND_TND | Cube GEMM tiling | 26.56 / 33.74 | 42.80 / 47.04 | Cube 慢 61%，Vec0 慢 39% |
> | BSND_PA_BSND | 已达标 | 9.16 / 14.61 | 10.80 / 12.19 | TileLang 用 8 核 vs AscendC 20 核，task 分配更合理 |
>
> **v7 AscendC arch22 tiling 对齐（源码验证 `service_cube.h:54-65`）**：
> ```cpp
> M_BASIC_BLOCK = 256        // L1 层 M 维（S1*G），固定 256
> S2_BASIC_BLOCK = 256       // L1 层 S2 维，固定 256
> D_BASIC_BLOCK = 128        // K 维（head_dim）
> M_BASIC_BLOCK_L0 = 128     // L0 层 M
> S2_BASIC_BLOCK_L0 = 128    // L0 层 S2
> // Q L1: 2 bufs × 256×128, K L1: 3 bufs × 256×128
> // L0: A/B 2 bufs × 128×128, C 2 bufs × 128×128
> ```
>
> **v7 sort 缓存优化设计（源码验证 `service_vector.h:372-407`）**：
> ```cpp
> if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
>     SortAll + MergeSort  // 标准路径
> } else {
>     // actS1Size <= 4: 用 SortedBasicBlock_ 缓存 4 块
>     Sort<float, true>(SortedBasicBlock_[cache_idx], ...);
>     if (globalTopkUbCacheIdx == 3 || isS2End) {
>         MrgBasicBlock(...);  // 4-way merge, 减少 75% merge_sort 调用
>     }
> }
> ```
>
> **v7 数据范围约束（来自官方文档 `li.md` + tiling.h 源码验证）**：
> - query N1 **仅支持 64**（BSND: [B,S1,64,128], TND: [T1,64,128]），tiling.h:76 `QUERY_HEAD_NUM_LIMIT = 64`
> - key N2 **仅支持 1**（PA_BSND: [block_count,block_size,1,128], BSND: [B,S2,1,128], TND: [T2,1,128]）
> - D = 128 固定，tiling.h:73 `HEAD_DIM_LIMIT = 128`
> - sparse_count ∈ [1, 2048]，tiling.h:74 `SPARSE_LIMIT = 2048`
> - sparse_mode ∈ {0, 3}，tiling.h:75 `SPARSE_MODE_LOWER = 3`
> - block_size ∈ [16, 1024]，16 的整数倍
> - G = N1/N2 = 64/1 = **64**（固定，最常见场景），测试用例中有 N1=8/24 但官方文档限制 N1=64
> - **设计优先优化 G=64 场景**
>
> **v7 保留 v6 的全部 API 级优化**（row_expand_mul + reduce_sum merge + 动态核数 + pipe_barrier），仅更新 Cube tiling 和 Vector sort 策略。v6 的 5 项 API 优化全部保留。
>
> **v7 预期性能目标**：
> | 用例 | v6 实测(us) | v7 目标(us) | 80%目标(us) | 主要优化项 |
> |------|------------|------------|------------|-----------|
> | BSND_BSND | 111.64 | ≤85 | ≤92.83 | Cube tiling 对齐 + sort 缓存 + num_stages |
> | TND_TND | 55.10 | ≤48 | ≤50.67 | Cube tiling 对齐 + sort 缓存 |
> | BSND_PA_BSND | 18.78 | 保持 | ≤25.25 | 已达标，不退化 |
> | TND_PA_BSND | 超时 | ≤35 | ≤38.05 | block_size=16 修复 |
>
> ---
>
> **v6 关键优化（2 项 P0 指令融合 + 2 项 P1 工程改进 + 1 项 P2 参考 + 3 项新增 Q&A）**：
>
> | 优化 | 优先级 | v5 现状 | v6 优化内容 | 影响章节 |
> |------|--------|---------|------------|---------|
> | #1: `T.tile.row_expand_mul_experiment` 替代 broadcast+mul | 🔴 P0 | broadcast+mul 两步 + weight_2d(32KB UB) | 单条 `row_expand_mul_experiment` 融合 Brcb+Mul，移除 weight_2d(-32KB) | §3.2, §4.3, §4.5, §8.5, §10.2 Q23 |
> | #2: `T.reduce_sum(clear=False)` 融合 reduce+累加 | 🔴 P0 | reduce_sum+add 两步 + scores_partial(2KB UB) | 单条 `reduce_sum(clear=False)` merge 语义，移除 scores_partial(-2KB) | §3.2, §4.3, §4.5, §8.5, §10.2 Q24 |
> | #3: 动态核数 `cube_core_num` | 🟡 P1 | 硬编码 NUM_CORES=24 | host 侧 `torch.npu.get_device_properties("npu").cube_core_num` 动态获取 | §5.2, §8.5 |
> | #4: `T.pipe_barrier("v")` 轻量同步 | 🟡 P1 | `T.barrier_all()` 全管线屏障 | Vector 侧 groupInner 循环内用 `pipe_barrier("v")` 仅 Vector 管线屏障 | §7, §8.5 |
> | #5: `T.tile.brcb_experiment` 参考 | 🟢 P2 | 无 | 标注为可选（row_expand_mul_experiment 已融合 Brcb，单独广播场景使用） | §3.2 |
>
> **v6 收益汇总**：
> - Vector 侧 UB: v5 68KB → **v6 34KB**（-50%，余量 158KB / 192KB）
> - Vector 侧指令数: v5 6 条/iteration → **v6 4 条/iteration**（-33%）
> - num_stages 理论容量: v5 2（保守）→ **v6 3-5**（UB 34KB，buffer 在 body 外不被 ring-buffer）
> - 适配性: A2(24核)/A3(20核)/950 等不同芯片（动态核数）
>
> **v6 保留 v5 的全部架构级优化**（groupInner=16 + L0 inner split + Fixed Core + T.Pipelined + TND 直消），仅更新 API 选择和容量计算。v5 的 8 项修正（3 严重 + 5 中等）全部保留。
>
> **v6 API 全部经源码验证**：
> - `row_expand_mul_experiment`: `tilelang/language/ascend_tile.py:2353-2372`，使用验证 `examples/xattention/xattention.py:901-913`
> - `brcb_experiment`: `tilelang/language/ascend_tile.py:742-794`，使用验证 `examples/xattention/xattention.py:743`
> - `reduce_sum(clear=False)`: `api-compute.md:119-121`（clear=False 为 merge 语义），使用验证 `examples/HISA/paged_block_sparse_mqa_attn_expert.py:343`
> - `pipe_barrier("v")`: 使用验证 `examples/xattention/xattention.py:898`、`examples/HISA/paged_block_sparse_mqa_attn_expert.py:320`
> - `cube_core_num`: 使用验证 `examples/xattention/xattention.py:24`
>
> **v5 关键修正（3 项严重容量缺陷 + 5 项中等缺陷 + 3 项新增 Q&A）**：
>
> | 修正 | 优先级 | 缺陷来源 | 修正内容 | 影响章节 |
> |------|--------|---------|---------|---------|
> | #1: Vector groupInner=16 分块 | 🔴 严重 | v4 UB 超限: qk_ub=[512,512]×4=1MB>>192KB | 引入 groupInner=16（Vector 侧 G 维内层分块），qk_ub 改为 [groupInner, BLOCK_N]=[16,512]×4=32KB | §4.3, §4.5, §8.5, §10.2 Q20 |
> | #2: Cube L0 inner split | 🔴 严重 | v4 L0C 超限: C_L0=[512,128]×4=256KB>128KB | 引入 Cube 侧 L0 inner split (M_L0=128, N_L0=128)，C_L0=[128,128]×4=64KB | §4.3, §4.6, §5.2, §8.5, §10.2 Q21 |
> | #3: broadcast 范围缩小 | 🔴 严重 | v4 weight_2d=[512,512]×4=1MB>>192KB | broadcast 范围改为 [groupInner, BLOCK_N]=[16,512]×4=32KB，配合 groupInner 分块 | §4.5, §8.5, §10.2 Q20 |
> | #4: num_stages 建议 | 🟡 中等 | v4 num_stages=2 重叠不足 | 标注 "num_stages=2 保守值，Stage 2 实测后尝试 3-4" | §6, §12.1 |
> | #5: Fixed Core 小任务 | 🟡 中等 | total_tasks<24 核空转 | launch_core_num = min(total_tasks, NUM_CORES) | §5.2 |
> | #6: TND act_seq 标量读取 | 🟡 中等 | GM tensor 标量读取未验证 | 标注 Stage 2 需验证 + fallback 方案 | §4.9, §10.2 Q18 |
> | #7: sparse_values 精度标准 | 🟡 中等 | §9.3 缺 sparse_values 行 | 新增 FP16/BF16 sparse_values 容差行 | §9.3 |
> | #8: isSparseCountOver2K UB | 🟡 中等 | sparse_count>2048 UB 未更新 | 新增 isSparseCountOver2K 场景 UB 计算 | §4.5 |
>
> **v5 保留 v4 的 4 项架构级优化方向**（修正容量计算，不改变优化方向）：
>
> | 优化 | 来源 | v4→v5 变化 | 影响章节 |
> |------|------|-----------|---------|
> | 1: T.Pipelined 核间流水 | performance-antipatterns.md §"AIC/AIV 混合算子未开启 CV overlap" | 保留，num_stages=2 保守（Stage 2 尝试 3-4） | §2, §6, §7, §8.5 |
> | 2: Fixed Core 任务分配 | performance-antipatterns.md §"launch core 数关注项 A" | 保留，新增 min(total_tasks, NUM_CORES) | §5.2, §8.5 |
> | 3: kernel 直接消费 TND | flash_attn_optimize.md §9(a) "BSND 免转置" | 保留，act_seq 标量读取标注 Stage 2 验证 | §4.9, §8.5, §10.2 |
> | 4: broadcast + 整 tile mul | performance-antipatterns.md §"Vector Core 内逐元素 for loop" | **修正**: broadcast 范围从 [BLOCK_M,BLOCK_N] 缩小为 [groupInner,BLOCK_N] | §3.2, §8.5 |
>
> **v5 性能目标**：达到 AscendC 实测性能的 80%（msprof op 基准，4 种 layout 场景）
>
> **msprof 实测基准（AscendC）**：
> | Case | B | S1 | S2 | N1 | AscendC(us) | 80%目标(us) | Cube% | Vec0% |
> |------|---|----|----|----|------------|------------|-------|-------|
> | BSND_BSND | 16 | 5 | 3072 | 64 | 73.2 | 91.5 | 21% | 65% |
> | BSND_PA_BSND | 2 | 1 | 2048 | 8 | 21.8 | 27.2 | 1.1% | 19% |
> | TND_TND | 8 | 5 | 3072 | 24 | 40.3 | 50.4 | 8.9% | 57.5% |
> | TND_PA_BSND | 20 | 3 | 512 | 64 | 42.3 | 52.9 | N/A | N/A |
>
> **关键发现**：Cube 利用率普遍很低（1~21%），GEMM 不是瓶颈；**Vector 是真正的瓶颈**（weight mul + reduce + topk），Vec0 利用率最高仅 65%。
>
> **v5 容量验证关键结论**（修正 v4 的 3 项严重容量缺陷）：
> - Vector 侧 UB: qk_ub(32KB) + weight_2d(32KB) + scores(4KB) = **~68KB / 192KB ✓**（groupInner=16 分块）
> - Cube 侧 L0C: C_L0=[128,128]×4 = **64KB / 128KB ✓**（L0 inner split M_L0=128, N_L0=128）
> - Cube 侧 L1: Q_L1(128KB) + K_L1(128KB) = **256KB / 512KB ✓**（pipeline body 外分配，不 ×num_stages）
>
> **v6 容量验证关键结论**（row_expand_mul + reduce_sum merge 后）：
> - Vector 侧 UB: qk_ub(32KB) + weight_ub(64B) + scores_accum(2KB) = **~34KB / 192KB ✓**（移除 weight_2d 32KB + scores_partial 2KB）
> - Cube 侧 L0C: 不变 **64KB / 128KB ✓**
> - Cube 侧 L1: 不变 **256KB / 512KB ✓**
> - num_stages 理论容量: 34KB UB（buffer 在 body 外不被 ring-buffer），GM workspace ring-buffer 无容量限制 → **可尝试 num_stages=3-5**
>
> **v3 基线保留**：本版基于 v3 的 14 项差距分析（AscendC arch22 实现深度分析）继续演进，v3 的所有技术决策均保留不变。
>
> **Q14 关键结论（v3 保留）**：TileLang **支持** AIC:AIV=1:2 单 Kernel CV 协同，通过 `threads=2` + `TL_ASCEND_AUTO_CV_COMBINE: True` 实现。源码验证：`tilelang/language/kernel.py:253` (`assert threads in [1, 2]`)，参考实现 `examples/developer_mode/matmul_add_developer.py`、`examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py`。

## 1. 概述

### 1.1 算子名称

lightning_indexer

### 1.2 功能描述

基于 Query-Key 注意力打分 + 加权归约 + TopK 选择，为每个 query token 输出 Top-k 个 key 位置索引。用于 Lightning Attention 的稀疏索引选择阶段，支持 GQA（Grouped Query Attention）和 PageAttention（PA）KV 缓存布局。

### 1.3 数学公式

$$
\text{Indices} = \text{Top-}k\left\{ [1]_{1\times g} @ \left[ (W @ [1]_{1\times S_k}) \odot \text{ReLU}(Q_{index} @ K_{index}^T) \right] \right\}
$$

其中 $Q_{index} \in \mathbb{R}^{g \times d}$（g 为 GQA group size，d 为头维度），$K_{index} \in \mathbb{R}^{S_k \times d}$（$S_k$ 为上下文长度），$W \in \mathbb{R}^{g \times 1}$（每头权重）。输出为每个 token 对应的 Top-k 位置索引。

### 1.4 算法描述（6 步骤分解）

对于每个 batch $b$、query 位置 $s_1$、kv-head $n_2$：

| 步骤 | 数学表达 | Shape 变换 | 说明 |
|------|----------|-----------|------|
| 1. QK BMM | $\text{qk}[g, s_2] = Q[b,s_1,g,:] @ K[b,s_2,n_2,:].T$ | [g,d]×[d,s2_block]→[g,s2_block] | GQA: g=N1/N2, K 在 n2 维广播 |
| 2. ReLU | $\text{qk\_relu} = \max(0, \text{qk})$ | [g, s2_block] | 逐元素 |
| 3. 加权 | $\text{weighted}[g, s_2] = \text{qk\_relu}[g, s_2] \times W[b, s_1, g]$ | [g,s2_block]×[g,1]→[g,s2_block] | 每头权重广播乘 |
| 4. Group Reduce | $\text{scores}[s_2] = \sum_g \text{weighted}[g, s_2]$ | [g,s2_block]→[s2_block] | 对 g 维求和 |
| 5. Mask | $\text{scores}[\text{mask}=1] = -\infty$ | [s2] | sparse_mode=3: rightDownCausal |
| 6. TopK | $\text{indices} = \text{TopK}(\text{scores}, k)$ | [s2]→[sparse_count] | 降序取前 k 个位置索引 |

**关键观察**：步骤 1-4 中 S2 维度可分块并行（split-N，非 split-K），各 s2_block 独立计算部分分数，无需跨核归约。步骤 5-6 需要完整 S2 维度分数，在第二个 Kernel 中完成。

### 1.4.1 Layout 支持范围（一期，基于完整测试用例集 37 例分析）

| layout_query | layout_key | 测试用例数 | 占比 | 一期支持 | 处理方式 |
|-------------|-----------|-----------|------|---------|---------|
| BSND | BSND | 21 | 57% | ✅ | kernel 直接处理 |
| BSND | PA_BSND | 7 | 19% | ✅ | kernel 直接处理（block_table 间接寻址） |
| TND | PA_BSND | 5 | 14% | ✅ | host wrapper: TND→BSND（query），kernel 直接处理 key |
| TND | TND | 4 | 11% | ✅ | host wrapper: TND→BSND（query+key），kernel 处理 BSND |

> **TND 纳入一期的决策依据**：完整测试用例集中 9 个用例（24%）需要 TND 支持，不可推迟到二期。详见 §4.9 TND Layout 处理方案。

### 1.4.2 S2 范围与 UB 策略（基于完整测试用例集分析）

| S2 范围 | 测试用例数 | UB 策略 | 说明 |
|---------|-----------|---------|------|
| ≤ 16384 | 31 | 单次 topk（score_accum 在 UB） | score_accum + topk sort_tmp ≤ 160KB < 192KB |
| > 16384 | 6 | 分段 topk + merge_sort（score_accum 在 GM） | score_accum + sort_tmp 超 UB → 分段处理 |

> **分段 topk 的决策依据**：S2=131072 时 score_accum 需 512KB、topk sort_tmp 需 768KB，远超 192KB UB。详见 §4.5 UB 内存预算与 §10.2 Q8。

### 1.5 数据流图

```
Phase 1 (Kernel 1: score computation, B×S1×S2_blocks cores):
  GM[Query] → L1[Q_L1] ─┐
  GM[Key] → L1[K_L1] ───┤
                          ↓
                     L0C[C_L0] (GEMM, v5: L0 inner split [128,128])
                          ↓ (enable_relu, L0C→GM workspace)
                     GM[qk_workspace]
                          ↓ (GM→UB, cross_flag sync)
                     UB[qk_ub] → ReLU(已融合)
                          ↓ × UB[weight_ub] (v6: row_expand_mul_experiment 融合 Brcb+Mul)
                     UB[qk_ub] (原地乘, 无需 weight_2d)
                          ↓ reduce_sum(dim=0, clear=False) (v6: 融合 reduce+累加)
                     UB[scores_accum] (直接累加, 无需 scores_partial)
                          ↓
                     GM[Scores workspace: B,N2,S1,S2]

Phase 2 (Kernel 2: mask + topk, B×S1 cores):
  GM[Scores workspace] → UB[score_accum_ub]
                          ↓ apply rightDownCausal mask (compare+select)
                     UB[score_accum_ub]
                          ↓ T.tile.topk(K, actual_num=act_k)
                     UB[topk_dst_ub] (value-index pairs)
                          ↓ T.tile.gather_mask("P1010") → indices
                     UB[topk_index_ub]
                          ↓ T.tile.cast("CAST_ROUND")
                     UB[output_ub] (int32)
                          ↓
                     GM[sparse_indices: B,S1,N2,sparse_count]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: 混合模式——双 Kernel 架构（v6: Kernel 1 T.Pipelined + groupInner=16 + L0 inner split + row_expand_mul + reduce_sum merge）

> **v7.4 架构确认（Phase 2 验证, 见 §12.12）**：
> - **实际实现为单 Kernel + CV 分离**（非两个独立 `@tilelang.jit` kernel）. `lightning_indexer.py` 已是单 kernel 架构, 参考 `examples/sparse_flash_attention/sparse_flash_attn_pa_no_cv_pipeline.py` 模式.
> - **单 Kernel 内 CV 分离**: `T.Scope("C")`（Cube: GEMM）+ `T.Scope("V")`（Vector: weight mul + reduce + mask + sort + topk）+ 手动 `set_flag/wait_flag` 同步, `AUTO_SYNC=False`, `MEMORY_PLANNING=True`.
> - **本文档 "Kernel 1 (score)" / "Kernel 2 (topk)" 为逻辑 CV 计算划分**, 描述 Cube 段与 Vector 段的职责分离, 非物理双 kernel. v7.3 文档沿用此逻辑划分命名, 不代表两个独立 kernel.
> - **修正原因**: v7.3 文档多处 "双 Kernel" 字样易误解为两个 `@tilelang.jit` kernel, Phase 2 确认实际为单 kernel + T.Scope 分离. 架构选择保持当前单 kernel + T.Scope("C")/"V") + 手动同步, 不走双 kernel.

| Kernel | 编程模式 | threads | 流水 | 说明 |
|--------|---------|---------|------|------|
| Kernel 1 (score) | **Developer + CV 融合 + T.Pipelined** | **2** (AIC:AIV=1:2) | **T.Pipelined(num_stages=2, cross_interval=2)** | v6: row_expand_mul_experiment + reduce_sum(clear=False) + pipe_barrier("v") + 动态核数; UB 34KB |
| Kernel 2 (topk) | **Developer** (纯 Vector) | 1 (默认) | 不使用 T.Pipelined (UB 占用高, ring-buffer 风险) | alloc_shared (映射 UB), AUTO_SYNC 自动同步, T.tile.xxx |

> **v5 关键修正**：v4 改 `BLOCK_M = s1BaseSize × G` 后 UB/L0C 容量验证未同步更新，导致 N1≥16 时全部场景编译失败。v5 引入 **groupInner=16**（Vector 侧 G 维内层分块）+ **L0 inner split**（Cube 侧 M_L0=128, N_L0=128），修正 3 项严重容量缺陷。
>
> **T.Pipelined + threads=2 + AUTO_CV_COMBINE 组合**（源码验证，见 Q16）：
> - `T.Pipelined(S2_blocks, num_stages=2, cross_interval=2)` 包裹 S2 循环
> - 编译器自动将每次迭代的 Cube 部分（GEMM）和 Vector 部分（weight mul + reduce）流水化
> - 迭代 k 的 Cube 与迭代 k-1 的 Vector 重叠执行，隐藏 Cube 延迟
> - `cross_interval=2` 每 2 次迭代同步一次，减少跨核同步开销
> - 参考实现：`examples/pipeline/matmul_add_pipeline.py`（T.Pipelined + AUTO_CV_COMBINE + (cid, vid) 模式，已验证可运行）
>
> **num_stages=2 的选择依据**（Q16 回答 + v6 修正）：
> - T.Pipelined 会使编译器对 pipeline body 内的 UB buffer 做 ring-buffer（`flash_attn_bhsd_auto_pipeline_h16_d128.py` 文档记录了此行为）
> - **v6 Kernel 1 Vector 侧 UB 占用**（row_expand_mul + reduce_sum merge 后）：qk_ub(32KB) + weight_ub(64B) + scores_accum(2KB) = **~34KB**（v5: 68KB）
> - UB buffer 在 pipeline body **外**分配 → 不被 ring-buffer → num_stages 不受 UB×num_stages 限制
> - **v4 的错误**：v4 用 BLOCK_M=[512,512] 的 qk_ub(1MB) + weight_2d(1MB)，未引入 groupInner 分块，UB 严重超限
> - **安全策略**：UB buffer 在 pipeline body **外**分配（与 `matmul_add_pipeline.py` 一致），仅 GM workspace 做 ring-buffer
> - num_stages=2 是保守安全选择；**v6 标注：Stage 2 实测后尝试 num_stages=3-4**（UB=34KB 在 body 外, GM workspace 多 slot 无容量限制）
>
> **Kernel 2 不使用 T.Pipelined 的原因**：
> - Kernel 2 UB 占用高（score_accum 64KB + topk_sort_tmp 96KB = 160KB，接近 192KB 上限）
> - T.Pipelined ring-buffer 会立即使 UB 溢出（160KB × 2 = 320KB > 192KB）
> - Kernel 2 是纯 Vector（无 Cube），T.Pipelined 的 CV overlap 收益不适用
> - Kernel 2 的性能优化通过分段 topk + merge_sort 已解决 UB 容量问题

### 2.2 选型理由

- **Kernel 1 含 GEMM + element-wise 后处理** → CV 融合算子。采用 Developer 模式 + `threads=2` + `TL_ASCEND_AUTO_CV_COMBINE` + **T.Pipelined**：
  - 编译器自动将 GEMM 分配给 AIC、weight mul + reduce 分配给 AIV，自动插入 cross_flag 同步
  - **T.Pipelined(num_stages=2)** 使迭代间 Cube/Vector 重叠，消除核间气泡（v4 新增）
  - 代码更简洁（无需手动 T.Scope/set_cross_flag/wait_cross_flag），与 AscendC arch22 的 `KERNEL_TYPE_MIX_AIC_1_2` 模式对齐
  - `threads=2` 是 TileLang 对 AIC:AIV=1:2 的标准支持（`kernel.py:253` assert threads in [1,2]）
- **Kernel 2 纯 Vector**（mask + topk + cast），无 GEMM → Developer 模式 (threads=1) 即可，开启 `AUTO_SYNC` 自动同步。
- 两 Kernel 分离使得 B/S1/S2 三维并行可在 Kernel 1 充分展开（GEMM 密集计算），而 Kernel 2 仅需 B×S1 并行（topk 轻量计算），各取所需并行度。

### 2.3 模式影响

| 维度 | Kernel 1 (score, Developer + CV + T.Pipelined) | Kernel 2 (topk, Developer) |
|------|----------------------------------------------|---------------------------|
| threads | **2** (AIC:AIV=1:2) | 1 (默认) |
| 流水 | **T.Pipelined(num_stages=2, cross_interval=2)** | 无 |
| 内存分配 | alloc_shared (L1: Q/K) + alloc_fragment (L0C: GEMM) + alloc_shared (UB: Vector 中间, pipeline body 外) | alloc_shared (映射 UB) |
| 计算方式 | T.gemm_v0 (AIC 自动) + T.tile.mul/reduce_sum (AIV 自动) | T.tile.compare/topk/gather_mask/cast |
| CV 分割 | AUTO_CV_COMBINE 自动（编译器识别 GEMM → AIC, 其余 → AIV） | 不涉及（纯 Vector） |
| 同步方式 | AUTO_CV_SYNC 自动 cross_flag + T.Pipelined cross_interval 批量同步 + AUTO_SYNC | AUTO_SYNC 自动同步 |
| pass_configs | AUTO_CV_COMBINE, AUTO_CV_SYNC, AUTO_SYNC, MEMORY_PLANNING, PTO_USE_PIPE_IN_CV_COPY=False | AUTO_SYNC, MEMORY_PLANNING |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | qk = Q @ K^T | GEMM, M=G, N=block_N, K=D=128 |
| 2 | qk_relu = max(0, qk) | L0C→GM copy 时融合 enable_relu |
| 3 | weighted = qk_relu * W | 逐行乘权重 |
| 4 | scores = sum(weighted, dim=g) | 按组归约 |
| 5 | scores[mask] = -inf | rightDownCausal mask |
| 6 | indices = topk(scores, k) | 降序 top-k |

### 3.2 TileLang API 映射（全部经源码验证）

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 | 验证来源 |
|------|----------|-------------|------|------|---------|
| 1 | Q @ K^T | `T.gemm_v0(Q_L1, K_L1, C_L0, transpose_B=True, init=True)` | A/B: L1 shared, C: L0C fragment | Expert | api-compute.md, ascend_tile.py |
| 2 | ReLU | `T.copy(C_L0, qk_workspace[cid,:,:], enable_relu=True)` | L0C→GM, enable_relu 融合 | Expert | copy_op.py:257 npu_copy_v2, __init__.py:53 导出为 T.copy |
| 3a | weighted = qk * W (v6: row_expand_mul_experiment 融合) | `T.tile.row_expand_mul_experiment(qk_ub, qk_ub, weight_ub)` | v6: dst=src0×src1(行广播), dst/src0=[groupInner,BLOCK_N], src1=[groupInner] | Developer | ascend_tile.py:2353-2372, xattention.py:901-913 |
| 3b | weighted = qk * W (v3 fallback: 逐行 mul) | `T.tile.mul(qk_ub[m, :], qk_ub[m, :], weight_ub[m])` | dst=src0×src1, src1 可标量 | Expert | ascend_tile.py:271 |
| 4 | sum over g (v6: reduce_sum clear=False 融合累加) | `T.reduce_sum(qk_ub, scores_accum, dim=0, clear=False)` | v6: clear=False merge 语义, reduce 结果直接累加到 scores_accum | Expert | reduce_ascend.py:391, api-compute.md:119-121 |
| 5a | mask 比较 | `T.tile.createvecindex(col_idx, s2_block*block_N)` + `T.tile.compare(mask_ub, col_idx_f, cutoff_f, "GE")` | 生成索引→比较→bitmask | Developer | ascend_tile.py:1416, 1626 |
| 5b | mask 应用 | `T.tile.select(scores_tile, mask_ub, -T.infinity(dtype), scores_tile, "VSEL_TENSOR_SCALAR_MODE")` | 按 mask 选 -inf | Developer | ascend_tile.py:550 |
| 6a | TopK | `T.tile.topk(topk_dst_ub, score_accum_ub, K, actual_num)` | dst=2K (val,idx pairs), actual_num=act_k | Developer | ascend_tile.py:419 |
| 6b | 提取索引 | `T.tile.gather_mask(topk_index_ub, topk_dst_ub, "P1010")` | 提取奇数位 (idx) | Developer | ascend_tile.py:465 |
| 6c | float→int | `T.tile.cast(output_ub, topk_index_ub, "CAST_ROUND", K)` | float→int32 | Developer | ascend_tile.py:1716 |
| **v4: 核间流水** | S2 循环 CV overlap | `for s2 in T.Pipelined(S2_blocks, num_stages=2, cross_interval=2):` | num_stages=2 (UB 安全), cross_interval=2 (批量同步) | Developer | pipeline.py:11, matmul_add_pipeline.py:46 (已验证可运行) |
| C/V sync | Cube→Vector | `T.set_cross_flag("FIX", 0)` + `T.wait_cross_flag(0)` | pipe="FIX"(L0C→GM), flag=0, mode=2(默认) | Expert | ascend.py:116, 138 |
| 核内同步 (v6) | Vector 管线屏障 | `T.pipe_barrier("v")` | v6: 仅 Vector 管线屏障 (替代部分 barrier_all) | Developer | xattention.py:898, HISA/paged_block_sparse_mqa_attn_expert.py:320 |
| 核内同步 (fallback) | 全管线屏障 | `T.barrier_all()` | 所有管线完成 (Cube 侧或 fallback) | Expert | ascend.py:200 |
| **v6 候选: Brcb 单独广播** | 行广播 (无乘法) | `T.tile.brcb_experiment(dst, src, repeat_times, dst_blk_stride, dst_repeat_stride)` | v6 可选: row_expand_mul_experiment 已融合 Brcb, 仅单独广播场景使用 | Developer | ascend_tile.py:742-794, xattention.py:743 |
| **v4 候选: 布局优化** | L1 分形优化 | `T.annotate_layout({Q_L1: make_zn_layout(Q_L1), K_L1: make_nz_layout(K_L1)})` | ZN/NZ 分形 (Stage 2 候选) | Developer | intrinsics/ascend_layout.py:34,89; __init__.py:101 |

### 3.3 计算伪代码

> **v6 说明**：以下 Kernel 1 Expert 伪代码为 v3 fallback 保留。v6 推荐 Developer + T.Pipelined 版本见 **§8.5**（Fixed Core + T.Pipelined + TND 直消 + row_expand_mul + reduce_sum merge）。

#### Kernel 1: lightning_indexer_score (v3 Expert fallback, 保留不删)

```python
@tilelang.jit(out_idx=[7], workspace_idx=[6], pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
})
def lightning_indexer_score(B, N1, N2, S1, S2, D, G, BLOCK_N, MAX_BLOCK_NUM,
                             layout_query, layout_key, input_dtype, calc_dtype):
    S2_blocks = T.ceildiv(S2, BLOCK_N)
    total_blocks = B * S1 * S2_blocks

    @T.prim_func
    def main(
        Query: T.Tensor((B, S1, N1, D), input_dtype),
        Key: T.Tensor((block_count, BLOCK_N, N2, D), input_dtype),  # PA: block_count; BSND: B
        Weights: T.Tensor((B, S1, N1), input_dtype),
        actual_seq_q: T.Tensor((B,), "int32"),
        actual_seq_k: T.Tensor((B,), "int32"),
        block_table: T.Tensor((B, MAX_BLOCK_NUM), "int32"),
        qk_workspace: T.Tensor((total_blocks, G, BLOCK_N), calc_dtype),  # workspace_idx=[6]
        Scores: T.Tensor((B, N2, S1, S2), calc_dtype),                   # out_idx=[7]
    ):
        with T.Kernel(total_blocks, is_npu=True) as (cid, vid):
            b = cid // (S1 * S2_blocks)
            s1 = (cid // S2_blocks) % S1
            s2_block = cid % S2_blocks

            act_q = actual_seq_q[b]
            act_k = actual_seq_k[b]

            if s1 < act_q:
                s2_start = s2_block * BLOCK_N
                if s2_start < act_k:
                    with T.Scope("C"):
                        Q_L1 = T.alloc_L1((G, D), input_dtype)
                        K_L1 = T.alloc_L1((BLOCK_N, D), input_dtype)
                        C_L0 = T.alloc_L0C((G, BLOCK_N), calc_dtype)

                        # Load Q[b, s1, :, :] → [G, D]
                        T.copy(Query[b, s1, :, :], Q_L1)

                        # Load K block (PA: indirect via block_table; BSND: direct)
                        if layout_key == "PA_BSND":
                            k_block_id = block_table[b, s2_block]
                            T.copy(Key[k_block_id, :, 0, :], K_L1)
                        else:  # BSND
                            T.copy(Key[b, s2_start:s2_start + BLOCK_N, 0, :], K_L1)

                        # GEMM: [G, D] × [D, BLOCK_N] → [G, BLOCK_N]
                        T.gemm_v0(Q_L1, K_L1, C_L0, transpose_B=True, init=True)

                        # L0C → GM workspace with ReLU fusion
                        T.copy(C_L0, qk_workspace[cid, :, :], enable_relu=True)
                        T.set_cross_flag("FIX", 0)

                    with T.Scope("V"):
                        T.wait_cross_flag(0)
                        qk_ub = T.alloc_ub((G, BLOCK_N), calc_dtype)
                        weight_ub = T.alloc_ub((G,), calc_dtype)
                        scores_ub = T.alloc_ub((BLOCK_N,), calc_dtype)

                        # GM workspace → UB
                        T.copy(qk_workspace[cid, :, :], qk_ub)

                        # Weight multiplication: [G, BLOCK_N] × [G, 1] → [G, BLOCK_N]
                        T.copy(Weights[b, s1, :], weight_ub)
                        for g in T.serial(G):
                            T.tile.mul(qk_ub[g, :], qk_ub[g, :], weight_ub[g])

                        # Group reduce: sum over G → [BLOCK_N]
                        T.reduce_sum(qk_ub, scores_ub, dim=0)

                        # Write partial scores to Scores workspace
                        T.copy(scores_ub, Scores[b, 0, s1, s2_start:s2_start + BLOCK_N])
```

#### Kernel 2: lightning_indexer_topk (Developer 纯 Vector)

```python
@tilelang.jit(out_idx=[-1], pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
})
def lightning_indexer_topk(B, S1, S2, N2, SPARSE_COUNT, MAX_S2, BLOCK_N, calc_dtype):
    @T.prim_func
    def main(
        Scores: T.Tensor((B, N2, S1, S2), calc_dtype),
        actual_seq_q: T.Tensor((B,), "int32"),
        actual_seq_k: T.Tensor((B,), "int32"),
        Output: T.Tensor((B, S1, N2, SPARSE_COUNT), "int32"),
    ):
        with T.Kernel(B * S1, is_npu=True) as (cid, vid):
            b = cid // S1
            s1 = cid % S1

            act_q = actual_seq_q[b]
            act_k = actual_seq_k[b]

            # Buffers (static shape for topk requirement)
            score_accum = T.alloc_shared((MAX_S2,), calc_dtype)
            topk_dst = T.alloc_shared((2 * SPARSE_COUNT,), calc_dtype)
            topk_index = T.alloc_shared((SPARSE_COUNT,), calc_dtype)
            output_ub = T.alloc_shared((SPARSE_COUNT,), "int32")
            col_idx = T.alloc_shared((BLOCK_N,), "int32")
            col_idx_f = T.alloc_shared((BLOCK_N,), calc_dtype)
            mask_ub = T.alloc_shared((BLOCK_N // 8,), "uint8")
            scores_tile = T.alloc_shared((BLOCK_N,), calc_dtype)

            if s1 < act_q:
                # Initialize score_accum with -inf (for positions beyond act_k)
                T.tile.fill(score_accum, -T.infinity(calc_dtype))

                # Load scores in blocks — 静态循环边界 (MAX_S2 // BLOCK_N) + 运行时 if 条件
                # (ascend-constraints: T.serial 循环次数不能依赖 tensor 值, 用编译期 MAX_S2)
                for s2_block in T.serial(T.ceildiv(MAX_S2, BLOCK_N)):
                    s2_start = s2_block * BLOCK_N
                    if s2_start < act_k:
                        T.copy(Scores[b, 0, s1, s2_start:s2_start + BLOCK_N], scores_tile)

                        # Apply rightDownCausal mask: j >= cutoff → -inf
                        cutoff = act_k - act_q + s1 + 1
                        T.tile.createvecindex(col_idx, s2_start)
                        T.copy(col_idx, col_idx_f)
                        T.tile.compare(mask_ub, col_idx_f, T.float32(cutoff), "GE")
                        T.tile.select(scores_tile, mask_ub,
                                      -T.infinity(calc_dtype), scores_tile,
                                      "VSEL_TENSOR_SCALAR_MODE")

                        # Scatter into score_accum
                        for i in T.serial(BLOCK_N):
                            score_accum[s2_start + i] = scores_tile[i]

                # TopK (actual_num = act_k, positions beyond act_k are -inf)
                T.tile.topk(topk_dst, score_accum, SPARSE_COUNT, act_k)
                T.tile.gather_mask(topk_index, topk_dst, "P1010")
                T.tile.cast(output_ub, topk_index, "CAST_ROUND", SPARSE_COUNT)

                T.copy(output_ub, Output[b, s1, 0, 0:SPARSE_COUNT])
            else:
                # Invalid query position: fill output with -1
                T.tile.fill(output_ub, -1)
                T.copy(output_ub, Output[b, s1, 0, 0:SPARSE_COUNT])
```

### 3.4 API 可行性确认

| API | 来源验证 | 状态 |
|-----|---------|------|
| T.copy (npu_copy_v2) | copy_op.py:257, __init__.py:53 (`npu_copy_v2 as copy`) | ✅ enable_relu 参数确认可用 |
| T.gemm_v0 | api-compute.md, examples/gemm/ | ✅ transpose_B, init 参数确认 |
| T.tile.topk | ascend_tile.py:419 | ✅ 要求 src 静态 shape（用 MAX_S2） |
| T.tile.gather_mask | ascend_tile.py:465 | ✅ "P1010" 提取奇数位（索引） |
| T.tile.cast | ascend_tile.py:1716 | ✅ "CAST_ROUND" 模式 |
| T.tile.mul | ascend_tile.py:271 | ✅ src1 可为标量 |
| T.tile.compare | ascend_tile.py:1626 | ✅ mode="GE" |
| T.tile.select | ascend_tile.py:550 | ✅ "VSEL_TENSOR_SCALAR_MODE" |
| T.tile.createvecindex | ascend_tile.py:1416 | ✅ 生成等差索引序列 |
| T.tile.fill | ascend_tile.py:221 | ✅ 填充常量 |
| T.reduce_sum | reduce_ascend.py:391 | ✅ dim, clear, real_shape 参数。**v6: clear=False 为 merge 语义**（api-compute.md:119-121: new_out = old_out + reduced_result）|
| T.set_cross_flag | ascend.py:116 | ✅ pipe="FIX", flag, mode=2(默认, AIC-AIV 同组) |
| T.wait_cross_flag | ascend.py:138 | ✅ flag 参数, pipe 仅 A5 平台 |
| T.barrier_all | ascend.py:200 | ✅ 全管线屏障 |
| T.ceildiv | tilelang/language/ | ✅ 向上取整除法 |
| T.alloc_L1/L0C/ub | api-kernel-memory.md | ✅ Expert 模式显式层级 |
| T.alloc_shared | api-kernel-memory.md | ✅ Developer 模式, 编译器映射 UB |
| **T.Pipelined** (v4) | pipeline.py:11 | ✅ `Pipelined(start, stop, num_stages, order, stage, sync, group, cross_interval=1)`, matmul_add_pipeline.py:46 已验证与 AUTO_CV_COMBINE 组合可运行 |
| **T.tile.broadcast** (v4) | ascend_tile.py:2031 | ✅ `broadcast(dst, src, axis=None)`, 1D→2D broadcast, dst/src 须在 UB. **v6: 仍可用作 fallback, 但推荐 row_expand_mul_experiment** |
| **T.tile.mul_add_dst** (v4 候选) | ascend_tile.py:1168 | ✅ `mul_add_dst(dst, src0, src1)` = dst = src0*src1 + dst, 融合 mul+add. **v6: 被 reduce_sum(clear=False) 替代, 降级为 fallback** |
| **T.tile.row_expand_mul_experiment** (v6 P0) | ascend_tile.py:2353-2372 | ✅ `row_expand_mul_experiment(dst, src0, src1, tmp=None)`, dst[i,j]=src0[i,j]*src1[i]. AscendC: brcb+mul_mask. PTO: TROWEXPANDMUL. 使用验证: xattention.py:901-913 |
| **T.tile.brcb_experiment** (v6 P2) | ascend_tile.py:742-794 | ✅ `brcb_experiment(dst, src, repeat_times, dst_blk_stride, dst_repeat_stride)`, BRCB 广播. 使用验证: xattention.py:743. **可选: row_expand_mul 已融合 Brcb** |
| **T.reduce_sum(clear=False)** (v6 P0) | reduce_ascend.py:391, api-compute.md:119-121 | ✅ clear=False merge 语义: new_out = old_out + reduced_result. 使用验证: HISA/paged_block_sparse_mqa_attn_expert.py:343 (clear=True 对照) |
| **T.pipe_barrier("v")** (v6 P1) | xattention.py:898, HISA/paged_block_sparse_mqa_attn_expert.py:320 | ✅ 仅 Vector 管线屏障, 比 barrier_all 更轻量 (不等 Cube/MTE2/MTE3) |
| **torch.npu.get_device_properties("npu").cube_core_num** (v6 P1) | xattention.py:24 | ✅ host 侧动态获取核数, 适配 A2(24)/A3(20)/950 |
| **T.annotate_layout** (v4 候选) | __init__.py:101 | ✅ `annotate_layout(layout_map: Dict)`, 配合 make_zn_layout/make_nz_layout, Stage 2 候选 |
| **make_zn_layout / make_nz_layout** (v4 候选) | intrinsics/ascend_layout.py:34,89 | ✅ L1 分形布局优化, intrinsics/__init__.py:19 导出, Stage 2 候选 |

---

## 3.5 技术约束确认

### 3.5.1 本项目已知限制检查

| 约束 | 本算子是否涉及 | 处理方案 |
|------|---------------|----------|
| 不支持三维 Kernel | **Yes** — B×S1×S2_blocks 需三维并行 | v4: Fixed Core `T.Kernel(NUM_CORES=24, threads=2)`, 每核多任务循环展开 B×S1 维度, S2 用 T.Pipelined |
| threads 参数限制（仅 1 或 2） | **Yes** (v4) — Kernel 1 使用 threads=2 | Kernel 1: `threads=2` (AIC:AIV=1:2 CV 融合), Kernel 2: threads=1 (纯 Vector) |
| 动态循环边界不支持 | **Yes** — act_q/act_k 为运行时 tensor 值 | Kernel 2: `T.serial(T.ceildiv(MAX_S2, BLOCK_N))` (编译期静态) + `if s2_start < act_k` 运行时条件判断；Kernel 1: `if s1 < act_q` / `if s2_start < act_k` 条件跳过无效块；v4: TND 前缀和作为数据访问索引（非循环边界） |
| 流水线不支持动态边界 | **No** (v4) — T.Pipelined 的 S2_blocks 为编译期静态值 `T.ceildiv(S2, s2BaseSize)` | v4: T.Pipelined(s2BaseNum, num_stages=2) 的 s2BaseNum 是编译期常量 |

### 3.5.2 参考实现差异说明

| 差异项 | 参考实现 (history_version) | v3 设计 | v4 设计 (新) | 转换方案 |
|--------|--------------------------|---------|-------------|----------|
| Kernel 并行度 | T.Kernel(B*N2) — 仅 B×N2 并行 | T.Kernel(B*S1*S2_blocks) — 按任务数 launch | T.Kernel(NUM_CORES=24, threads=2) — Fixed Core + 每核多任务 | v4: Fixed Core + T.Pipelined |
| S2 处理 | Vector 内层 serial 循环 | Kernel 1 并行切分 S2_blocks (split-N) | v4: T.Pipelined(s2BaseNum, num_stages=2) 核间流水 | v4: CV overlap |
| TND layout | 不支持 | host wrapper TND→BSND 转换 | kernel 直接 strided DMA 消费 | v4: 消除转置 kernel |
| Weight mul | 逐行 for loop | for m in T.serial(BLOCK_M): T.tile.mul | broadcast + 整 tile mul (v5: groupInner 分块) | v4: 减少指令下发; **v6: row_expand_mul_experiment 融合 Brcb+Mul** (替代 broadcast+mul) |
| Mask | 不支持 | Kernel 2 在线生成 rightDownCausal | 同 v3 (不变) | compare + select |
| PA_BSND | 不支持 | Kernel 1 通过 block_table 间接寻址 | 同 v3 (不变) | 参考 SFA PA 示例 |
| actual_seq | 不支持 | 运行时条件判断 | v4: TND 前缀和作为数据访问索引 | T.ceildiv + if |
| 编程模式 | Expert 单 Kernel | 混合: Kernel 1 Expert + Kernel 2 Developer | v4: Kernel 1 Developer + threads=2 + T.Pipelined + Kernel 2 Developer | v4: 全 Developer 模式 |
| weights dtype | float32 (calc_dtype) | bfloat16/float16 (input_dtype) | 同 v3 (不变) | **用户明确禁止 float32 weights**，在 Vector 阶段 cast 到 calc_dtype 计算 |

### 3.5.3 本项目同类实现参考

| 文件路径 | 相似度 | 关键参考点 |
|----------|--------|-----------|
| `examples/lightning_indexer/history_version/example_lightning_indexer_original.py.bak` | 高度相似（前身） | T.set_cross_flag/wait_cross_flag, T.tile.topk/gather_mask/cast, T.reduce_sum, enable_relu copy |
| `examples/sparse_flash_attention/example_sparse_flash_attn_mask_pa.py` | 高度相似 (PA + mask) | block_table 间接寻址 (`block_table[b_i, block_idx]`), T.tile.compare/select mask 应用, T.set_cross_flag 多 flag 协同 |
| `examples/topk_selector/example_topk_selector.py` | 高度相似 (TopK) | T.tile.topk + gather_mask + cast 完整流程, score_accum 累积模式 |
| `examples/seer_attention/block_sparse_attn.py` | 中度相似 (CV 融合 + mask) | T.Scope("C")/T.Scope("V") 模式, createvecindex + compare + select causal mask |
| `examples/gemm_splitk/example_tilelang_gemm_splitk.py` | 中度相似 (并行切分) | T.Kernel 展平多维度并行, T.tile.atomic_add 归约（本设计用 split-N 无需归约） |
| `examples/sort/example_merge_sort.py` | 参考 (排序) | T.tile.merge_sort 多路归并（未来扩展: 分段 topk 合并可用） |
| **`examples/pipeline/matmul_add_pipeline.py`** (v4 新增) | **高度相似 (T.Pipelined + CV)** | **v4 核心参考**: T.Pipelined(loop_k, num_stages=3) + AUTO_CV_COMBINE + (cid, vid) 模式, 已验证可运行 |
| **`examples/pipeline/sparse_flash_attn_gqa_pipeline.py`** (v4 新增) | **高度相似 (T.Pipelined + CV + Expert 风格)** | **v4 核心参考**: T.Pipelined(NI, num_stages=2) + AUTO_CV_COMBINE + alloc_L1/ub, 已验证可运行 |
| **`examples/deepseek_v4/sparse_attention.py`** (v4 新增) | **中度相似 (strided DMA)** | **v4 TND 参考**: q_shape=[b,m,h,d], T.copy(q[by,bx,:,:], q_l1) strided DMA 模式 |
| **`examples/moe_token_permute/moe_token_permute.py`** (v4 新增) | **中度相似 (Fixed Core)** | **v4 Fixed Core 参考**: T.Kernel(actual_cores, ...) + 每核多任务循环, workspace 按 core 分配 |
| **`examples/flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_h16_d128.py`** (v4 新增) | **反面参考 (UB 超限)** | **v4 风险警示**: T.Pipelined + threads=2 num_stages=8 时 UB ring-buffer 超限 (KNOWN BROKEN), 本设计用 num_stages=2 规避 |
| **`examples/xattention/xattention.py`** (v6 新增) | **高度相似 (row_expand_mul + brcb + pipe_barrier)** | **v6 核心参考**: `:901-913` row_expand_mul_experiment (dst=src0 原地乘), `:743` brcb_experiment, `:898` pipe_barrier("v"), `:24` cube_core_num 动态核数 |
| **`examples/HISA/paged_block_sparse_mqa_attn_expert.py`** (v6 新增) | **中度相似 (pipe_barrier + Expert 优化)** | **v6 参考**: `:320` pipe_barrier("v"), `:343` reduce_sum(clear=True) (对照 clear=False), 4×K L1 + 4×L0C + MTE2∥V overlap (Stage 3 fallback) |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape (BSND) | Shape (TND) | dtype | 说明 |
|--------|-------------|-------------|-------|------|
| Query | [B, S1, N1, D] | [T1, N1, D] | bfloat16 / float16 | N1∈{8,16,24,32,64}, D=128, 不支持非连续 |
| Key (PA_BSND) | [block_count, block_size, N2, D] | — | bfloat16 / float16 | N2=1, block_size∈[16,1024] 且 16 整数倍 |
| Key (BSND) | [B, S2, N2, D] | [T2, N2, D] | bfloat16 / float16 | N2=1 |
| Weights | [B, S1, N1] | [T1, N1] | bfloat16 / float16 | **绝对禁止 float32**（用户明确） |
| actual_seq_q | [B] (非前缀和) | [B] (**前缀和**) | int32 | BSND: 可为 None(=S1); TND: 必传, 元素为累积 token 数 |
| actual_seq_k | [B] (非前缀和) | [B] (**前缀和**) | int32 | PA_BSND: 非前缀和; BSND: 可为 None(=S2); TND: 前缀和 |
| block_table | [B, maxBlockNumPerSeq] | [B, maxBlockNumPerSeq] | int32 | 仅 PA 场景必传, 值为物理 block ID |

> **TND 前缀和说明**：TND 的 act_seq_q[i] 表示前 i+1 个 batch 的 token 数总和（非递减）。例：B=2 时 act_seq_q=[3, 6] 表示 batch 0 有 3 个 token，batch 1 有 3 个 token，T1=6。host wrapper 负责前缀和→per-batch 转换（详见 §4.9）。

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| sparse_indices | [B, S1, N2, sparse_count] (BSND) | int32 | 无效位置填 -1 |
| sparse_values | (optional) same shape as sparse_indices | **bfloat16 / float16** | return_value=True 时输出, 一期支持。**v3 修正**：dtype 为 FP16/BF16（与 query/key 一致），非 int32。 AscendC 中 Cast(FP32→K_T) 后输出 |

> **v3 dtype 修正依据**：
> - AscendC 代码 `lightning_indexer_kernel.h:112`: `GlobalTensor<K_T> valueOutGm` (K_T = float16_t/bfloat16_t)
> - AscendC `ProcessVec`: `Cast(valueULocal1, outValueUb, CAST_ROUND, copyLen)` 将 FP32 分数 Cast 到 K_T
> - AscendC `CleanInvalidOutput`: `negInf = 0xFC00`(FP16) 或 `0xFF80`(BF16)
> - aclnn 文档 `aclnnLightningIndexer.md:301`: sparseValuesOut dtype = FLOAT16、BFLOAT16
>
> **sparse_values 计算逻辑**：topk 输出的 value（FP32 注意力分数）经 `T.tile.cast(dst, src, "CAST_ROUND", K)` 从 FP32 转为 FP16/BF16 后输出。无效位置填 -inf（FP16: 0xFC00, BF16: 0xFF80）。

### 4.3 中间缓冲区

#### Kernel 1 (score computation, v6: Fixed Core + T.Pipelined + groupInner=16 + L0 inner split + row_expand_mul + reduce_sum merge)

**v6 关键修正**：v5 的 broadcast+mul 两步 + reduce+add 两步，被 `row_expand_mul_experiment`（融合 Brcb+Mul）+ `reduce_sum(clear=False)`（融合 reduce+累加）替代，移除 weight_2d 和 scores_partial 两个 UB buffer。
**v5 保留**：groupInner=16（Vector 侧 G 维内层分块）+ L0 inner split（Cube 侧 M_L0=128, N_L0=128）

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| Q_L1 | [BLOCK_M, D] | input_dtype | L1 (Developer: alloc_shared) | Query tile 缓存 (BLOCK_M = s1BaseSize × G, 一次加载完整 tile) |
| K_L1 | [BLOCK_N, D] | input_dtype | L1 (Developer: alloc_shared) | Key tile 缓存 |
| **C_L0** (v5 修正) | **[M_L0=128, N_L0=128]** | calc_dtype (float32) | L0C (Developer: alloc_fragment) | GEMM 累加输出 (L0 inner split, 非 [BLOCK_M,BLOCK_N]) |
| qk_workspace | **[NUM_CORES, num_stages, BLOCK_M, BLOCK_N]** (v4) | calc_dtype | GM (workspace_idx) | C→V 中转: L0C→GM(ReLU)→UB, v4: Fixed Core + ring-buffer |
| **qk_ub** (v5 修正, v6 保留) | **[groupInner=16, BLOCK_N]** | calc_dtype | UB (Developer: alloc_shared) | QK ReLU 结果 (Vector 侧, groupInner 分块). **v6: row_expand_mul 的 dst=src0, 原地乘** |
| **weight_ub** (v5 修正, v6 保留) | **[groupInner=16]** | calc_dtype | UB | 权重 (groupInner 分块). **v6: row_expand_mul 的 src1 (1D 行向量)** |
| ~~weight_2d~~ | ~~[groupInner=16, BLOCK_N]~~ | ~~calc_dtype~~ | ~~UB~~ | **v6 移除**: row_expand_mul_experiment 内部完成 Brcb 广播, 不需要显式 2D 权重 buffer |
| ~~scores_partial~~ | ~~[BLOCK_N]~~ | ~~calc_dtype~~ | ~~UB~~ | **v6 移除**: reduce_sum(clear=False) merge 语义直接累加到 scores_accum, 不需要中间 buffer |
| **scores_accum** (v5 修正, v6 保留) | **[BLOCK_N]** | calc_dtype | UB | 跨 g_idx 累加的分数 (固定大小, 与 G 无关). **v6: reduce_sum(clear=False) 的 dst, 直接累加** |
| Scores | [B, N2, S1, S2] | calc_dtype | GM (out_idx) | 完整分数矩阵 (Kernel 1 输出, Kernel 2 输入) |

> **v6 buffer 变化总结**：移除 weight_2d (32KB) + scores_partial (2KB) = **-34KB UB**。v5 7 个 UB buffer → v6 5 个 UB buffer。
>
> **v5 groupInner=16 参数来源**：AscendC arch22 `lightning_indexer_service_vector.h:198` `groupInner_ = 16`（固定 16）
>
> **v5 L0 inner split 参数来源**：AscendC arch22 `lightning_indexer_service_cube.h:54-65` `M_BASIC_BLOCK_L0=128, S2_BASIC_BLOCK_L0=128`
>
> **v6 row_expand_mul_experiment 参数对齐**（Q23 详述）：
> - dst = qk_ub [groupInner=16, BLOCK_N=512] — 2D, 原地更新
> - src0 = qk_ub [groupInner=16, BLOCK_N=512] — 2D, 与 dst 同一 buffer
> - src1 = weight_ub [groupInner=16] — 1D, 每行一个标量
> - 语义: dst[i,j] = src0[i,j] * src1[i]，等价于 v5 的 broadcast+mul
>
> **v6 reduce_sum(clear=False) 参数对齐**（Q24 详述）：
> - src = qk_ub [groupInner=16, BLOCK_N=512] — reduce dim=0
> - dst = scores_accum [BLOCK_N=512] — clear=False, new_out = old_out + reduced_result
> - 等价于 v5 的 reduce_sum + add 两步
>
> **v5 关键维度说明**：
> - BLOCK_M = s1BaseSize × G（联合切分保留，如 8×64=512）
> - groupInner = 16（Vector 侧 G 维内层分块，固定值）
> - M_L0 = 128, N_L0 = 128（Cube 侧 L0 级 GEMM 块大小）
> - qk_workspace 仍是 [BLOCK_M, BLOCK_N]（GM 无容量限制），Vector 侧分次读取 [groupInner, BLOCK_N] 切片

#### Kernel 2 (topk) — S2 ≤ MAX_S2_UB (16384) 单次 topk 路径

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| score_accum | [MAX_S2] | calc_dtype | UB (Developer: alloc_shared) | 完整 S2 分数累积 |
| scores_tile | [BLOCK_N] | calc_dtype | UB | 单块分数处理 (mask 应用) |
| col_idx | [BLOCK_N] | int32 | UB | 位置索引向量 |
| col_idx_f | [BLOCK_N] | calc_dtype | UB | 浮点索引 (compare 用) |
| mask_ub | [BLOCK_N // 8] | uint8 | UB | 比较结果 bitmask |
| topk_dst | [2 * SPARSE_COUNT] | calc_dtype | UB | topk 输出 (val,idx pairs) |
| topk_index | [SPARSE_COUNT] | calc_dtype | UB | 提取的索引 (float) |
| output_ub | [SPARSE_COUNT] | int32 | UB | 最终输出索引 (int32) |
| **topk_sort_tmp** (注入) | [MAX_S2 × 6] | uint8 | UB (pass 自动注入) | **topk 内部排序临时区**（§4.5 详述） |

#### Kernel 2 (topk) — S2 > MAX_S2_UB (16384) 分段 topk 路径

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| score_accum_gm | [MAX_S2] | calc_dtype | **GM (workspace)** | 完整 S2 分数累积（放不下 UB） |
| segment_buf | [SEGMENT_SIZE] | calc_dtype | UB | 单段分数（从 GM 加载） |
| segment_topk_dst | [2 * SPARSE_COUNT] | calc_dtype | UB | 单段 topk 输出 (val,local_idx pairs) |
| running_topk | [2 * SPARSE_COUNT] | calc_dtype | UB | 累积 top-k 候选 (val,global_idx pairs) |
| merge_output | [4 * SPARSE_COUNT] | calc_dtype | UB | merge_sort 输出 (2-way merge) |
| topk_index | [SPARSE_COUNT] | calc_dtype | UB | 提取的索引 (float) |
| output_ub | [SPARSE_COUNT] | int32 | UB | 最终输出索引 (int32) |
| mask_buf | [BLOCK_N // 8] | uint8 | UB | mask bitmask (mask 应用阶段) |
| scores_tile | [BLOCK_N] | calc_dtype | UB | 单块 mask 处理 |
| col_idx / col_idx_f | [BLOCK_N] | int32 / calc_dtype | UB | mask 索引向量 |

> **SEGMENT_SIZE = 8192**：保证单段 topk 的 UB 占用（segment_buf 32KB + sort_tmp 48KB = 80KB）在 192KB 内。
> **分段流程**：①mask+scatter 到 GM score_accum_gm → ②逐段 topk → ③merge_sort 合并 → ④输出。详见 §5.2 和 §6。

### 4.4 内存搬运路径

```
Kernel 1 (v6: Developer + CV 融合 + T.Pipelined + groupInner=16 + L0 inner split + row_expand_mul + reduce_sum merge):
  GM[Query] --T.copy(strided DMA)--> L1[Q_L1]               (v5: TND 直接消费, 无 host 转置)
  GM[Key]   --T.copy--> L1[K_L1]                             (PA: 间接 block_table; TND: strided DMA; BSND: 直接)
  L1[Q_L1] + L1[K_L1] --T.gemm_v0(切片)--> L0C[C_L0]        (AIC, v5: L0 inner split [M_L0=128,N_L0=128], T.Pipelined 流水)
  L0C[C_L0] --T.copy(enable_relu=True)--> GM[qk_workspace]  (Cube 侧, AUTO_CV_SYNC 自动 cross_flag, ring-buffer slot)
  GM[qk_workspace] --T.copy(切片)--> UB[qk_ub]               (Vector 侧, v5: [groupInner=16, BLOCK_N] 切片, AUTO_CV_SYNC 自动 wait_cross_flag)
  GM[Weights] --T.copy(strided DMA)--> UB[weight_ub]         (v5: TND 直接消费, [groupInner=16] 切片)
  UB[qk_ub] --T.tile.row_expand_mul_experiment(qk_ub, weight_ub)--> UB[qk_ub]  (v6: 融合 Brcb+Mul, 替代 broadcast+mul, 无需 weight_2d)
  UB[qk_ub] --T.reduce_sum(dim=0, clear=False)--> UB[scores_accum]  (v6: 融合 reduce+累加, 替代 reduce+add, 无需 scores_partial)
  UB[scores_accum] --T.copy--> GM[Scores]                     (部分分数写入, split-N)

Kernel 2 (Developer 纯 Vector):
  GM[Scores] --T.copy--> UB[scores_tile] --mask--> UB[scores_tile] --scatter--> UB[score_accum]
  UB[score_accum] --T.tile.topk--> UB[topk_dst]
  UB[topk_dst] --T.tile.gather_mask--> UB[topk_index]
  UB[topk_index] --T.tile.cast--> UB[output_ub]
  UB[output_ub] --T.copy--> GM[Output]
```

**合规性确认**：所有跨级搬运均通过 T.copy，无 GM→L0 直搬。L0C→UB 经 GM workspace 中转（Ascend 硬件限制：UB 与 L1 不能直通）。v4: TND strided DMA 使用 T.copy 原生支持（D 连续维 + N·D 跨步）。**v6: row_expand_mul_experiment 和 reduce_sum(clear=False) 均为 UB 内操作，不涉及跨级搬运。**

### 4.5 UB 内存预算

> **重要修正**：T.tile.topk 内部自动注入排序临时 buffer（`allocate_tmp_buffer.cc` pass），大小 = `aligned_count × multiplier` 字节。
> - float32 (calc_dtype): multiplier = **6**（bufA + bufB + bufC = 6×alignedCount）
> - half (fp16/bf16): multiplier = **16**（含 cast-to-float pool）
> - `aligned_count = ((max_actual_num + 31) // 32) * 32`，`max_actual_num = MAX_S2`（编译期常量）
>
> 前版 DESIGN.md 遗漏了此 buffer，UB 预算被低估。本版已修正。

#### Kernel 1 (per core, v6: Developer + groupInner=16 + L0 inner split + row_expand_mul + reduce_sum merge)

**v6 关键修正**：v5 的 broadcast+mul 两步 + reduce+add 两步，被 `row_expand_mul_experiment`（融合 Brcb+Mul）+ `reduce_sum(clear=False)`（融合 reduce+累加）替代。移除 weight_2d (32KB) 和 scores_partial (2KB)，UB 从 68KB 降到 34KB。

| Buffer | Shape | dtype | 大小 (Bytes) | v5 → v6 变化 |
|--------|-------|-------|-------------|-------------|
| **qk_ub** (v5 修正, v6 保留) | **[groupInner=16, BLOCK_N=512]** | float32 | **32768 (32KB)** | 不变 (v6: row_expand_mul 的 dst=src0 原地乘) |
| **weight_ub** (v5 修正, v6 保留) | **[groupInner=16]** | float32 | **64** | 不变 (v6: row_expand_mul 的 src1, 1D 行向量) |
| ~~weight_2d~~ | ~~[groupInner=16, BLOCK_N=512]~~ | ~~float32~~ | ~~32768 (32KB)~~ | **v6 移除**: row_expand_mul_experiment 内部 Brcb 广播, 不需显式 2D buffer |
| ~~scores_partial~~ | ~~[BLOCK_N=512]~~ | ~~float32~~ | ~~2048 (2KB)~~ | **v6 移除**: reduce_sum(clear=False) merge 语义直接累加 |
| **scores_accum** (v5 修正, v6 保留) | **[BLOCK_N=512]** | float32 | **2048 (2KB)** | 不变 (v6: reduce_sum(clear=False) 的 dst, 直接累加) |
| **总计 (同时活跃)** | | | **~34 KB** / 192KB (A2/A3) ✓ | v5: 68KB → v6: 34KB (-50%) |

> **v6 UB 容量验证**：34KB < 192KB ✓。UB buffer 在 pipeline body 外分配（不被 ring-buffer），num_stages 主要受 GM workspace ring-buffer slots 和 pipeline 有效性限制。
>
> **v6 num_stages 理论容量分析**（Q25 详述）：
> - UB buffer (34KB) 在 pipeline body 外分配 → 不被 ring-buffer → 不受 num_stages × UB 限制
> - GM workspace ring-buffer（[NUM_CORES, num_stages, BLOCK_M, BLOCK_N]）→ GM 无容量限制
> - num_stages=3: 3 slot GM workspace, pipeline 重叠更好 ✓
> - num_stages=4: 4 slot, 仍有收益（diminishing returns）✓
> - num_stages=5: 理论可行, 但 pipeline 收益递减, 需实测验证
> - **v6 建议**: num_stages=2 (保守基线) → Stage 2 实测后尝试 3-4
>
> **v6 groupInner=16 分块的工作原理**（Q20 详述, v5 保留）：
> - BLOCK_M = s1BaseSize × G（如 8×64=512），qk_workspace 仍是 [BLOCK_M, BLOCK_N]=[512,512]（GM 无容量限制）
> - Vector 侧每次读取 [groupInner=16, BLOCK_N=512] 切片，循环 ceil(BLOCK_M/groupInner)=32 次（s1BaseSize × ceil(G/groupInner)）
> - scores_accum 跨所有 groupInner 迭代累加，最终得到 [BLOCK_N] 的完整分数
> - **v6 差异**: 每次 iteration 从 v5 的 6 条指令 (copy qk + copy weight + broadcast + mul + reduce + add) 降为 4 条 (copy qk + copy weight + row_expand_mul + reduce_sum merge)
>
> **v4 的错误（v5 已修正）**：
> | N1 | G | BLOCK_M | BLOCK_N | v4 qk_ub | v4 weight_2d | v4 总 UB | UB 限? |
> |----|---|---------|---------|----------|-------------|---------|--------|
> | 64 | 64 | 512 | 512 | 1MB | 1MB | 2MB | ❌ |
> | 16 | 16 | 128 | 512 | 256KB | 256KB | 512KB | ❌ |
> | 8 | 8 | 64 | 512 | 128KB | 128KB | 256KB | ❌ |
>
> **v6 修正后**（所有 N1 场景均安全, UB 与 N1 无关）：
> | N1 | G | BLOCK_M | BLOCK_N | v6 qk_ub | v6 weight_ub | v6 总 UB | UB 限? |
> |----|---|---------|---------|----------|-------------|---------|--------|
> | 64 | 64 | 512 | 512 | 32KB | 64B | 34KB | ✓ |
> | 16 | 16 | 128 | 512 | 32KB | 64B | 34KB | ✓ |
> | 8 | 8 | 64 | 512 | 32KB | 64B | 34KB | ✓ |

#### Kernel 2 单次 topk 路径 (S2 ≤ MAX_S2_UB=16384, float32 calc_dtype)

| Buffer | Shape | 大小 (MAX_S2=8192) | 大小 (MAX_S2=16384) | 生命周期 |
|--------|-------|-------------------|---------------------|---------|
| score_accum | [MAX_S2] | 32 KB | 64 KB | scatter 写 + topk 读 |
| **topk_sort_tmp** (注入) | [MAX_S2 × 6] | **48 KB** | **96 KB** | topk 执行期间 |
| scores_tile / col_idx / col_idx_f / mask_ub | ~[BLOCK_N] | ~2 KB | ~2 KB | scatter 阶段（topk 前死亡，可与 sort_tmp 复用） |
| topk_dst | [2×SPARSE_COUNT] | 16 KB | 16 KB | topk 后（可复用 score_accum 空间） |
| topk_index | [SPARSE_COUNT] | 8 KB | 8 KB | gather_mask 后 |
| output_ub | [SPARSE_COUNT] | 8 KB | 8 KB | cast 后 |

**MEMORY_PLANNING 地址复用后的峰值 UB**：

| MAX_S2 | score_accum + sort_tmp (同时活跃) | 后处理 (复用 score_accum) | 峰值 | 192KB? |
|--------|----------------------------------|--------------------------|------|--------|
| 8192 | 32 + 48 = 80 KB | 32 KB | **80 KB** | ✓ |
| 12288 | 48 + 72 = 120 KB | 32 KB | **120 KB** | ✓ |
| 16384 | 64 + 96 = 160 KB | 32 KB | **160 KB** | ✓ |
| 20480 | 80 + 120 = 200 KB | — | **200 KB** | ✗ 超限 |

> **结论**：MAX_S2 ≤ 16384 时单次 topk 路径可行（峰值 160KB < 192KB）。MAX_S2 > 16384 必须使用分段 topk 路径。

#### Kernel 2 分段 topk 路径 (S2 > 16384, float32 calc_dtype, SEGMENT_SIZE=8192)

**Phase 1: mask + scatter 到 GM score_accum_gm**

| Buffer | 大小 | 说明 |
|--------|------|------|
| scores_tile / col_idx / col_idx_f / mask_ub | ~2 KB | mask 应用，scatter 到 GM |

**Phase 2: 逐段 topk + merge_sort**

| Buffer | 大小 | 生命周期 |
|--------|------|---------|
| segment_buf [SEGMENT_SIZE=8192] | 32 KB | 段 topk 期间 |
| segment_sort_tmp (注入) [8192×6] | 48 KB | 段 topk 期间 |
| segment_topk_dst [2×SPARSE_COUNT] | 16 KB | 段 topk 后 → merge 期间 |
| running_topk [2×SPARSE_COUNT] | 16 KB | 全程累积 |
| merge_output [4×SPARSE_COUNT] | 32 KB | merge_sort 期间 |
| merge_sort_tmp (注入) | ~32 KB | merge_sort 期间 |
| topk_index / output_ub | 16 KB | 最终输出 |

**地址复用后峰值**（段 topk 与 merge 非同时）：
- 段 topk 阶段: segment_buf(32) + sort_tmp(48) + running_topk(16) = 96 KB
- merge 阶段: merge_output(32) + merge_sort_tmp(32) + running_topk(16) + segment_topk_dst(16) = 96 KB
- **峰值 = 96 KB < 192KB ✓**

> **S2=131072 分段数**：ceil(131072/8192) = 16 段，每段一次 topk + 一次 merge_sort（首段无 merge），共 16 次 topk + 15 次 merge_sort。

#### Kernel 1 isSparseCountOver2K 场景 UB 验证（v5 新增缺陷 #8, v6 更新）

**问题背景**：isSparseCountOver2K 分支下 virTopK 最大 8192，sortOutBuf 可能超限。v5 补充此场景的 UB 计算，v6 更新为 row_expand_mul + reduce_sum merge 后的值。

| 参数 | sparse_count=8192 (isSparseCountOver2K) | 说明 |
|------|---------------------------------------|------|
| s1BaseSize | 2 | 8192/8192*2 = 2 |
| s2BaseSize | S2 (整段，不分块) | s2BaseNum = 1 |
| G (典型) | 16 (N1=16) | BLOCK_M = 2 × 16 = 32 |
| groupInner | 16 | v5 固定值 |
| Vector 侧 UB (v6) | qk_ub(32KB) + weight_ub(64B) + scores_accum(2KB) = **34KB** | v6: 移除 weight_2d + scores_partial, 与非 Over2K 场景一致 ✓ |
| Vector 侧 UB (v5 对照) | qk_ub(32KB) + weight_2d(32KB) + scores(4KB) = **68KB** | v5 基线 (已作废) |
| Cube 侧 L0C | [M_L0=128, N_L0=128]×4 = **64KB** | L0 inner split ✓ (BLOCK_M=32 < M_L0=128, 无需 inner split M) |

> **isSparseCountOver2K 场景结论**：v6 row_expand_mul + reduce_sum merge 后，Vector 侧 UB 固定 34KB（与 sparse_count 无关），isSparseCountOver2K 场景 UB 安全。
>
> **注意**：sparse_count=8192 时 S2 整段处理（s2BaseSize=S2），score_accum = [S2]×4B 可能很大。但 score_accum 在 Kernel 2（topk）中处理，Kernel 2 用分段 topk 路径（SEGMENT_SIZE=8192）解决。Kernel 1 的 scores_accum 仅需 [BLOCK_N=s2BaseSize]×4B。当 s2BaseSize=S2 且 S2 很大时（如 131072），scores_accum = [131072]×4B = 512KB >> 192KB。
>
> **v5/v6 修正**：isSparseCountOver2K 场景下 s2BaseSize=S2（整段），Kernel 1 Vector 侧 scores_accum 不能放 [S2]。**解决方案**：Kernel 1 Vector 侧每个 s2_block（此时 s2BaseNum=1，即整段一次处理）的 scores_accum 仍按 [BLOCK_N=s2BaseSize] 分配，但当 s2BaseSize 过大时需进一步分块。实际实现中，isSparseCountOver2K 场景的 S2 通常 ≤ 8192（sparse_count=8192 时 S2 不会远超 8192），scores_accum = [8192]×4B = 32KB ✓。若 S2 > 192KB/4 = 49152，需在 Kernel 1 内进一步分段（Stage 2 验证）。

### 4.6 L0C / L1 容量验证

> **v5 关键修正**：v4 的 C_L0 = [BLOCK_M, BLOCK_N] 在 N1≥16 时超限（如 [512,512]×4=1MB>>128KB）。v5 引入 L0 inner split (M_L0=128, N_L0=128)，C_L0 固定 64KB。

**v5 L0C 容量验证（L0 inner split 后）**：

| 场景 | BLOCK_M | BLOCK_N | M_L0 | N_L0 | C_L0 (float32) | 128KB? | L0 inner split 迭代次数 |
|------|---------|---------|------|------|----------------|--------|------------------------|
| G=16, s1BaseSize=8, s2BaseSize=512 | 128 | 512 | 128 | 128 | 128×128×4 = 64KB | ✓ | M:1次, N:4次 |
| G=64, s1BaseSize=8, s2BaseSize=512 | 512 | 512 | 128 | 128 | 128×128×4 = 64KB | ✓ | M:4次, N:4次 |
| G=8, s1BaseSize=8, s2BaseSize=512 | 64 | 512 | 128 | 128 | 128×128×4 = 64KB | ✓ | M:1次, N:4次 (M不足128按实际) |
| G=16, block_size=768 | 128 | 768 | 128 | 128 | 64KB | ✓ | M:1次, N:6次 (768=128×6) |
| G=64, block_size=768 | 512 | 768 | 128 | 128 | 64KB | ✓ | M:4次, N:6次 |
| G=64, block_size=1024 | 512 | 1024 | 128 | 128 | 64KB | ✓ | M:4次, N:8次 |

> **v5 结论**：L0 inner split (M_L0=128, N_L0=128) 后，**所有场景 C_L0 = 64KB < 128KB ✓**。不再需要 v4 的 "inner split N" 条件判断（统一用 M_L0=128, N_L0=128）。

**v7.3 L0C 容量验证（4×L0C，参考算子模式 1 / HISA 行 156-159）**：

> **v7.3 关键更新（参考算子模式 1）**：v7.2 单 C_L0（64KB）在 4 子 GEMM 序列中，MMA 与 L0C→GM copy 共用同一 fragment，存在 FIX↔M flag 争用（HISA 验证）。v7.3 升级为 4×L0C，每个子 GEMM 独占一个 L0C fragment，消除争用。

| 场景 | 单 L0C (v7.2) | 4×L0C (v7.3) | L0C 上限 | 状态 |
|------|-------------|--------------|---------|------|
| M_L0=128, N_L0=128, calc_dtype=float32 | 128×128×4 = 64KB | 4 × 64KB = **256KB** | 512KB | ✓ (余量 256KB) |
| M_L0=128, N_L0=128, calc_dtype=float16 (fallback) | 128×128×2 = 32KB | 4 × 32KB = **128KB** | 512KB | ✓ (余量 384KB) |

> **v7.3 结论**：4×L0C 总占用 256KB < 512KB（Ascend910B3 单 AIC L0C 上限）✓。每个子 GEMM 独占一个 L0C fragment，消除 FIX↔M flag 争用。参考 `examples/HISA/paged_block_sparse_mqa_attn_expert.py:156-159` 的 4×L0C 模式。
>
> **Stage 2 验证项**：编译后检查 4×L0C fragment 是否被合并/共享（get_kernel_source），确认每个 fragment 独立。

**v5 L1 容量验证**：

| Buffer | Shape | dtype | 大小 | 上限 | 状态 |
|--------|-------|-------|------|------|------|
| Q_L1 (BLOCK_M=512, D=128) | [512, 128] | bfloat16 | 128 KB | 512 KB | ✓ |
| K_L1 (BLOCK_N=512, D=128) | [512, 128] | bfloat16 | 128 KB | 512 KB | ✓ |
| Q_L1 + K_L1 总计 | | | 256 KB | 512 KB | ✓ (pipeline body 外分配, 不 ×num_stages) |
| Q_L1 (BLOCK_M=128, D=128) | [128, 128] | bfloat16 | 32 KB | 512 KB | ✓ |
| K_L1 (BLOCK_N=128, D=128) | [128, 128] | bfloat16 | 32 KB | 512 KB | ✓ |

> **T.Pipelined num_stages 对 L1 的影响**：
> - v5 策略：Q_L1/K_L1 在 pipeline body **外**分配（参考 `matmul_add_pipeline.py`），编译器对 body 外 buffer 不做 ring-buffer
> - 因此 L1 占用 = Q_L1 + K_L1 = 256KB（不 ×num_stages）< 512KB ✓
> - **若编译器仍 ring-buffer**（最坏情况）：256KB × 2 = 512KB = 512KB L1（刚好满，风险）。此时需 num_stages=1 或减小 BLOCK_M
> - **Stage 2 验证项**：编译后检查 L1 buffer 是否被 ring-buffer（get_kernel_source）

**v4 的错误（已修正）**：

| 场景 | v4 C_L0 shape | v4 C_L0 大小 | 128KB? | v5 C_L0 shape | v5 C_L0 大小 | 128KB? |
|------|--------------|-------------|--------|--------------|-------------|--------|
| G=64, s2BaseSize=512 | [512, 512] | 1MB | ❌ | [128, 128] | 64KB | ✓ |
| G=16, s2BaseSize=512 | [128, 512] | 256KB | ❌ | [128, 128] | 64KB | ✓ |
| G=64, block_size=768 | [512, 768] | 1.5MB | ❌ | [128, 128] | 64KB | ✓ |

**block_size=768 支持确认**（Q9 回答，v5 更新）：
- T.gemm_v0 源码（`ascend.py:343-448`）仅要求 `kL0Size % 16 == 0` 且 `kL0Size <= 4095`，**无 block_size (N 维) 必须为 2 的幂次约束**
- 768 = 48 × 16，满足 16 对齐要求 ✓
- v5 L0 inner split (N_L0=128) 后，768 = 128 × 6，N 维 6 次迭代 ✓
- **结论：block_size=768 可用**，v5 统一用 L0 inner split 处理

### 4.7 动态轴定义

| 动态轴 | 声明方式 | 运行时范围 | 测试用例最大值 | 说明 |
|--------|----------|-----------|-------------|------|
| B (batch_size) | 编译期参数 | 1 ~ 256 | 56 | — |
| S1 (query seq len) | 编译期参数 | 1 ~ **8192** | 8192 | 扩展（前版 4096 不足） |
| S2 (key seq len) | 编译期 MAX_S2 + 运行时 act_k | 1 ~ **131072** | 131072 | 扩展（前版 8192 远不足） |
| act_q / act_k | 运行时 tensor (int32) | 0 ~ S1 / 0 ~ S2 | — | BSND: 非前缀和; TND: 前缀和 |
| sparse_count (k) | 编译期参数 | **[1, 2048] ∪ {3072, 4096, 5120, 6144, 7168, 8192}** | 8192 | **v3 扩展**：支持 >2048 的离散值。aclnn 文档确认完整范围。>2048 时 tiling 策略调整（§5.2 isSparseCountOver2K 分支） |
| block_size | 编译期参数 | 16 ~ 1024 (16 整数倍) | 1024 | 768/1024 均支持 |
| N1 (q_head_num) | 编译期参数 | 8 / 16 / 24 / 32 / 64 | 64 | G = N1/N2 = N1 |

> **S1=8192 并行度评估**：Kernel 1 并行度 = B × S1 × S2_blocks。最坏情况 B=1, S1=8192, S2=8192, BLOCK_N=128 → 1×8192×64 = 524288 blocks → 21845 waves (24 cores) → 计算密集，充分利用核数。Kernel 2 并行度 = B × S1 = 1×8192 = 8192 blocks → 342 waves → 足够。

### 4.8 JIT 配置（v4 更新）

```python
# Kernel 1: v4 Developer + CV 融合 + T.Pipelined
@tilelang.jit(
    out_idx=[7],           # Scores 为输出
    workspace_idx=[6],     # qk_workspace 为 C→V 中转 (v4: [NUM_CORES, num_stages, ...])
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,      # 自动 CV 分割
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,         # 自动跨核同步 (配合 T.Pipelined)
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,            # 自动 Vector 内同步
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,      # UB 地址复用
        tilelang.PassConfigKey.TL_ASCEND_PTO_USE_PIPE_IN_CV_COPY: False,  # CV copy 不用 PTO
    },
)

# Kernel 2: Developer 纯 Vector
@tilelang.jit(
    out_idx=[-1],          # Output 为输出
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
```

### 4.9 TND Layout 处理方案（v4 重写：kernel 直接消费 TND，消除 host wrapper 转置）

#### 4.9.1 TND 与 BSND 的关键差异

| 维度 | BSND | TND |
|------|------|-----|
| Query shape | [B, S1, N1, D] | [T1, N1, D]（T1 = sum(act_seq_q_per_batch)） |
| Key shape | [B, S2, N2, D] | [T2, N2, D]（T2 = sum(act_seq_k_per_batch)） |
| Weights shape | [B, S1, N1] | [T1, N1] |
| act_seq_q | 非前缀和：每元素为该 batch 有效 token 数 | **前缀和**：act_seq_q[i] = sum(act_seq_q[0..i])，非递减 |
| act_seq_k | 非前缀和 | **前缀和** |
| B 维度 | 显式存在 | 隐式（通过 act_seq 长度推导 B = len(act_seq)） |
| 输出 shape | [B, S1, N2, sparse_count] | [T1, N2, sparse_count] |

#### 4.9.2 方案选型：kernel 直接消费 TND（v4 选定，替代 v3 的 host wrapper 转换）

**v4 选定方案**：kernel 直接按 TND stride 索引，不做物理转置。host wrapper 仅传递 act_seq 前缀和，不做 TND→BSND 转换。

**v4 选型理由（替代 v3 host wrapper 方案的原因）**：
1. **性能**：msprof 实测 TND 场景（9/37=24% 用例）的 host wrapper 转置 kernel 额外开销可能占 15-30%（`flash_attn_optimize.md` §9(a) 实测数据）
2. **T.copy 原生支持 strided DMA**：TND 的 `[T, N, D]` 布局中，D 仍是最内层连续维，`T.copy(Query[t_start:t_start+s1BaseSize, :, :], Q_L1)` 是 D-连续、按 N·D 跨步的 strided DMA
3. **已有先例**：`examples/deepseek_v4/sparse_attention.py` 中 `q_shape=[b,m,h,d]`，`T.copy(q[by,bx,:,:], q_l1)` 即 strided DMA（b/m 为 symbolic 变量）
4. **前缀和读取无性能影响**：act_seq 前缀和作为 GM tensor，kernel 内读取 1 个 int32 标量/任务，相比 GEMM 计算可忽略

**v3 host wrapper 方案不选的原因**：
- 额外的 Transpose/Copy kernel 纯搬运不产生有效计算，小 shape 下转置耗时可达 main_kernel 的数倍
- TND→BSND 转换需要额外 GM 内存（B × S1_max × N1 × D），内存浪费

#### 4.9.3 kernel 直接消费 TND 的实现方案

**关键约束分析（Q18 回答）**：
- **"循环边界不能依赖 tensor 值"约束**：此约束针对 **循环次数**（`for i in T.serial(act_q)` 禁止），不针对**数据访问索引**（`T.copy(Query[t_start, :, :], Q_L1)` 允许，t_start 为标量读取）
- **act_seq 前缀和传递**：act_seq_q/act_seq_k 是 GM tensor，kernel 内通过 `act_seq_q[b]` 读取为标量（v3 已有此模式：`act_q = actual_seq_q[b]`）
- **动态切片长度问题**：TND 每个 batch 的 token 数不同（act_q 动态），但 `T.copy` 要求 dst buffer 静态 shape。解法：**切片长度固定（s1BaseSize），切片起始动态（t_start + offset）**

**kernel 内 TND 数据访问模式**：

```python
# TND Query 访问（D 连续，N·D 跨步）
if layout_query == "TND":
    # 读取前缀和标量
    t_start = T.if_then_else(b == 0, 0, actual_seq_q[b - 1])
    act_q_batch = actual_seq_q[b] - t_start  # per-batch token count
    # 切片起始动态（t_start + s1_tile * s1BaseSize），长度固定（s1BaseSize × N1 × D）
    q_offset = t_start + s1_tile * s1BaseSize
    T.copy(Query[q_offset : q_offset + s1BaseSize, :, :], Q_L1)
else:  # BSND
    act_q_batch = actual_seq_q[b] if actual_seq_q is not None else S1
    T.copy(Query[b, s1_tile * s1BaseSize : (s1_tile + 1) * s1BaseSize, :, :], Q_L1)

# TND Key 访问（同理）
if layout_key == "TND":
    t_start_k = T.if_then_else(b == 0, 0, actual_seq_k[b - 1])
    act_k_batch = actual_seq_k[b] - t_start_k
    k_offset = t_start_k + s2_block * BLOCK_N
    T.copy(Key[k_offset + vid * BLOCK_N//2 : k_offset + (vid+1) * BLOCK_N//2, 0, :], K_L1)
elif layout_key == "PA_BSND":
    k_block_id = block_table[b, s2_block]
    T.copy(Key[k_block_id, :, 0, :], K_L1)
else:  # BSND
    T.copy(Key[b, s2_block * BLOCK_N : (s2_block + 1) * BLOCK_N, 0, :], K_L1)
```

**关键设计决策**：
1. **切片长度固定为 s1BaseSize**：dst buffer Q_L1 shape = [s1BaseSize, N1, D]（编译期常量），src 切片长度匹配
2. **切片起始动态**：`q_offset = t_start + s1_tile * s1BaseSize`，t_start 从 act_seq 前缀和读取
3. **尾块越界保护**：最后一个 batch 的最后一个 s1_tile 可能越过 T1 边界 → host 侧需确保 T1 是 s1BaseSize 的整数倍（padding 0），或 kernel 内用 `if s1_tile * s1BaseSize < act_q_batch:` 条件跳过
4. **跨 batch 读数据问题**：TND 紧凑排列，切片可能读到下一个 batch 的数据。但 `if s1_local < act_q_batch:` 条件保证只处理有效 token，无效数据被 GEMM 计算后被 mask 丢弃

> **⚠️ v5 风险标注（缺陷 #6）**：kernel 内读 `act_seq_q[b-1]`（前缀和标量读取）是 GM tensor 标量读取，TileLang 是否支持此模式**未在 examples 中找到先例**，需 Stage 2 实际验证。
>
> **Stage 2 验证方案**：
> 1. 尝试 `T.copy(act_seq_q[b-1:b], scalar_ub)` 或直接 `act_seq_q[b-1]` 标量读取
> 2. 若不可用，**fallback 方案**：host wrapper 预处理——将前缀和转为 per-batch 偏移量（`t_start_q[b]` 数组）传入 kernel，kernel 内 `t_start = t_start_q[b]` 直接索引
> 3. fallback 方案的 host wrapper 代码：
>    ```python
>    # host wrapper: 前缀和 → per-batch 偏移
>    t_start_q = torch.zeros(B, dtype=torch.int32)
>    t_start_q[1:] = actual_seq_q[:-1]  # t_start_q[b] = actual_seq_q[b-1] (b>0)
>    # 传入 kernel: t_start_q 作为额外输入 tensor
>    ```

#### 4.9.4 host wrapper 简化（v4）

```python
def lightning_indexer_wrapper(query, key, weights, actual_seq_q, actual_seq_k,
                               block_table, layout_query, layout_key, **kwargs):
    """v4 host wrapper: 直接传 TND 张量，不做转置"""
    # v4: 不再做 TND→BSND 转换，直接传原张量
    # kernel 根据 layout_query / layout_key 参数选择访问模式
    output = lightning_indexer_kernel(
        query, key, weights, actual_seq_q, actual_seq_k, block_table,
        layout_query=layout_query, layout_key=layout_key, **kwargs)
    return output
```

**v4 host wrapper vs v3 host wrapper 对比**：

| 维度 | v3 host wrapper | v4 host wrapper |
|------|----------------|-----------------|
| TND→BSND 转换 | ✅ 需要（额外 kernel） | ❌ 不需要 |
| 前缀和→per-batch 转换 | ✅ 需要 | ❌ 不需要（kernel 直接读前缀和） |
| 额外 GM 内存 | B × S1_max × N1 × D | 无 |
| 额外 kernel launch | 1-2 个 Transpose/Copy kernel | 0 |
| kernel 复杂度 | 简单（仅 BSND） | 中等（BSND + TND 双路径） |

#### 4.9.5 TND 各场景处理方式

| 场景 | query 处理 | key 处理 | act_seq | 输出 |
|------|-----------|---------|---------|------|
| BSND_BSND | kernel 直接 BSND 索引 | kernel 直接 BSND 索引 | 非前缀和 | [B, S1, N2, sparse_count] |
| BSND_PA_BSND | kernel 直接 BSND 索引 | kernel 间接 block_table 寻址 | 非前缀和 | [B, S1, N2, sparse_count] |
| TND_PA_BSND | kernel 直接 TND strided DMA | kernel 间接 block_table 寻址 | query 前缀和, key 非前缀和 | [T1, N2, sparse_count] |
| TND_TND | kernel 直接 TND strided DMA | kernel 直接 TND strided DMA | 均前缀和 | [T1, N2, sparse_count] |

> **TND 输出格式**：TND layout 的输出为 `[T1, N2, sparse_count]`，kernel 直接按 TND stride 写入。host wrapper 无需做 BSND→TND 输出转换。
>
> **TND_PA_BSND 混合场景**：query 用 TND strided DMA 访问，key 用 PA_BSND block_table 间接寻址。act_seq_q 为前缀和，act_seq_k 为非前缀和（PA_BSND 的 block_table 已按 per-batch 构建）。

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 混合（CV 融合）

**判定依据**: Kernel 1 含 GEMM (Cube) + weight mul + reduce (Vector)，为 CV 融合算子；Kernel 2 纯 Vector (mask + topk)。

### 5.2 Block 划分

#### Kernel 1 (score computation, v6: Fixed Core + T.Pipelined + groupInner=16 + L0 inner split + row_expand_mul + reduce_sum merge)

**v6 关键调整**：
1. **Fixed Core 模式**：`launch_core_num = min(total_tasks, NUM_CORES)` 替代固定 `T.Kernel(NUM_CORES=24)`（v5 缺陷 #5 修正：小任务场景避免空转）
2. **v6 动态核数**：`NUM_CORES = torch.npu.get_device_properties("npu").cube_core_num`（替代 v5 硬编码 24，适配 A2/A3/950）
3. **T.Pipelined 核间流水**：S2 循环用 `T.Pipelined(num_stages=2, cross_interval=2)` 表达 CV overlap
4. `BLOCK_M = s1BaseSize × G`（v3 联合切分保留）
5. **v5 新增 groupInner=16**（Vector 侧 G 维内层分块）
6. **v5 新增 L0 inner split**（Cube 侧 M_L0=128, N_L0=128）
7. **v6 新增 row_expand_mul_experiment**（融合 Brcb+Mul，替代 broadcast+mul）
8. **v6 新增 reduce_sum(clear=False)**（融合 reduce+累加，替代 reduce+add）
9. `isSparseCountOver2K` 分支（v3 保留）

```python
# v6: 动态核数 (优化 #3, 替代 v5 硬编码 24)
# host 侧获取 (参考 examples/xattention/xattention.py:24)
NUM_CORES = int(torch.npu.get_device_properties("npu").cube_core_num)  # A2=24, A3=20, 950=...

# v3 保留: isSparseCountOver2K 分支
SPARSE_COUNT_8K = 8192
BASE_TOPK = 2048

if sparse_count <= BASE_TOPK:
    isSparseCountOver2K = False
    s1BaseSize = 8
    s2BaseSize = 512
    s2BaseNum = T.ceildiv(S2, s2BaseSize)
else:
    isSparseCountOver2K = True
    s1BaseSize = SPARSE_COUNT_8K // sparse_count * 2
    s2BaseSize = S2
    s2BaseNum = 1 if S2 > 0 else 0

# GEMM 维度: M = s1BaseSize × G (v3 联合切分), N = s2BaseSize, K = D
BLOCK_M = s1BaseSize * G
BLOCK_M_ALIGN = ((BLOCK_M + 15) // 16) * 16
BLOCK_N = s2BaseSize
BLOCK_K = D  # = 128

# v5 新增: Vector 侧 groupInner 分块
groupInner = 16  # AscendC arch22 lightning_indexer_service_vector.h:198
outerG = T.ceildiv(G, groupInner)  # G=64→4, G=16→1

# v5 新增: Cube 侧 L0 inner split (AscendC arch22 lightning_indexer_service_cube.h:54-65)
M_L0 = 128  # L0 级 GEMM M 维块大小
N_L0 = 128  # L0 级 GEMM N 维块大小
# C_L0 = [M_L0, N_L0] × 4B = 64KB < 128KB ✓

# v6: Fixed Core 任务分配 (动态核数)
S1_tiles = T.ceildiv(S1, s1BaseSize)
total_tasks = B * S1_tiles  # 每个 task = 一个 (b, s1_tile)
launch_core_num = T.min(total_tasks, NUM_CORES)  # v6: 动态核数, 小任务场景不空转
tasks_per_core = T.ceildiv(total_tasks, launch_core_num)

# v4 保留: workspace 按 NUM_CORES × num_stages 分配
num_stages = 2  # T.Pipelined 流水深度 (v6: UB=34KB 有余量, Stage 2 尝试 3-4; v7: 尝试 3-4)
# qk_workspace: [NUM_CORES, num_stages, BLOCK_M, BLOCK_N] calc_dtype
# 每 core 独立管理自己的 workspace slot, L2 cache 友好
```

> **v7.4 边界 Case tiling 修正（Phase 3 验证, 见 §12.12）**：
>
> **边界 Case 1: block_size < 64 超时修复**
> - **问题**: PA_BSND 场景 block_size=16 时, `BLOCKS_PER_TILE = BLOCK_N / block_size = 256/16 = 16`, 每个 tile 需 16 次 PA gather, 寻址开销过大 → 超时
> - **AscendC 对比**: `S2_BASE_SIZE=512` 固定, PA gather 在 Vector 侧逐 block 处理, 不受 BLOCK_N 影响
> - **v7.4 修复**: PA 场景且 `block_size < 64` 时减小 BLOCK_N, 降低 BLOCKS_PER_TILE
>   ```python
>   is_pa = (layout_key == "PA_BSND")
>   if is_pa and block_size < 64:
>       BLOCK_N = min(BLOCK_N, 128)  # BLOCKS_PER_TILE = 128/16 = 8（原 16 → 8, 减半）
>   ```
> - **影响**: 仅 PA + 小 block_size 场景（如 TND_PA_BSND bs=16）, 其他场景不变
>
> **边界 Case 3: Over2K (sparse_count > 2048) UB 超限修复**
> - **问题**: sparse_count=8192 时, `topk_a_ub` 按 `virTopK=8192` 分配 → UB=586KB 严重超限（192KB 上限）
> - **AscendC 处理**（`kernel.h:183-185`）:
>   ```cpp
>   isSparseCountOver2K = (sparseCount <= BASE_TOPK) ? false : true;
>   s1BaseSize = isOver2K ? SPARSE_COUNT_8K / sparseCount * 2 : 8;  // 缩小 S1_BLOCK
>   virTopK = isOver2K ? sparseCount : 2048;                        // virTopK = sparse_count
>   ```
> - **v7.4 修复**: Over2K 时走标准路径（不缓存 sort）+ 缩小 S1_BLOCK + virTopK=sparse_count
>   ```python
>   is_over2k = sparse_count > BASE_TOPK  # BASE_TOPK=2048
>   if is_over2k:
>       S1_BLOCK = max(2, SPARSE_COUNT_8K // sparse_count * 2)  # 8192/8192*2=2, 8192/4096*2=4
>       virTopK = sparse_count                                   # 最大 8192
>       # 走标准路径（actS1Size>4 或 isSparseCountOver2K, 不启用 sort 缓存, 见 §8.8 适用条件）
>       # topk_a_ub 按 virTopK 分配, 但 S1_BLOCK 缩小使总 UB 可控
>   else:
>       S1_BLOCK = 8 if S1 >= 8 else 4
>       virTopK = 2048
>   ```
> - **UB 预算**: Over2K 时 topk_a_ub = virTopK × 2 × 4B = 8192×2×4 = 64KB（原 586KB 的主因是 S1_BLOCK 未缩小导致多行累积）; S1_BLOCK=2 使单核同时处理的 S1 行数减半, UB 回到可控范围. ⚠️ Stage 2 实测确认实际 UB 占用.
>
> **边界 Case 4: TOP_K > S2 处理**
> - **问题**: sparse_count=2048 > S2=128（或 S2=512）时, 需从不足 sparse_count 个元素中选 sparse_count 个
> - **v7.4 修复**: 由 §8.8 步骤 2.5 的 -inf 填充自然处理 — 无效位置 score=-inf 经 sort 沉底, topk 后输出区由 `T.tile.select(..., -1.0, "VSEL_TENSOR_SCALAR_MODE")` 填 -1. 需确认 `topk_a_ub` 初始化为 -inf.

#### §5.2.1 v7 Cube GEMM tiling 对齐 AscendC（msprof 数据驱动）

> **v7 新增**：基于 msprof op 实测数据，BSND_BSND 用例 Cube 时间 98.83us vs AscendC 60.14us（慢 64%），**Cube GEMM tiling 差异是主要瓶颈**。本节对齐 AscendC arch22 的 3 层 tiling 策略。

**AscendC arch22 Cube tiling（源码验证 `service_cube.h:54-65`）**：

```cpp
// L1 层（MTE2→MTE1 搬运粒度）
M_BASIC_BLOCK = 256        // L1 层 M 维（S1*G），固定 256
S2_BASIC_BLOCK = 256       // L1 层 S2 维，固定 256
D_BASIC_BLOCK = 128        // K 维（head_dim），固定 128

// L0 层（MTE1→M MMA 粒度）
M_BASIC_BLOCK_L0 = 128     // L0 层 M
S2_BASIC_BLOCK_L0 = 128    // L0 层 S2
D_BASIC_BLOCK_L0 = 128     // L0 层 K

// Buffer 数量
KEY_BUF_NUM = 3             // K L1 三缓冲
QUERY_BUF_NUM = 2           // Q L1 双缓冲
L0_BUF_NUM = 2              // L0A/L0B/L0C 双缓冲

// Buffer 大小
QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK = 256 × 128 = 32K elements
KEY_BUFFER_OFFSET = S2_BASIC_BLOCK * D_BASIC_BLOCK = 256 × 128 = 32K elements
L0AB_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 = 128 × 128 = 16K elements
L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 = 128 × 128 = 16K elements
```

**v6 vs v7 Cube tiling 对比**：

| 参数 | v6（当前） | v7（对齐 AscendC） | 变化原因 |
|------|-----------|-------------------|---------|
| M_L1（L1 层 M） | S1_BLOCK×G=8×64=512 | **min(S1_BLOCK×G, 256)=256** | 对齐 AscendC M_BASIC_BLOCK=256，减小 L1 buffer |
| S2_L1（L1 层 S2） | s2BaseSize=512 | **256** | 对齐 AscendC S2_BASIC_BLOCK=256，减小 L1 buffer |
| M_L0（L0 层 M） | 128 | 128（不变） | 已对齐 |
| N_L0（L0 层 S2） | 128 | 128（不变） | 已对齐 |
| Q_L1 buffer | 2 bufs × [512,128]×2B = 256KB | **v7.1: 2 bufs × [128,128]×2B = 64KB**（独立 buffer, 无切片） | -75% L1 占用 |
| K_L1 buffer | 3 bufs × [512,128]×2B = 384KB | **v7.1: 2 bufs × [128,128]×2B = 64KB**（独立 buffer, 无切片） | -83% L1 占用 |
| L1 总占用 | 256KB + 384KB = 640KB > 512KB ⚠️ | **v7.1: 64KB + 64KB = 128KB < 512KB ✓** | v6 已超 L1 容量! v7.1 独立 buffer 修正 |
| S1 外层循环 | 1 次（M_L1=512 覆盖全部） | **2 次**（M_L1=256，分 2 轮） | 增加循环次数但提升流水效率 |

> **⚠️ v6 L1 容量隐患**：v6 的 Q_L1(256KB) + K_L1(384KB) = 640KB > 512KB L1 容量！这是 v6 Cube 慢的潜在原因之一——L1 超限导致数据被驱逐到 GM，增加 MTE2 延迟。
>
> **⚠️ v7.1 关键修正（API 验证 3 发现）**：v7 原设计的 "L1 行切片传 `T.gemm_v0(transpose_B=True)`" **不可行**——`transpose_B=True` + L1 行切片导致精度错误（diff=66.5，验证脚本 `test_3layer_tiling.py`）。**v7.1 改用独立 L1 buffer 方案**：为每个 (M_L0, N_L0) 子 tile 分配独立的 [128,128] L1 buffer，从 GM 直接载入，避免切片 + transpose_B 冲突。独立 buffer 方案精度正确（diff=0.0156）。L1 总占用 128KB < 512KB ✓。详见 §12.8 API 验证结果。

**v7.1 Cube 3 层 tiling 伪代码（独立 L1 buffer 方案, API 验证 3 修正）**：

> **v7.1 关键修正**：v7 原用 `T.gemm_v0(Q_L1[m_slice, :], K_L1[s2_slice, :], transpose_B=True)`（L1 行切片），验证发现 `transpose_B=True` + L1 行切片 → 精度错误（diff=66.5）。**v7.1 改用独立 L1 buffer**：每个 (M_L0, N_L0) 子 tile 分配独立的 [128,128] L1 buffer，从 GM 直接载入，无切片。

```python
# v7.1: Cube 3 层 tiling (L1 → L0 → MMA), 独立 L1 buffer 方案 (API 验证 3 修正)
# 对齐 AscendC service_cube.h:148-204, 但 L1 buffer 策略改为独立 buffer (非切片)
M_L1 = 256        # v7: L1 层 M 维 (对齐 AscendC M_BASIC_BLOCK)
S2_L1 = 256       # v7: L1 层 S2 维 (对齐 AscendC S2_BASIC_BLOCK)
M_L0 = 128        # v7: L0 层 M 维 (不变)
N_L0 = 128        # v7: L0 层 S2 维 (不变)
D = 128           # head_dim

# v7.1: 独立 L1 buffer (每个 [M_L0, D] 或 [N_L0, D], 避免切片 + transpose_B 冲突)
# M_L1=256 = 2×M_L0=128, S2_L1=256 = 2×N_L0=128 → 每个 L1 tile 拆为 2×2=4 个 L0 sub-tile
A_L1_0 = T.alloc_shared((M_L0, D), input_dtype)   # [128, 128] = 32KB (fp16), A 行块 0
A_L1_1 = T.alloc_shared((M_L0, D), input_dtype)   # [128, 128] = 32KB (fp16), A 行块 1
B_L1_0 = T.alloc_shared((N_L0, D), input_dtype)   # [128, 128] = 32KB (fp16), B 行块 0
B_L1_1 = T.alloc_shared((N_L0, D), input_dtype)   # [128, 128] = 32KB (fp16), B 行块 1
C_L0 = T.alloc_fragment((M_L0, N_L0), calc_dtype)  # [128, 128] = 64KB (fp32)

# v7.1: 2 层循环 (L1 层 s2_l1 → m_l1), L0 层 2×2 sub-tile 显式展开 (无内层循环)
for s2_l1 in T.serial(T.ceildiv(s2ProcessSize, S2_L1)):   # L1 层 S2 循环 (AscendC: s2GmOffset)
    s2_l1_real = T.min(S2_L1, s2ProcessSize - s2_l1 * S2_L1)
    # B_L1: GM → 独立 L1 buffer (2 个 N_L0 行块, 无切片)
    T.copy(Key[s2_l1 * S2_L1 : s2_l1 * S2_L1 + N_L0, ...], B_L1_0)
    T.copy(Key[s2_l1 * S2_L1 + N_L0 : s2_l1 * S2_L1 + 2 * N_L0, ...], B_L1_1)

    for m_l1 in T.serial(T.ceildiv(s1gProcessSize, M_L1)):  # L1 层 M 循环 (AscendC: s1gGmOffset)
        m_l1_real = T.min(M_L1, s1gProcessSize - m_l1 * M_L1)
        # A_L1: GM → 独立 L1 buffer (2 个 M_L0 行块, 无切片)
        T.copy(Query[m_l1 * M_L1 : m_l1 * M_L1 + M_L0, ...], A_L1_0)
        T.copy(Query[m_l1 * M_L1 + M_L0 : m_l1 * M_L1 + 2 * M_L0, ...], A_L1_1)

        # v7.1: 4 个子 GEMM (每个用独立 buffer, 无 L1 切片, transpose_B=True 精度正确)
        # (m0, n0): A_L1_0 × B_L1_0^T → C_L0
        T.gemm_v0(A_L1_0, B_L1_0, C_L0, transpose_B=True, init=True)
        T.copy(C_L0,
               qk_workspace[cid, s2_block % num_stages,
                            m_l1 * M_L1 : m_l1 * M_L1 + M_L0,
                            s2_l1 * S2_L1 : s2_l1 * S2_L1 + N_L0],
               enable_relu=True)
        # (m1, n0): A_L1_1 × B_L1_0^T → C_L0
        T.gemm_v0(A_L1_1, B_L1_0, C_L0, transpose_B=True, init=True)
        T.copy(C_L0,
               qk_workspace[cid, s2_block % num_stages,
                            m_l1 * M_L1 + M_L0 : m_l1 * M_L1 + 2 * M_L0,
                            s2_l1 * S2_L1 : s2_l1 * S2_L1 + N_L0],
               enable_relu=True)
        # (m0, n1): A_L1_0 × B_L1_1^T → C_L0
        T.gemm_v0(A_L1_0, B_L1_1, C_L0, transpose_B=True, init=True)
        T.copy(C_L0,
               qk_workspace[cid, s2_block % num_stages,
                            m_l1 * M_L1 : m_l1 * M_L1 + M_L0,
                            s2_l1 * S2_L1 + N_L0 : s2_l1 * S2_L1 + 2 * N_L0],
               enable_relu=True)
        # (m1, n1): A_L1_1 × B_L1_1^T → C_L0
        T.gemm_v0(A_L1_1, B_L1_1, C_L0, transpose_B=True, init=True)
        T.copy(C_L0,
               qk_workspace[cid, s2_block % num_stages,
                            m_l1 * M_L1 + M_L0 : m_l1 * M_L1 + 2 * M_L0,
                            s2_l1 * S2_L1 + N_L0 : s2_l1 * S2_L1 + 2 * N_L0],
               enable_relu=True)
```

> **v7.1 尾块处理**：当 `s2_l1_real < S2_L1` 或 `m_l1_real < M_L1` 时，2×2 sub-tile 中部分 tile 无效。Stage 2 实现时用 `if` 条件保护无效 sub-tile 的 `T.copy` 和 `T.gemm_v0`（如 `if s2_l1_real > N_L0: T.copy(...B_L1_1...)`）。主路径（整除场景）不受影响。
>
> **v7.1 L1 占用核算**：A_L1_0 + A_L1_1 + B_L1_0 + B_L1_1 = 4 × 128×128 × 2B = **128KB < 512KB ✓**（v7 原方案 320KB, v7.1 进一步降至 128KB）

**v7 Cube tiling 预期收益**：

| 用例 | v6 Cube(us) | v7 预期 Cube(us) | 收益来源 |
|------|------------|-----------------|---------|
| BSND_BSND | 98.83 | ~65-70 | L1 容量修正(640KB→320KB) + tiling 对齐提升流水效率 |
| TND_TND | 42.80 | ~30-35 | 同上 |
| BSND_PA_BSND | 10.80 | ~10（保持） | 数据量小，tiling 影响有限 |

**v7.1 L1 容量验证（独立 buffer 方案, API 验证 3 修正）**：

| Buffer | v6 大小 | v7 原方案 | v7.1 独立 buffer | L1 容量 512KB |
|--------|---------|---------|-----------------|--------------|
| A_L1 (Query) | 2 × [512,128]×2B = 256KB | 2 × [256,128]×2B = 128KB | **2 × [128,128]×2B = 64KB** | ✓ |
| B_L1 (Key) | 3 × [512,128]×2B = 384KB | 3 × [256,128]×2B = 192KB | **2 × [128,128]×2B = 64KB** | ✓ |
| **总计** | **640KB > 512KB ❌** | 320KB < 512KB ✓ | **128KB < 512KB ✓** | v7.1 最优 |

> **v7.1 关键发现**：v7 原方案的 `transpose_B=True` + L1 行切片不可行（验证 3, diff=66.5）。v7.1 改用独立 L1 buffer 后，不仅修正了精度问题，还将 L1 占用从 320KB 进一步降至 128KB（-60%），为 T.Pipelined num_stages=3-4 流水留出更多 L1 空间。

**Q17 回答（v4 新增）：Fixed Core 模式下 workspace 分配**

| 维度 | v3 (按任务数) | v4 (Fixed Core) |
|------|-------------|-----------------|
| workspace shape | [total_blocks, BLOCK_M, BLOCK_N] | [NUM_CORES, num_stages, BLOCK_M, BLOCK_N] |
| workspace 大小 | total_blocks × BLOCK_M × BLOCK_N × 4B | 24 × 2 × BLOCK_M × BLOCK_N × 4B |
| 示例 (B=16,S1=5,S2=3072,G=16) | 16×1×6 × 128×128 × 4 = 6MB | 24 × 2 × 128×128 × 4 = 3MB |
| L2 cache 友好性 | 差（6MB > L2 cache 行） | 好（3MB < 192MB L2, 高命中率） |
| 内存膨胀 | 严重（随 B×S1×S2_blocks 线性增长） | 固定（仅 NUM_CORES × num_stages） |

**Fixed Core 任务循环伪代码（v6: groupInner + L0 inner split + row_expand_mul + reduce_sum merge）**：

```python
# v6 参数
groupInner = 16  # Vector 侧 G 维内层分块
M_L0 = 128       # Cube 侧 L0 M 维块大小
N_L0 = 128       # Cube 侧 L0 N 维块大小
NUM_CORES = int(torch.npu.get_device_properties("npu").cube_core_num)  # v6: 动态核数
launch_core_num = T.min(total_tasks, NUM_CORES)  # v5 缺陷 #5 修正

with T.Kernel(launch_core_num, threads=2, is_npu=True) as (cid, vid):
    # UB/L1/L0C buffer 在 pipeline body 外分配 (避免 ring-buffer, Q19)
    Q_L1 = T.alloc_shared((BLOCK_M, D), input_dtype)
    K_L1 = T.alloc_shared((BLOCK_N, D), input_dtype)
    C_L0 = T.alloc_fragment((M_L0, N_L0), calc_dtype)  # v5: [128,128] 非 [BLOCK_M,BLOCK_N]

    # Vector 侧 UB buffer (v6: 移除 weight_2d 和 scores_partial, pipeline body 外)
    qk_ub = T.alloc_shared((groupInner, BLOCK_N), calc_dtype)       # [16, 512] = 32KB
    weight_ub = T.alloc_shared((groupInner,), calc_dtype)            # [16] = 64B
    # v6 移除: weight_2d = T.alloc_shared((groupInner, BLOCK_N), ...)  # 已由 row_expand_mul 融合
    # v6 移除: scores_partial = T.alloc_shared((BLOCK_N,), ...)       # 已由 reduce_sum(clear=False) 融合
    scores_accum = T.alloc_shared((BLOCK_N,), calc_dtype)            # [512] = 2KB

    # v5: 每核多任务循环
    tasks_per_core = T.ceildiv(total_tasks, launch_core_num)
    for t in T.serial(tasks_per_core):
        task_id = cid * tasks_per_core + t
        if task_id < total_tasks:
            b = task_id // S1_tiles
            s1_tile = task_id % S1_tiles

            # v5: T.Pipelined 核间流水 (S2 循环)
            for s2_block in T.Pipelined(s2BaseNum, num_stages=2, cross_interval=2):
                s2_start = s2_block * BLOCK_N
                # --- AIC 侧 (编译器自动识别 T.gemm_v0) ---
                # v5: L0 inner split (Q21 详述)
                T.copy(Query[...], Q_L1)  # 加载完整 [BLOCK_M, D] 到 L1
                T.copy(Key[...], K_L1)    # 加载完整 [BLOCK_N, D] 到 L1

                # v5: L0 inner split GEMM 循环
                for m_l0 in T.serial(T.ceildiv(BLOCK_M, M_L0)):
                    for n_l0 in T.serial(T.ceildiv(BLOCK_N, N_L0)):
                        # T.gemm_v0 + BufferRegion 切片 (Q21 详述)
                        # M, N 从 C_L0 shape 推导 = [M_L0, N_L0] = [128, 128]
                        T.gemm_v0(
                            Q_L1[m_l0 * M_L0:(m_l0 + 1) * M_L0, :],  # L1 切片 [M_L0, D]
                            K_L1[n_l0 * N_L0:(n_l0 + 1) * N_L0, :],  # L1 切片 [N_L0, D]
                            C_L0,                                      # [M_L0, N_L0]
                            transpose_B=True,
                            init=(n_l0 == 0)                           # N 维首次 init, 后续累加
                        )
                    # C_L0 完成一整行 [M_L0, BLOCK_N], copy 到 workspace (enable_relu)
                    T.copy(C_L0, qk_workspace[cid, s2_block % num_stages,
                                               m_l0 * M_L0:(m_l0 + 1) * M_L0, :],
                           enable_relu=True)

                # --- AIV 侧 (编译器自动插 wait_cross_flag) ---
                # v5: groupInner=16 分块 (Q20 详述)
                T.tile.fill(scores_accum, 0)  # 初始化累加器

                for s1_inner in T.serial(s1BaseSize):
                    for g_idx in T.serial(outerG):
                        # 从 qk_workspace 读取 [groupInner, BLOCK_N] 切片
                        m_start = s1_inner * G + g_idx * groupInner
                        T.copy(qk_workspace[cid, s2_block % num_stages,
                                            m_start:m_start + groupInner, :], qk_ub)

                        # 读取权重 [groupInner]
                        T.copy(Weights[b, s1_tile * s1BaseSize + s1_inner,
                                       g_idx * groupInner:(g_idx + 1) * groupInner], weight_ub)

                        # v6: row_expand_mul_experiment 融合 Brcb+Mul (替代 v5 broadcast+mul)
                        # dst[i,j] = src0[i,j] * src1[i], src1=[groupInner] 行广播
                        T.tile.row_expand_mul_experiment(qk_ub, qk_ub, weight_ub)  # [16,512] = [16,512] * [16]

                        # v6: reduce_sum(clear=False) 融合 reduce+累加 (替代 v5 reduce+add)
                        # clear=False: new_out = old_out + reduced_result
                        T.reduce_sum(qk_ub, scores_accum, dim=0, clear=False)  # [16,512]→[512], 累加

                        # v6: Vector 管线轻量同步 (优化 #4, 替代部分 barrier_all)
                        T.pipe_barrier("v")

                T.copy(scores_accum, Scores[b, 0, s1_tile, s2_start:s2_start + BLOCK_N])
```

> **⚠️ v5 Stage 2 验证项**（v7.1 已验证, 见 §12.8 API 验证结果）：
> 1. ~~`T.gemm_v0(Q_L1[m_l0*M_L0:(m_l0+1)*M_L0, :], ...)` 的 2D L1 行切片是否被 T.gemm_v0 支持~~ → **v7.1 验证结论：`transpose_B=True` + L1 行切片 → 精度错误（diff=66.5），不可行！** 改用独立 L1 buffer 方案（§5.2.1 / §8.5 v7.1 伪代码）
> 2. ~~**fallback**：若 2D 行切片不可用，改为分配 `Q_L1 = T.alloc_shared((M_L0, D), input_dtype)`~~ → **v7.1 已采纳此 fallback 思路**：独立 buffer `A_L1_0/A_L1_1` 各 `[M_L0, D]`，L1 占用 128KB < 512KB ✓
> 3. `T.copy(C_L0, qk_workspace[..., m_slice, :], enable_relu=True)` 的 L0C→GM 部分写入是否可用 — **待 Stage 2 验证**（v7.1 伪代码仍用此模式）

**v3 保留内容**：

**Q6 回答（v3 更新）：M 轴 = G × s1BaseSize 联合切分**

| 方案 | 描述 | 优势 | 劣势 | 采用 |
|------|------|------|------|------|
| v2: BLOCK_M = G (仅 G 维) | S1 在外层串行 | 简单 | GEMM 调用次数多 (S1 次) | ✗ |
| **v3: BLOCK_M = s1BaseSize × G** | 一次 GEMM 处理 s1BaseSize 个 S1 × G 个 group | GEMM 调用次数减少 s1BaseSize 倍 | C_L0 容量增大 (需验证) | **✓** |

**BLOCK_M 联合切分后的 L0C 容量验证（v5: L0 inner split 后）**：

| sparse_count | s1BaseSize | G | BLOCK_M | BLOCK_N | v5 C_L0 [M_L0,N_L0] | 128KB? | L0 inner split 迭代 |
|-------------|-----------|---|---------|---------|---------------------|--------|---------------------|
| ≤2048 | 8 | 16 | 128 | 512 | [128,128]=64KB | ✓ | M:1, N:4 |
| ≤2048 | 8 | 64 | 512 | 512 | [128,128]=64KB | ✓ | M:4, N:4 |
| ≤2048 | 8 | 8 | 64 | 512 | [128,128]=64KB | ✓ | M:1, N:4 |
| >2048 (4096) | 4 | 16 | 64 | S2(整段) | [128,128]=64KB | ✓ | M:1, N:S2/128 |
| >2048 (8192) | 2 | 16 | 32 | S2(整段) | [128,128]=64KB | ✓ | M:1, N:S2/128 |

> **v5 L0 inner split 策略**（替代 v4 的 "inner split N"）：统一用 M_L0=128, N_L0=128，C_L0 固定 64KB。M 维和 N 维均内部分块，通过 T.gemm_v0 + BufferRegion 切片实现（Q21 详述）。
>
> **v4 的 "inner split N" 已废弃**：v4 仅对 N 维 inner split（BLOCK_N_inner = 128KB // (BLOCK_M × 4)），但 v4 的 BLOCK_M = s1BaseSize × G 已超 128KB（如 G=64 时 BLOCK_M=512, C_L0=[512,512]×4=1MB），仅 split N 不够。v5 同时 split M 和 N。

**Q13 回答（v3 新增）：sparse_count > 2048 时 tiling 调整**

| 参数 | sparse_count ≤ 2048 | sparse_count > 2048 |
|------|---------------------|---------------------|
| s1BaseSize | 8 | 8192/sparse_count*2 (如 4096→4, 8192→2) |
| s2BaseSize | 512 (S2 分块) | S2 (整段，不分块) |
| s2BaseNum | ceil(S2/512) | 1 (或 0) |
| virTopK | sparse_count | sparse_count (= virTopK) |
| UB sortOutBuf | CeilDiv(s1BaseSize,2) × virTopK × 2 × sizeof(float) | 同左 (s1BaseSize 更小) |
| GEMM 调用 | S2_blocks 次 (每块 BLOCK_N=512) | 1 次 (整段, inner split N) |

> **设计依据**：sparse_count > 2048 时，AscendC 通过缩小 s1BaseSize（减少并行 S1 数）来腾出 UB 空间容纳更大的 virTopK。S2 不分块是因为 topk 需要在完整 S2 上排序。

#### Kernel 1 并行度

**v4: Fixed Core + T.Pipelined 模式**

```python
# v4: Fixed Core (NUM_CORES=24) + 每核多任务 + S2 T.Pipelined
# 每个 core 处理 tasks_per_core = ceil(total_tasks / NUM_CORES) 个 (b, s1_tile) 任务
# 每个任务内 S2_blocks 次 T.Pipelined 迭代 (num_stages=2 流水)
# with T.Kernel(NUM_CORES, threads=2, is_npu=True) as (cid, vid):
```

**v3 方案 B（双 Kernel, split-N 并行）保留，v4 在此基础上改为 Fixed Core**：
- v4 Kernel 1: `T.Kernel(NUM_CORES, threads=2)` — Fixed Core + 每核多任务 + S2 T.Pipelined CV 融合
- Kernel 2: `T.Kernel(B * S1)` — topk 轻量计算 (纯 Vector, threads=1)，**不改 Fixed Core**（topk 任务数 B×S1 通常不大，且无 workspace 膨胀问题）
- **关键**: S2 切分是 split-N（各 block 处理不同 s2 位置），非 split-K，因此**无需跨核归约**。各 block 独立计算部分分数，写入 Scores workspace 的不同 s2 位置段。

**Q1 回答：B/S1/S2 三维切分映射到核维度**

| 方案 | 描述 | 优势 | 劣势 | 采用 |
|------|------|------|------|------|
| A: T.Kernel(B*N2), S2 serial | 参考实现方案 | 简单 | S2 无法并行, 核数用不满 | ✗ |
| B: T.Kernel(B*S1_tiles*S2_blocks), split-N, threads=2 | v3: 每个 block 处理一个 (b,s1_tile,s2_block) | S2 并行, CV 融合, 无需跨核归约 | workspace 膨胀 | ✗ (v4 替代) |
| **C: Fixed Core + T.Pipelined (v5)** | T.Kernel(min(total_tasks,NUM_CORES), threads=2), 每核多任务, S2 用 T.Pipelined 流水 | workspace 固定大小, CV overlap, L2 cache 友好, **小任务不空转 (v5 缺陷 #5)** | 每核需多任务循环 | **✓** |

L0 测试用例核数分析（A2, 24 cores, v5 Fixed Core with min(total_tasks, NUM_CORES)）：
- Kernel 1 (l0_li_default_a2): total_tasks = 18 × 1 = 18, launch_core_num=min(18,24)=18 → 每 core 1 任务, S2_blocks=6 → T.Pipelined 6 次（充分利用流水, 6 核不空转）
- Kernel 1 (l0_li_large_b_48): total_tasks = 48 × 1 = 48, launch_core_num=min(48,24)=24 → 每 core 2 任务, S2_blocks=6 → T.Pipelined 12 次
- Kernel 1 (l0_li_small_bf16): total_tasks = 1 × 1 = 1, launch_core_num=min(1,24)=1 → 1 核工作, 23 核休眠（v5 修正: 不 launch 24 核空转）
- Kernel 2: 18 × 3 = 54 blocks → 3 waves → 足够（topk 轻量, 不改 Fixed Core）

#### Kernel 2 (topk)

##### 路径选择：单次 topk vs 分段 topk

```python
MAX_S2_UB = 16384            # 单次 topk 的 UB 安全上限
SEGMENT_SIZE = 8192           # 分段 topk 的段大小

# 路径选择（host wrapper 或 JIT dispatch）
if MAX_S2 <= MAX_S2_UB:
    use_single_topk = True    # 路径 A: 单次 topk（score_accum 在 UB）
else:
    use_single_topk = False   # 路径 B: 分段 topk + merge_sort（score_accum 在 GM）
```

##### 路径 A: 单次 topk (S2 ≤ 16384)

```python
BLOCK_N_K2 = 128      # mask 处理块大小 (与 Kernel 1 BLOCK_N 一致)
block_num_k2 = B * S1  # 每个 core 处理一个 (b, s1) 的完整 S2 topk
# score_accum = T.alloc_shared((MAX_S2,), calc_dtype)  # UB
# T.tile.topk(topk_dst, score_accum, SPARSE_COUNT, act_k)  # 一次 topk
```

##### 路径 B: 分段 topk + merge_sort (S2 > 16384)

```python
SEGMENT_SIZE = 8192   # 单段大小（保证段 topk UB < 192KB）
num_segments = T.ceildiv(MAX_S2, SEGMENT_SIZE)
block_num_k2 = B * S1  # 每个 core 处理一个 (b, s1) 的完整 S2 topk

# Phase 1: mask + scatter → GM score_accum_gm
# Phase 2: 逐段 topk + merge_sort
for seg in T.serial(num_segments):
    seg_start = seg * SEGMENT_SIZE
    # 2a. 从 GM 加载段到 UB
    T.copy(score_accum_gm[seg_start:seg_start+SEGMENT_SIZE], segment_buf)
    # 2b. 段 topk → segment_topk_dst [2*SPARSE_COUNT] (value, local_index)
    T.tile.topk(segment_topk_dst, segment_buf, SPARSE_COUNT, seg_actual_num)
    # 2c. local_index → global_index: segment_topk_dst 中奇数位 += seg_start
    for i in T.serial(SPARSE_COUNT):
        segment_topk_dst[2*i + 1] = segment_topk_dst[2*i + 1] + seg_start
    # 2d. merge_sort: running_topk + segment_topk_dst → merge_output [4*SPARSE_COUNT]
    if seg == 0:
        T.copy(segment_topk_dst, running_topk)  # 首段直接复制
    else:
        T.tile.merge_sort(merge_output, running_topk, segment_topk_dst)
        T.copy(merge_output[0:2*SPARSE_COUNT], running_topk)  # 取前 k 个
# Phase 3: 提取索引 → output
T.tile.gather_mask(topk_index, running_topk, "P1010")
T.tile.cast(output_ub, topk_index, "CAST_ROUND", SPARSE_COUNT)
```

> **merge_sort 索引对齐**：T.tile.merge_sort 按 (value, index) 对排序（interleaved 格式，2 floats per element）。段 topk 输出的 local_index 转换为 global_index 后，merge_sort 能正确保留全局索引。最终 gather_mask("P1010") 提取奇数位即全局位置索引。
> **T.tile.merge_sort 来源验证**：`ascend_tile.py:320`，支持 2/3/4-way merge，参考 `examples/sort/example_merge_sort.py`。

### 5.3 约束分析

- **分形限制**: BLOCK_M=G≥16 ✓ (G=16/64), BLOCK_N=128≥16 ✓, BLOCK_K=128≥16 (fp16/bf16) ✓
- **v5 L0 inner split**: M_L0=128≥16 ✓, N_L0=128≥16 ✓ (分形约束满足)
- **v7 Cube 3 层 tiling**: M_L1=256≥16 ✓, S2_L1=256≥16 ✓, M_L0=128≥16 ✓, N_L0=128≥16 ✓ (分形约束满足)
- **block_size=768 对齐**: 768 % 16 = 0 ✓（T.gemm_v0 仅要求 16 倍数，无 2 的幂次约束）
- **对齐约束**: UB/L1 32B 对齐: BLOCK_N×sizeof(fp16)=256B ✓, G×D×sizeof(fp16)=4096B ✓
- **L0C 容量** (v5 修正): C_L0=[M_L0=128, N_L0=128]×4=64KB < 128KB ✓ **所有场景安全**（v4 的 [BLOCK_M,BLOCK_N] 全尺寸超限已修正）
- **v7.1 L1 容量** (v7.1 修正): A_L1_0+A_L1_1+B_L1_0+B_L1_1 = 4×32KB = **128KB < 512KB ✓**（v7 原方案 320KB 但 transpose_B+切片不可行, v7.1 独立 buffer 修正, §5.2.1）
- **UB 容量 (Kernel 1)** (v6 修正): row_expand_mul + reduce_sum merge 后峰值 ~34KB < 192KB ✓ **所有 N1 场景安全**（v5 的 68KB 已降为 34KB）
- **UB 容量 (Kernel 2 单次 topk 路径)**: 峰值 ~80KB (MAX_S2=8192) / ~160KB (MAX_S2=16384) < 192KB ✓
- **UB 容量 (分段 topk 路径)**: 峰值 ~96KB < 192KB ✓（与 S2 无关，仅与 SEGMENT_SIZE=8192 有关）
- **v7.2 UB 容量 (Kernel 2 sort 缓存优化, 方案 C: VID_S1=1)** (缺口 2 修正): v7.1 原方案 UB=216.4KB > 192KB ❌ 超限; **v7.2 方案 C (VID_S1=1 + buffer 复用) UB=184.4KB < 192KB ✓**（§8.8, 缺口 2 验证 `test_memory_budget.py`）
  - 方案 C 详情: VID_S1 从 2 改为 1（仅 S1≤4 场景）, topk_a_ub 从 (VID_S1=2, TA2)=32KB 缩小到 (VID_S1=1, TA2)=16KB, 省 16KB
  - sort_tmp 复用 cache_tmp_ub, merge_output 复用 merged_ub, 新增 sorted_cache 16KB
  - 净变化: +16KB (sorted_cache) - 16KB (topk_a_ub 缩小) = 0KB, UB 保持 184.4KB
  - 性能影响: S1≤4 用例占 46% (17/37) → 缓存路径, VID_S1=1 多 1 轮循环, 影响 <0.5%; S1>4 用例占 54% → 标准路径, 不受影响

### 5.4 尾块/非整除处理

| 场景 | 处理方案 |
|------|---------|
| S2 不被 BLOCK_N 整除 | `T.ceildiv(S2, BLOCK_N)` 计算块数; 尾块 valid_n = `T.min(BLOCK_N, act_k - s2_start)`; GEMM 仍计算满块, Vector 仅写 valid_n 个分数 |
| act_k < S2 (动态边界) | Kernel 2: `T.tile.fill(score_accum, -inf)` 初始化, 再写入有效位置; topk 的 actual_num=act_k 自动处理 |
| act_k < sparse_count | topk 输出不足 sparse_count 个, 剩余位置填 -1 (golden 行为一致) |
| G × block_size > L0C (大 block_size) | v5: L0 inner split (M_L0=128, N_L0=128), C_L0 固定 64KB (§4.6) |
| S2 > 16384 (UB 放不下 score_accum + sort_tmp) | 切换分段 topk 路径：score_accum 放 GM, 逐段 topk + merge_sort |
| TND layout | v5: kernel 直接消费 TND strided DMA, 无 host 转置 (§4.9) |
| block_size 非整除 S2 (如 768 整除 20481) | T.ceildiv + 尾块 valid_n 处理（同 S2 不整除场景） |

---

## 6. 循环与调度结构

### 6.1 循环结构总结

#### Kernel 1 (v6: Fixed Core + T.Pipelined + groupInner + L0 inner split + row_expand_mul + reduce_sum merge)

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| launch_core_num | block 级并行 | T.Kernel(min(total_tasks,动态核数), threads=2) | v6: Fixed Core + 动态核数, 小任务不空转 (缺陷 #5) |
| tasks_per_core | 串行 | T.serial(tasks_per_core) | v4: 每核多任务循环 |
| S2_blocks | **核间流水** | **T.Pipelined(s2BaseNum, num_stages=2, cross_interval=2)** | v4: CV overlap, 迭代间 Cube/Vector 重叠 |
| **L0 inner split M** (v5 新增) | 串行 | T.serial(T.ceildiv(BLOCK_M, M_L0)) | v5: Cube 侧 L0 M 维分块 (M_L0=128) |
| **L0 inner split N** (v5 新增) | 串行 | T.serial(T.ceildiv(BLOCK_N, N_L0)) | v5: Cube 侧 L0 N 维分块 (N_L0=128) |
| **s1_inner** (v5 新增) | 串行 | T.serial(s1BaseSize) | v5: Vector 侧 s1 维内层循环 |
| **outerG** (v5 新增) | 串行 | T.serial(T.ceildiv(G, groupInner)) | v5: Vector 侧 G 维 groupInner=16 分块 |
| **innermost body** (v6 更新) | — | **row_expand_mul_experiment** + **reduce_sum(clear=False)** + **pipe_barrier("v")** | v6: 融合 Brcb+Mul + 融合 reduce+累加 + Vector 管线轻量同步 (替代 v5 broadcast+mul+reduce+add+barrier_all) |

#### Kernel 2

##### 单次 topk 路径 (S2 ≤ 16384)

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| B × S1 | block 级并行 | T.Kernel(B * S1) | 每个 core 处理一个 (b, s1) 的完整 topk |
| S2_blocks (mask) | 串行 | T.serial(T.ceildiv(MAX_S2, BLOCK_N)) + if s2_start < act_k | 静态循环边界 + 运行时条件 (ascend 约束: 循环次数不能依赖 tensor 值) |
| BLOCK_N (scatter) | 串行 | T.serial(BLOCK_N) | 逐元素 scatter 到 score_accum |
| topk | 单次 | T.tile.topk(score_accum, k, act_k) | score_accum 在 UB |

##### 分段 topk 路径 (S2 > 16384)

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| B × S1 | block 级并行 | T.Kernel(B * S1) | 每个 core 处理一个 (b, s1) |
| S2_blocks (mask+scatter) | 串行 | T.serial(T.ceildiv(MAX_S2, BLOCK_N)) + if | mask 应用后 scatter 到 GM score_accum_gm |
| num_segments (段 topk) | 串行 | T.serial(T.ceildiv(MAX_S2, SEGMENT_SIZE)) + if seg_start < act_k | 逐段从 GM 加载 + topk |
| SPARSE_COUNT (index 转换) | 串行 | T.serial(SPARSE_COUNT) | local_index → global_index (加 seg_start) |
| merge_sort | 条件执行 | T.tile.merge_sort(running, segment) | 非首段时合并, 首段直接复制 |

### 6.2 循环伪代码

见 §3.3 完整伪代码。

### 6.3 流水线优化（v4: T.Pipelined 核间流水）

#### Kernel 1: T.Pipelined 核间流水（v4 新增）

**v4 流水设计**：S2 循环用 `T.Pipelined(s2BaseNum, num_stages=2, cross_interval=2)` 表达核间 CV overlap。

**流水工作原理**：
```
时间 →
迭代 0:  [Cube: QK_0 → ws_0] [Vector: mul+reduce_0 → Scores_0]
迭代 1:  [Cube: QK_1 → ws_1] [Vector: mul+reduce_1 → Scores_1]
                    ↑ 重叠 ↑
v4 T.Pipelined(num_stages=2):
迭代 0:  [Cube: QK_0 → ws_0]
迭代 1:  [Cube: QK_1 → ws_1] [Vector: mul+reduce_0 → Scores_0]  ← Cube/Vector 重叠
迭代 2:  [Cube: QK_2 → ws_0] [Vector: mul+reduce_1 → Scores_1]  ← ws_0 复用 (ring-buffer)
...
```

**关键参数**：
- `num_stages=2`：2 级流水（double buffer），Cube 迭代 k 与 Vector 迭代 k-1 重叠
- `cross_interval=2`：每 2 次迭代同步一次，减少跨核同步开销（参考 `performance-antipatterns.md` §"AIC/AIV 混合算子未开启 CV overlap" 调参建议）
- workspace ring-buffer：`qk_workspace[cid, s2_block % num_stages, :, :]`，2 slot 交替使用

**为什么 num_stages=2 而非 8-16**（与 flash_attn_optimize.md §1 的 FA 场景对比）：
- FA 场景：seq_len/block_N 通常 64-512 次，num_stages=8-16 可充分流水
- 本算子：s2BaseNum = ceil(S2/512)，S2=3072 时仅 6 次，num_stages=2 已能覆盖大部分迭代
- **v6 UB 安全性修正**：v6 row_expand_mul + reduce_sum merge 后 UB=34KB（v5: 68KB），UB buffer 在 pipeline body 外不被 ring-buffer，num_stages 不受 UB×num_stages 限制
- **v4 的错误**：v4 UB=129KB（BLOCK_M=128,BLOCK_N=128），num_stages=2 时 ring-buffer 258KB > 192KB 有风险
- **安全策略**：UB buffer 在 pipeline body **外**分配（参考 `matmul_add_pipeline.py`），编译器对 body 外 buffer 不做 ring-buffer

> **⚠️ v6 num_stages 建议（v7.1 已验证, 见 §12.8 API 验证结果）**：
> - **v6 UB=34KB**（v5: 68KB），buffer 在 pipeline body 外不被 ring-buffer
> - num_stages 不受 UB×num_stages 限制，仅受 GM workspace ring-buffer slots 和 pipeline 有效性限制
> - **v7.1 验证结论**：num_stages=3 和 4 均编译通过且精度正确（验证 4 PASS, diff=0.0312）
> - **关键前提（v7.1 已验证）**：buffer 必须在 `T.Pipelined` body **外部**分配（验证 4 确认, body 内分配会 ring-buffer 放大 UB）
> - **threads=2 解包（v7.1 已验证）**：`with T.Kernel(blocks, threads=2, is_npu=True) as (cid)` — 1 个值, 不是 `as (cid, _)`
> - num_stages=3 参考 `matmul_add_pipeline.py:46`（已验证可运行）
> - **参考 `flash_attn_optimize.md` 的 "KNOWN BROKEN" 教训**：UB buffer 不在 pipeline body 内分配（`flash_attn_bhsd_auto_pipeline_h16_d128.py` num_stages=8 时 UB ring-buffer 超限）
> - **Stage 2 验证步骤**：① 编译后 get_kernel_source 检查 UB buffer 是否在 body 外（不被 ring-buffer）② msprof 对比 num_stages=2/3/4 的 Cube/Vector 重叠度

**cross_interval=2 的效果**：
- cross_interval=1（默认）：每次迭代都同步，S2=3072 时同步 6 次
- cross_interval=2：每 2 次迭代同步，S2=3072 时同步 3 次（减少 50% 同步开销）
- **代价**：需要 2 个 workspace slot（已由 num_stages=2 保证）

#### Kernel 2: 不使用 T.Pipelined

- Kernel 2 UB 占用高（score_accum 64KB + topk_sort_tmp 96KB = 160KB），T.Pipelined ring-buffer 会立即溢出
- Kernel 2 是纯 Vector（无 Cube），T.Pipelined 的 CV overlap 收益不适用
- Kernel 2 性能优化通过分段 topk + merge_sort 已解决

### 6.4 尾块处理

见 §5.4。

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 混合（Kernel 1: AUTO_CV_SYNC + T.Pipelined cross_interval 批量同步, Kernel 2: 自动同步）

### 7.2 同步点说明

#### Kernel 1 (v6: Developer + CV 融合 + T.Pipelined + pipe_barrier("v"))

| 位置 | 同步机制 | 理由 |
|------|----------|------|
| T.Pipelined 循环内 Cube→Vector | **AUTO_CV_SYNC 自动 cross_flag** + **T.Pipelined cross_interval=2 批量同步** | v4: 编译器自动插入 set_cross_flag/wait_cross_flag, cross_interval=2 每 2 次迭代同步一次 |
| T.Pipelined 循环内 Vector 内部 (v6) | **AUTO_SYNC 自动 barrier** + **v6: `T.pipe_barrier("v")` Vector 管线屏障** | v6: row_expand_mul 后插 `pipe_barrier("v")` 确保 qk_ub 写完成再 reduce_sum; 比 barrier_all 更轻量 (仅等 Vector 管线, 不等 Cube/MTE2/MTE3) |
| tasks_per_core 循环间 | **T.serial 自然串行** | v4: 每核多任务串行执行, 无需额外同步 |

**v4 T.Pipelined + cross_interval 同步说明（Q16 回答）**：

**T.Pipelined 与 threads=2 + AUTO_CV_COMBINE 的组合方式**（源码验证）：

| 验证项 | 来源 | 证据 |
|--------|------|------|
| T.Pipelined API | `pipeline.py:11` | `Pipelined(start, stop, num_stages, order, stage, sync, group, cross_interval=1)` |
| T.Pipelined + AUTO_CV_COMBINE 可运行 | `examples/pipeline/matmul_add_pipeline.py:46` | `for k in T.Pipelined(loop_k, num_stages=3)` + AUTO_CV_COMBINE + (cid, vid) 模式, 已验证可运行 |
| T.Pipelined + AUTO_CV_COMBINE (Expert 风格) | `examples/pipeline/sparse_flash_attn_gqa_pipeline.py:128` | `for i_i in T.Pipelined(NI, num_stages=2)` + AUTO_CV_COMBINE + alloc_L1/ub, 已验证可运行 |
| T.Pipelined + threads=2 UB 风险 | `examples/flash_attention/fa_opt/flash_attn_bhsd_auto_pipeline_h16_d128.py` | 文档标注 KNOWN BROKEN: num_stages=8 时 UB ring-buffer 超限。num_stages=2 安全 |
| cross_interval 参数 | `pipeline.py:19` | `cross_interval: int = 1`, 默认每次同步, N=2 每 N 次同步 |

**组合方式**：
1. `pass_configs` 开启 `AUTO_CV_COMBINE: True` + `AUTO_CV_SYNC: True` + `AUTO_SYNC: True`
2. `T.Kernel(NUM_CORES, threads=2, is_npu=True) as (cid, vid)` — threads=2 使编译器生成 AIC+AIV 双核代码
3. `for s2_block in T.Pipelined(s2BaseNum, num_stages=2, cross_interval=2):` — S2 循环用 T.Pipelined
4. 编译器自动：识别 GEMM → AIC 代码段, 其余 → AIV 代码段, 插入 cross_flag 同步
5. `cross_interval=2` 使编译器每 2 次迭代才插入一次 cross_flag 同步（减少同步开销）

**v3 Expert 模式同步（fallback, 保留）**：

| 位置 | 同步 API | 理由 |
|------|----------|------|
| GEMM 前后 | `T.barrier_all()` | 确保 L1 数据搬运完成, GEMM 结果写入 L0C |
| L0C→GM copy 后 (Cube 侧) | `T.set_cross_flag("FIX", 0)` | 通知 Vector: QK ReLU 结果已写入 workspace |
| GM→UB copy 前 (Vector 侧) | `T.wait_cross_flag(0)` | 等待 Cube 完成 QK 计算 |
| weight mul 前 | `T.barrier_all()` | 确保 workspace→UB 搬运完成 |

**Q5 回答（v3 保留）: cross_flag 同步设计**
- `T.set_cross_flag(pipe, flag, mode=2)` 在 ascend.py:116 确认可用
- `T.wait_cross_flag(flag, pipe="")` 在 ascend.py:138 确认可用
- pipe="FIX" 用于 Cube 侧 L0C→GM 通路, pipe="MTE3" 用于 Vector 侧 UB→GM 通路
- mode=2（默认）表示同组 AIC-AIV 之间同步, 适用于 CV 融合场景
- 与 Developer 模式的自动同步不冲突: Kernel 1 是 Expert 模式（无 AUTO_SYNC）, Kernel 2 是 Developer 模式（有 AUTO_SYNC, 无需手动同步）

#### Kernel 2 (Developer)

- `TL_ASCEND_AUTO_SYNC=True`: 编译器自动插入同步, 无需手动 T.barrier_all
- 无 C/V 交互（纯 Vector）, 无需 cross_flag

#### Kernel 间同步

- Kernel 1 和 Kernel 2 通过顺序 launch 保证依赖: Kernel 2 读取 Kernel 1 的 Scores 输出
- `torch.npu.synchronize()` 或 TileLang 隐式同步保证 Kernel 1 完成后再启动 Kernel 2

### 7.3 pass_configs 配置（v4 更新）

```python
# Kernel 1 (v4: Developer + CV 融合 + T.Pipelined)
pass_configs_k1 = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,      # 自动 CV 分割
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,         # 自动跨核同步 (配合 T.Pipelined)
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,            # 自动 Vector 内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,      # UB 地址复用
    tilelang.PassConfigKey.TL_ASCEND_PTO_USE_PIPE_IN_CV_COPY: False,  # CV copy 不用 PTO
}

# Kernel 2 (Developer)
pass_configs_k2 = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,        # 自动同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # UB 地址复用
}
```

---

## 8. 融合算子设计

### 8.1 融合算子判定

**判定结果**: 是（Kernel 1 为 CV 融合算子: GEMM + weight mul + group reduce）

**判定依据**: Kernel 1 包含 Cube 计算 (GEMM) 和 Vector 后处理 (weight mul, reduce_sum), 需 CV 协同。

### 8.2 v3 CV 协同方案：threads=2 + AUTO_CV_COMBINE（推荐）

**Q14 回答（v3 新增）：TileLang 支持 AIC:AIV=1:2 单 Kernel CV 协同**

**源码验证结论**：TileLang **支持** AIC:AIV=1:2 模式，通过 `threads=2` 参数实现。

| 验证项 | 来源 | 证据 |
|--------|------|------|
| threads 参数支持 2 | `tilelang/language/kernel.py:253` | `assert threads in [1, 2], f"NPU kernel threads must be 1 or 2"` |
| dev_mode 标记 | `kernel.py:262` | `attrs["tilelang.is_npu_kernel_frame_dev_mode"] = True` (threads 指定时设置) |
| AUTO_CV_COMBINE pass | `pass_config.py` | `TL_ASCEND_AUTO_CV_COMBINE` 配置项 |
| AUTO_CV_SYNC pass | `pass_config.py:50` | `TL_ASCEND_AUTO_CV_SYNC = "tl.ascend_auto_cross_core_sync"` |
| CrossCorePipeline pass | `transform/__init__.py:406` | `CrossCorePipeline()` 跨核流水线 pass |
| 参考实现 1 | `examples/developer_mode/matmul_add_developer.py:40` | `T.Kernel(m_num * n_num, threads=2, is_npu=True)` + GEMM + add CV 融合 |
| 参考实现 2 | `examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py:85` | `T.Kernel(block_num, threads=2, is_npu=True)` + GEMM + softmax + reduce CV 融合 |
| cross_flag mode=2 | `ascend.py:130` | `mode=2: between AICs and AIVs within the same group` |

**Developer + threads=2 CV 融合模式**：

```python
@tilelang.jit(out_idx=[7], workspace_idx=[6], pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,      # 自动 CV 分割
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,         # 自动跨核同步
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,            # 自动 Vector 内同步
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,      # UB 地址复用
    tilelang.PassConfigKey.TL_ASCEND_PTO_USE_PIPE_IN_CV_COPY: False,  # CV copy 不用 PTO
})
def lightning_indexer_score(...):
    @T.prim_func
    def main(...):
        with T.Kernel(block_num_k1, threads=2, is_npu=True) as (cid, vid):
            # --- 以下代码编译器自动分割: GEMM → AIC, 其余 → AIV ---
            Q_L1 = T.alloc_shared((BLOCK_M, D), input_dtype)      # L1
            K_L1 = T.alloc_shared((BLOCK_N, D), input_dtype)      # L1
            C_L0 = T.alloc_fragment((BLOCK_M, BLOCK_N), calc_dtype)  # L0C

            T.copy(Query[...], Q_L1)
            T.copy(Key[...], K_L1)
            T.gemm_v0(Q_L1, K_L1, C_L0, transpose_B=True, init=True)  # AIC 执行
            T.copy(C_L0, qk_workspace[...])                          # L0C → GM (编译器自动插 cross_flag)

            # Vector 侧 (编译器自动插 wait_cross_flag)
            qk_ub = T.alloc_shared((BLOCK_M, BLOCK_N), calc_dtype)  # UB
            T.copy(qk_workspace[...], qk_ub)
            # ... weight mul + reduce_sum ...
```

**AUTO_CV_COMBINE 工作原理**：
1. 编译器扫描 prim_func，识别 GEMM 调用（`T.gemm_v0`）→ 分配给 AIC
2. 其余计算（T.copy, T.tile.mul, T.reduce_sum 等）→ 分配给 AIV
3. AIC → AIV 的数据依赖通过 GM workspace 中转，编译器自动插入 `set_cross_flag`/`wait_cross_flag`
4. `TL_ASCEND_AUTO_CV_SYNC` 自动管理 flag ID 分配和同步点

**与 AscendC `KERNEL_TYPE_MIX_AIC_1_2` 的对应关系**：

| AscendC | TileLang | 说明 |
|---------|----------|------|
| `KERNEL_TYPE_MIX_AIC_1_2` | `threads=2` | AIC:AIV=1:2 模式 |
| `CrossCoreSetFlag/CrossCoreWaitFlag` | `AUTO_CV_SYNC` 自动插入 | 跨核同步 |
| `TQue<TPosition::AIC>` / `TQue<TPosition::VEC>` | `alloc_shared` (L1) / `alloc_shared` (UB) | 编译器根据位置分配 |
| `TBuf<TPosition::A1>` / `TBuf<TPosition::A2>` | `alloc_shared` (L1) / `alloc_fragment` (L0C) | L1/L0C 分配 |

### 8.3 S2 跨核 topk 归并方案分析（Q12 回答）

**问题背景**：当 S2 被分到多个核（s2BaseNum > 1）时，每个核只计算部分 S2 的分数。AscendC 通过 ProcessLD 阶段做跨核 topk 归并。当前 v2 设计中，Kernel 2 每个 (b,s1) 由一个核处理完整 S2 的 topk（分段 topk 在核内完成），**不涉及跨核归并**。

**方案对比**：

| 方案 | 描述 | 优势 | 劣势 | 推荐 |
|------|------|------|------|------|
| **A: 双 Kernel + 核内分段 topk (v2 现有)** | Kernel 1 split-N 写分数到 GM, Kernel 2 每 (b,s1) 核内分段 topk + merge_sort | 架构简单, 无跨核通信, 已验证 | Kernel 2 大 S2 时单核串行 (分段 topk 缓解) | **✓ (一期)** |
| B: 单 Kernel + threads=2 + 跨核归并 | 单 Kernel 内 AIC 做 GEMM, AIV 做 topk, SyncAll 后首核归并 | 与 AscendC 一致, 性能最优 | 跨核通信复杂, TileLang 无直接 SyncAll API | ✗ (二期) |
| C: 三 Kernel (GEMM + per-segment topk + merge) | Kernel 1 GEMM, Kernel 2 每 (b,s1,s2_seg) topk, Kernel 3 归并 | 职责单一 | 3 Kernel launch 开销, GM 读写多 | ✗ |

**推荐方案 A（一期保留 v2 架构，理由如下）**：

1. **v2 的分段 topk 已解决大 S2 UB 容量问题**：S2 > 16384 时，Kernel 2 在核内做分段 topk + merge_sort，UB 峰值 ~96KB < 192KB ✓（§4.5）。无需跨核归并。

2. **跨核归并是性能优化，非功能需求**：AscendC 的 ProcessLD 跨核归并是为了并行化 topk 阶段（多核同时做不同 S2 段的 topk）。当前 v2 方案中 topk 在单核内分段串行，功能正确但大 S2 时性能可能不如 AscendC。

3. **TileLang 跨核通信 API 限制**：
   - `T.sync_all()` (`ascend.py:229`) 是核内全管线屏障，非跨核 SyncAll
   - `T.set_cross_flag(mode=0)` (ascend.py:128) 支持核间同步，但 flag ID 管理复杂
   - 无直接的 `CrossCoreSyncAll` API（AscendC `SyncAll<HardEvent::MTE3>` 的等价物）
   - 实现跨核归并需要手动 GM workspace + flag 同步，复杂度高

4. **一期优先功能正确性**：方案 A 已通过设计验证，Stage 2 可直接实现。方案 B 的跨核归并留作 Stage 3 性能优化。

**方案 B（二期优化路径）**：若 Stage 3 性能分析发现 Kernel 2 大 S2 topk 成为瓶颈，可升级为方案 B：
- Kernel 1: `T.Kernel(B * S1_tiles * S2_blocks, threads=2)` — CV 融合 + 每 block 做段 topk(2048) 写 GM
- Kernel 2: `T.Kernel(B * S1)` — 读所有 S2_blocks 段的 topk, T.tile.merge_sort 归并
- 需评估 GM workspace 增量：B × S1 × S2_blocks × 2 × 2048 × 4B

### 8.4 workspace 设计（v5: Fixed Core + ring-buffer + L0 inner split）

**v5 Developer + threads=2 + T.Pipelined 模式 workspace**：

| workspace | v3 Shape | v4/v5 Shape | dtype | 用途 |
|-----------|---------|---------|-------|------|
| qk_workspace | [total_blocks, BLOCK_M, BLOCK_N] | **[NUM_CORES, num_stages, BLOCK_M, BLOCK_N]** | calc_dtype (float32) | C→V 中转: L0C→GM(ReLU)→UB, v4: ring-buffer 2 slot, v5: Vector 侧按 [groupInner, BLOCK_N] 切片读取 |
| Scores | [B, N2, S1, S2] | [B, N2, S1, S2] (不变) | calc_dtype (float32) | Kernel 1 输出 / Kernel 2 输入 |

**v5 workspace 关键改进（v4 基础上）**：
- `qk_workspace` 仍是 `[NUM_CORES, num_stages, BLOCK_M, BLOCK_N]`（GM 无容量限制，BLOCK_M×BLOCK_N 可达 512×512）
- v5 Cube 侧：L0 inner split 后分次写入 `qk_workspace[cid, slot, m_l0*M_L0:(m_l0+1)*M_L0, :]`（部分写入）
- v5 Vector 侧：按 `[groupInner, BLOCK_N]` 切片读取 `qk_workspace[cid, slot, m_start:m_start+groupInner, :]`
- `workspace_idx=[6]` 指定 qk_workspace 为编译器管理的 workspace
- **L2 cache 友好**：24 × 2 × BLOCK_M × BLOCK_N × 4B（如 24×2×512×512×4=48MB，可能超 L2 cache 行，但 GM 无容量限制）
- **v4 保留**：ring-buffer 2 slot 交替访问，每核独立管理

**Expert 模式（v2 fallback）workspace**：

| workspace | Shape | dtype | 用途 |
|-----------|-------|-------|------|
| qk_workspace | [total_blocks, BLOCK_M, BLOCK_N] | calc_dtype (float32) | C→V 中转: L0C→GM(ReLU)→UB |
| Scores | [B, N2, S1, S2] | calc_dtype (float32) | Kernel 1 输出 / Kernel 2 输入 |

### 8.5 Cube/Vector 核计算流程（v7.1: 独立 L1 buffer + 1D 平铺 sort cache + v6 API 优化保留, API 验证后修正）

> **v7.1 关键更新（API 验证后修正）**：
> 1. **Cube 侧**：M_L1=256, S2_L1=256 对齐 AscendC, **独立 L1 buffer 方案**（v7 L1 行切片 + transpose_B 不可行, 验证 3 修正）, 2×2 子 GEMM 展开
> 2. **Vector 侧**：保留 v6 的 row_expand_mul + reduce_sum(clear=False) + pipe_barrier("v")（v7.1 验证 5 PASS）, sort 缓存优化用 1D 平铺 + sort_tmp（§8.8, 验证 1+2 修正）
> 3. **数据范围特化**：G=64 优先（官方文档 N1=64），groupInner=16，outerG=4
> 4. **已验证 API**：num_stages=3-4（验证 4 PASS）, threads=2 `as (cid)`（验证 4 确认）, pipe_barrier("v")（验证 5 PASS）
>
> **v7.2 关键更新（缺口验证后修正, 见 §12.10）**：
> 5. **Cube→Vector 数据路径约束** (缺口 1): L0C 不能直接到 UB（硬件内存层级约束, L0C 是 Cube 专用 accumulator, Vector 核不能直接访问）。正确路径: `L0C(float32) → GM workspace(cast to float16, enable_relu) → UB → Vector 处理 → GM output`。参考 `examples/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa_no_cv_pipeline.py` 的 Cube→Vector 数据传递模式。
> 6. **Expert 模式 T.Scope("V") 解包要求** (缺口 1): 若使用 Expert 模式 `T.Scope("V")`（Vector 核内显式作用域）, Kernel 必须用 `with T.Kernel(blocks, is_npu=True) as (cid, vid):` 解包 2 个值, **不能用 `as (cid, _)`**。codegen 在 T.Scope("V") 内会引用 `vid`, 用 `_` 会报 `Find undefined Variable _`。参考 `lightning_indexer.py:383` 的 `as (cid, vid)` 用法。
> 7. **Expert T.mma + 独立 L1 buffer 已验证可行** (缺口 1): 4 个独立 (128,128) L1 buffer + T.copy(transpose=True) + T.mma 组合精度正确 (diff=0.0156), L1 占用 128KB < 512KB ✓（与本节 Developer 模式 T.gemm_v0 方案一致, 可作为 Expert 模式 fallback）

```python
# v7: Cube 3 层 tiling 对齐 AscendC + v6 API 优化保留 + sort 缓存优化
# v7 保留 v6: row_expand_mul_experiment + reduce_sum(clear=False) + pipe_barrier("v") + 动态核数

NUM_CORES = int(torch.npu.get_device_properties("npu").cube_core_num)  # v6: 动态核数 (A2=24, A3=20)
num_stages = 2  # T.Pipelined 流水深度 (v7: UB=34KB 有余量, Stage 2 尝试 3-4)
groupInner = 16  # v5: Vector 侧 G 维内层分块 (AscendC arch22)

# v7 新增: Cube 3 层 tiling 参数 (对齐 AscendC service_cube.h:54-65)
M_L1 = 256        # v7: L1 层 M 维 (对齐 AscendC M_BASIC_BLOCK=256, v6 是 S1_BLOCK*G=512)
S2_L1 = 256       # v7: L1 层 S2 维 (对齐 AscendC S2_BASIC_BLOCK=256, v6 是 s2BaseSize=512)
M_L0 = 128        # v5: L0 层 M 维 (不变, 对齐 AscendC M_BASIC_BLOCK_L0=128)
N_L0 = 128        # v5: L0 层 S2 维 (不变, 对齐 AscendC S2_BASIC_BLOCK_L0=128)

launch_core_num = T.min(total_tasks, NUM_CORES)  # v5 缺陷 #5: 小任务不空转

with T.Kernel(launch_core_num, threads=2, is_npu=True) as (cid, vid):
    # --- UB/L1/L0C buffer 在 pipeline body 外分配 (避免 ring-buffer, Q19) ---
    # v7.1: 独立 L1 buffer (API 验证 3 修正: transpose_B + L1 切片不可行)
    # 每个 [M_L0, D] 或 [N_L0, D], 从 GM 直接载入, 无切片
    A_L1_0 = T.alloc_shared((M_L0, D), input_dtype)    # v7.1: L1, [128,128]=32KB (A 行块 0)
    A_L1_1 = T.alloc_shared((M_L0, D), input_dtype)    # v7.1: L1, [128,128]=32KB (A 行块 1)
    B_L1_0 = T.alloc_shared((N_L0, D), input_dtype)    # v7.1: L1, [128,128]=32KB (B 行块 0)
    B_L1_1 = T.alloc_shared((N_L0, D), input_dtype)    # v7.1: L1, [128,128]=32KB (B 行块 1)
    # v7.3: 4×L0C (参考算子模式 1, HISA 行 156-159)
    # 每个子 GEMM 独占一个 L0C fragment, 消除 MMA 与 L0C→GM copy 的 FIX↔M flag 争用
    # v7.2: 单 C_L0 (64KB), 4 子 GEMM 串行复用 → MMA 与 copy 争用
    # v7.3: 4×L0C (256KB < 512KB ✓), 4 子 GEMM 可并行 issue 到独立 L0C
    C_L0_0 = T.alloc_fragment((M_L0, N_L0), calc_dtype)  # v7.3: L0C for (m0, n0) 子 GEMM
    C_L0_1 = T.alloc_fragment((M_L0, N_L0), calc_dtype)  # v7.3: L0C for (m1, n0) 子 GEMM
    C_L0_2 = T.alloc_fragment((M_L0, N_L0), calc_dtype)  # v7.3: L0C for (m0, n1) 子 GEMM
    C_L0_3 = T.alloc_fragment((M_L0, N_L0), calc_dtype)  # v7.3: L0C for (m1, n1) 子 GEMM
    # v7.1 L1 总占用: 4×32KB = 128KB < 512KB ✓ (v7 原方案 320KB, v6 640KB 超限)
    # v7.3 L0C 总占用: 4×64KB = 256KB < 512KB ✓ (v7.2 单 L0C 64KB, 见 §4.6 验证)

    # v7.3: annotate_layout (参考算子模式 2, fa_opt 行 105-112)
    # Q L1 用 ZN layout (行主序, GEMM 行访问连续)
    # K L1 用 NZ layout (列主序, transpose_B=True 时 B^T 访问连续, 减少 L1 bank conflict)
    # ⚠️ Stage 2 验证: annotate_layout 与独立 L1 buffer 方案的兼容性 (v7.1 已验证独立 buffer, layout 未验证)
    T.annotate_layout({
        A_L1_0: tilelang.language.make_zn_layout(A_L1_0),
        A_L1_1: tilelang.language.make_zn_layout(A_L1_1),
        B_L1_0: tilelang.language.make_nz_layout(B_L1_0),
        B_L1_1: tilelang.language.make_nz_layout(B_L1_1),
    })

    # Vector 侧 UB buffer (v6: 移除 weight_2d + scores_partial, pipeline body 外, 不被 ring-buffer)
    qk_ub = T.alloc_shared((groupInner, BLOCK_N), calc_dtype)       # v5: [16,512]=32KB, v6: row_expand_mul dst=src0
    weight_ub = T.alloc_shared((groupInner,), calc_dtype)            # v5: [16]=64B, v6: row_expand_mul src1 (1D)
    # v6 移除: weight_2d (32KB) — row_expand_mul 内部 Brcb 广播
    # v6 移除: scores_partial (2KB) — reduce_sum(clear=False) merge 语义
    scores_accum = T.alloc_shared((BLOCK_N,), calc_dtype)            # v5: [512]=2KB, v6: reduce_sum(clear=False) dst

    # v5: Fixed Core 每核多任务循环
    tasks_per_core = T.ceildiv(total_tasks, launch_core_num)
    for t in T.serial(tasks_per_core):
        task_id = cid * tasks_per_core + t
        if task_id < total_tasks:
            b = task_id // S1_tiles
            s1_tile = task_id % S1_tiles

            # v4: TND 前缀和读取 (Q18, v5 标注 Stage 2 验证)
            if layout_query == "TND":
                # ⚠️ v5 缺陷 #6: act_seq_q[b-1] 标量读取需 Stage 2 验证
                # fallback: host wrapper 预处理前缀和→per-batch 偏移
                t_start_q = T.if_then_else(b == 0, 0, actual_seq_q[b - 1])
                act_q_batch = actual_seq_q[b] - t_start_q
            else:
                t_start_q = b  # BSND: batch 维直接索引
                act_q_batch = actual_seq_q[b] if actual_seq_q is not None else S1

            if layout_key == "TND":
                t_start_k = T.if_then_else(b == 0, 0, actual_seq_k[b - 1])
                act_k_batch = actual_seq_k[b] - t_start_k
            else:
                t_start_k = b  # BSND/PA_BSND
                act_k_batch = actual_seq_k[b] if actual_seq_k is not None else S2

            # v5: S2 核间流水 (T.Pipelined)
            for s2_block in T.Pipelined(s2BaseNum, num_stages=num_stages, cross_interval=2):
                s2_start = s2_block * BLOCK_N
                if s2_start < act_k_batch:
                    # ============ AIC 侧 (编译器自动识别 T.gemm_v0) ============

                    # v7.1: Cube 3 层 tiling GEMM (独立 L1 buffer, API 验证 3 修正)
                    # v7 原方案用 Q_L1[m_slice, :] + K_L1[s2_slice, :] + transpose_B=True → 精度错误
                    # v7.1: 每个 (M_L0, N_L0) 子 tile 用独立 L1 buffer, 从 GM 直接载入, 无切片
                    # M_L1=256=2×M_L0=128, S2_L1=256=2×N_L0=128 → 每 L1 tile 4 个 L0 sub-tile
                    # T.gemm_v0 的 M,N 从 C_L0 shape 推导 (ascend.py:413)

                    # v7.1: L1 层 S2 循环 (AscendC: s2GmOffset, S2_BASIC_BLOCK=256)
                    for s2_l1 in T.serial(T.ceildiv(BLOCK_N, S2_L1)):
                        s2_l1_base = s2_start + s2_l1 * S2_L1

                        # B_L1: GM → 独立 L1 buffer (2 个 N_L0 行块, layout-aware, 无切片)
                        if layout_key == "TND":
                            k_off = t_start_k + s2_l1_base
                            T.copy(Key[k_off : k_off + N_L0, 0, :], B_L1_0)
                            T.copy(Key[k_off + N_L0 : k_off + 2 * N_L0, 0, :], B_L1_1)
                        elif layout_key == "PA_BSND":
                            # PA: block_table 寻址, block_size=128 时 1 block = 1 个 N_L0 子 tile
                            # Stage 2 需处理 block_size < N_L0 的多 block 聚合加载
                            T.copy(Key[block_table[b, s2_block + s2_l1 * 2], :, 0, :], B_L1_0)
                            T.copy(Key[block_table[b, s2_block + s2_l1 * 2 + 1], :, 0, :], B_L1_1)
                        else:  # BSND
                            T.copy(Key[b, s2_l1_base : s2_l1_base + N_L0, 0, :], B_L1_0)
                            T.copy(Key[b, s2_l1_base + N_L0 : s2_l1_base + 2 * N_L0, 0, :], B_L1_1)

                        # v7.1: L1 层 M 循环 (AscendC: s1gGmOffset, M_BASIC_BLOCK=256)
                        for m_l1 in T.serial(T.ceildiv(BLOCK_M, M_L1)):
                            m_l1_base = s1_tile * s1BaseSize + m_l1 * M_L1

                            # A_L1: GM → 独立 L1 buffer (2 个 M_L0 行块, layout-aware, 无切片)
                            if layout_query == "TND":
                                q_off = t_start_q + m_l1_base
                                T.copy(Query[q_off : q_off + M_L0, :, :], A_L1_0)
                                T.copy(Query[q_off + M_L0 : q_off + 2 * M_L0, :, :], A_L1_1)
                            else:  # BSND
                                T.copy(Query[b, m_l1_base : m_l1_base + M_L0, :, :], A_L1_0)
                                T.copy(Query[b, m_l1_base + M_L0 : m_l1_base + 2 * M_L0, :, :], A_L1_1)

                            # v7.3: 4×L0C — 4 子 GEMM 并行 issue 到独立 L0C (参考算子模式 1, HISA 行 156-159)
                            # v7.2: 4 子 GEMM 串行复用单 C_L0 (GEMM→copy→GEMM→copy→...), FIX↔M flag 争用
                            # v7.3: 4 子 GEMM 各写独立 L0C (GEMM×4 → copy×4), 消除争用, M pipeline 可 overlap
                            # 每个 init=True (K 维 D=128 一次算完, 无 K 维累加)
                            # v7.2 缺口 1: L0C → GM workspace 数据路径 (L0C 不能直接到 UB, 硬件内存层级约束)
                            #   Cube 侧: GEMM → L0C(float32) → GM qk_workspace(enable_relu 融合, cast to calc_dtype)
                            #   Vector 侧: GM qk_workspace → UB → row_expand_mul + reduce_sum → GM Scores

                            # v7.3 DMA 重排 (参考算子模式 3, HISA Wave 0):
                            # Wave 0: A_L1_0 + B_L1_0 优先 DMA (已在上面 L1 加载循环完成)
                            # Wave 1+: A_L1_1 + B_L1_1 与首个 GEMM (m0,n0) overlap
                            # 4 GEMM 并行 issue (硬件自动调度, 每个 GEMM 独占 L0C, 无 flag 争用)

                            # (m0, n0): A_L1_0 × B_L1_0^T → C_L0_0
                            T.gemm_v0(A_L1_0, B_L1_0, C_L0_0, transpose_B=True, init=True)
                            # (m1, n0): A_L1_1 × B_L1_0^T → C_L0_1
                            T.gemm_v0(A_L1_1, B_L1_0, C_L0_1, transpose_B=True, init=True)
                            # (m0, n1): A_L1_0 × B_L1_1^T → C_L0_2
                            T.gemm_v0(A_L1_0, B_L1_1, C_L0_2, transpose_B=True, init=True)
                            # (m1, n1): A_L1_1 × B_L1_1^T → C_L0_3
                            T.gemm_v0(A_L1_1, B_L1_1, C_L0_3, transpose_B=True, init=True)

                            # v7.3: 4 L0C→GM copies (M pipeline 可 overlap 下一轮 GEMM)
                            # enable_relu=True 融合 ReLU (L0C→GM 路径, 硬件保证精度)
                            T.copy(C_L0_0,
                                   qk_workspace[cid, s2_block % num_stages,
                                                m_l1 * M_L1 : m_l1 * M_L1 + M_L0,
                                                s2_l1 * S2_L1 : s2_l1 * S2_L1 + N_L0],
                                   enable_relu=True)
                            T.copy(C_L0_1,
                                   qk_workspace[cid, s2_block % num_stages,
                                                m_l1 * M_L1 + M_L0 : m_l1 * M_L1 + 2 * M_L0,
                                                s2_l1 * S2_L1 : s2_l1 * S2_L1 + N_L0],
                                   enable_relu=True)
                            T.copy(C_L0_2,
                                   qk_workspace[cid, s2_block % num_stages,
                                                m_l1 * M_L1 : m_l1 * M_L1 + M_L0,
                                                s2_l1 * S2_L1 + N_L0 : s2_l1 * S2_L1 + 2 * N_L0],
                                   enable_relu=True)
                            T.copy(C_L0_3,
                                   qk_workspace[cid, s2_block % num_stages,
                                                m_l1 * M_L1 + M_L0 : m_l1 * M_L1 + 2 * M_L0,
                                                s2_l1 * S2_L1 + N_L0 : s2_l1 * S2_L1 + 2 * N_L0],
                                   enable_relu=True)

                    # ============ AIV 侧 (编译器自动插 wait_cross_flag) ============

                    # v5: groupInner=16 分块 (Q20 详述)
                    T.tile.fill(scores_accum, 0)  # 初始化累加器

                    for s1_inner in T.serial(s1BaseSize):
                        for g_idx in T.serial(T.ceildiv(G, groupInner)):
                            # 从 qk_workspace 读取 [groupInner, BLOCK_N] 切片
                            m_start = s1_inner * G + g_idx * groupInner
                            T.copy(qk_workspace[cid, s2_block % num_stages,
                                                m_start:m_start + groupInner, :], qk_ub)

                            # 读取权重 [groupInner]
                            # v5: TND 直消 Weights
                            if layout_query == "TND":
                                w_offset = t_start_q + s1_tile * s1BaseSize + s1_inner
                                T.copy(Weights[w_offset, g_idx * groupInner:(g_idx + 1) * groupInner],
                                       weight_ub)
                            else:
                                T.copy(Weights[b, s1_tile * s1BaseSize + s1_inner,
                                               g_idx * groupInner:(g_idx + 1) * groupInner],
                                       weight_ub)

                            # v6: row_expand_mul_experiment 融合 Brcb+Mul (优化 #1)
                            # 替代 v5: T.tile.broadcast(weight_2d, weight_ub, axis=1) + T.tile.mul(qk_ub, qk_ub, weight_2d)
                            # dst[i,j] = src0[i,j] * src1[i], src1=[groupInner] 行广播
                            # AscendC 后端: brcb(src1→tmp) + mul_mask(dst, src0, tmp)
                            # PTO 后端: TROWEXPANDMUL_row_vec(dst, src0, src1)
                            #
                            # v7.3: brcb_experiment + row_expand_mul_experiment 三步模式 (参考算子模式 7, xattention 行 891-916)
                            # ⚠️ Stage 2 验证: 是否在 row_expand_mul_experiment 前加 brcb_experiment 预处理
                            #   - 若 weight_ub 需要多次复用 (如多个 o_chunk), 先 brcb 到 shared_broadcast_buf 再 row_expand_mul
                            #   - 三步模式: scalar → brcb_experiment → lane broadcast → row_expand_mul_experiment per chunk
                            #   - 预期收益: 减少 pipe_barrier 数量 (xattention 验证)
                            #   - 当前 weight_ub=[groupInner]=[16] 已是 1D, row_expand_mul_experiment 内部 Brcb 广播
                            #   - v7.3 保留当前单次 row_expand_mul_experiment, brcb 前缀作为 Stage 2 优化候选
                            T.tile.row_expand_mul_experiment(qk_ub, qk_ub, weight_ub)  # [16,512] = [16,512] * [16]

                            # v6: reduce_sum(clear=False) 融合 reduce+累加 (优化 #2)
                            # 替代 v5: T.reduce_sum(qk_ub, scores_partial, dim=0) + T.tile.add(scores_accum, scores_accum, scores_partial)
                            # clear=False: new_out = old_out + reduced_result (merge 语义)
                            #
                            # v7.3: reduce_sum(dim=0) 硬件归约 (参考算子模式 8, HISA 行 343)
                            # - clear=False (v6 已用): 累加场景, new_out = old_out + reduced_result
                            # - clear=True (HISA 变体): 单次归约场景, new_out = reduced_result (清空旧值)
                            # - lightning_indexer 用 clear=False (累加多个 s1_inner/g_idx 的 scores), 正确
                            # - HISA 的 clear=True 用于单次 logits→s 归约, 不适用于本场景
                            T.reduce_sum(qk_ub, scores_accum, dim=0, clear=False)  # [16,512]→[512], 累加

                            # v6: Vector 管线轻量同步 (优化 #4, v7.1 已验证)
                            # 替代部分 T.barrier_all() — 仅等 Vector 管线, 不等 Cube/MTE2/MTE3
                            # v7.1 验证结论 (验证 5 PASS): pipe_barrier("v") 精度与 barrier_all 一致 (diff=0.0312)
                            # 注意: 跨管线同步 (Cube→Vector) 仍需 barrier_all 或 set_flag/wait_flag
                            T.pipe_barrier("v")

                    # Write scores to GM (split-N, 各 s2_block 写不同位置)
                    T.copy(scores_accum, Scores[b, 0, s1_tile, s2_start:s2_start + BLOCK_N])
```

**v6 关键改进对比**：

| 改进项 | v4 写法 | v5 写法 | v6 写法 | 修正原因 |
|--------|---------|---------|---------|---------|
| **Vector qk_ub** | [BLOCK_M, BLOCK_N]=[512,512]=1MB | [groupInner, BLOCK_N]=[16,512]=32KB | 同 v5 (row_expand_mul dst=src0) | v4 UB 超限 (缺陷 #1) |
| **Vector weight_2d** | [BLOCK_M, BLOCK_N]=[512,512]=1MB | [groupInner, BLOCK_N]=[16,512]=32KB | **v6 移除** (row_expand_mul 融合 Brcb) | v6 指令融合 (优化 #1) |
| **Vector scores_partial** | 无 (v3 逐行 mul 直接 reduce) | [BLOCK_N]=2KB | **v6 移除** (reduce_sum clear=False 融合) | v6 指令融合 (优化 #2) |
| **Vector weight mul** | 逐行 for loop mul | broadcast + mul (2 步) | **row_expand_mul_experiment** (1 步, 融合 Brcb+Mul) | v6 指令融合 (优化 #1) |
| **Vector reduce+累加** | reduce_sum (单次) | reduce_sum + add (2 步) | **reduce_sum(clear=False)** (1 步, merge 语义) | v6 指令融合 (优化 #2) |
| **Vector 指令数/iter** | 2+ (逐行) | 6 (copy×2+broadcast+mul+reduce+add) | **4** (copy×2+row_expand_mul+reduce_sum) | v6 -33% 指令数 |
| **Vector UB 总量** | ~1MB+ | 68KB | **34KB** | v6 -50% UB |
| **Cube C_L0** | [BLOCK_M, BLOCK_N]=[512,512]=1MB | [M_L0, N_L0]=[128,128]=64KB | 同 v5 | v4 L0C 超限 (缺陷 #2) |
| **Cube GEMM** | 单次 T.gemm_v0 | m_l0 × n_l0 次循环, T.gemm_v0+切片 | 同 v5 | v5 L0 inner split |
| **Cube GEMM (v7.1)** | — | — | **独立 L1 buffer, 2×2 展开, 无切片** | **v7.1: transpose_B+切片不可行 (验证 3), 改独立 buffer** |
| **核数** | 硬编码 24 | 硬编码 24 | **动态 cube_core_num** | v6 适配 A2/A3/950 (优化 #3) |
| **Vector 内同步** | barrier_all | barrier_all (AUTO_SYNC) | **pipe_barrier("v")** (Vector 管线, v7.1 已验证) | v6 轻量同步 (优化 #4) |
| 核间流水 | T.Pipelined(num_stages=2) | 同 (Stage 2 尝试 3-4) | 同 (v7.1: num_stages=3-4 已验证) | 保留 |
| Core 分配 | T.Kernel(NUM_CORES=24) | T.Kernel(min(total_tasks,24)) | T.Kernel(min(total_tasks,动态核数)) | v5 缺陷 #5 + v6 动态核数 |
| TND 处理 | kernel 直接 strided DMA | 同 (act_seq 标量读取标注 Stage 2 验证) | 同 | 保留 + 风险标注 |
| workspace | [NUM_CORES, num_stages, BLOCK_M, BLOCK_N] | 同 (BLOCK_M,BLOCK_N 仍是全尺寸, GM 无容量限制) | 同 | 保留 |

> **⚠️ v6 Stage 2 验证清单**（v7.1 已验证项标注, 见 §12.8）：
> 1. ~~`T.gemm_v0(Q_L1[m_l0*M_L0:(m_l0+1)*M_L0, :], ...)` — 2D L1 行切片~~ → **v7.1 验证 (验证 3): transpose_B=True + L1 行切片 → 精度错误 (diff=66.5), 不可行! 改用独立 L1 buffer (§8.5)**
> 2. `T.copy(C_L0, qk_workspace[..., m_slice, :], enable_relu=True)` — L0C→GM 部分写入 + enable_relu — **待 Stage 2 验证**
> 3. **v6 新增**: `T.tile.row_expand_mul_experiment(qk_ub, qk_ub, weight_ub)` — dst=src0 原地乘, src1 1D 行广播 — **待 Stage 2 验证**
> 4. **v6 新增**: `T.reduce_sum(qk_ub, scores_accum, dim=0, clear=False)` — merge 语义, 直接累加到 scores_accum — **待 Stage 2 验证**
> 5. ~~**v6 新增**: `T.pipe_barrier("v")` — Vector 管线轻量同步~~ → **v7.1 验证 (验证 5 PASS): 精度与 barrier_all 一致 (diff=0.0312) ✅**
> 6. **v6 新增**: `torch.npu.get_device_properties("npu").cube_core_num` — host 侧动态核数获取 — **待 Stage 2 验证**
> 7. `T.tile.fill(scores_accum, 0)` — UB buffer 填充 — **待 Stage 2 验证**
> 8. `actual_seq_q[b - 1]` — GM tensor 标量读取（缺陷 #6）— **待 Stage 2 验证**
> 9. ~~fallback：若 2D 行切片不可用，Q_L1 改为 [M_L0, D]，外层循环多次 GM→L1 加载~~ → **v7.1 已采纳此思路: 独立 buffer A_L1_0/A_L1_1 各 [M_L0, D] ✅**

### 8.6 Vector 核计算流程（Kernel 2, 纯 Vector）

见 §3.3 Kernel 2 伪代码（v2 保留，v3 仅更新 mask 策略见 §10.2 Q3）。

### 8.7 注意事项

- **AUTO_CV_COMBINE 的 GEMM 识别**：编译器通过 `T.gemm_v0` 调用识别 AIC 代码段。确保 GEMM 的输入是 `alloc_shared`（L1）或 `alloc_fragment`（L0C），输出是 `alloc_fragment`（L0C）
- **L0C → UB 中转**：AscendC 硬件限制 UB 与 L1/L0C 不能直通，必须经 GM workspace。AUTO_CV_COMBINE 自动处理此中转
- **ReLU 融合**：v2 用 `T.copy(C_L0, qk_workspace, enable_relu=True)`。v3 Developer 模式下，若 `enable_relu` 在 L0C→UB 路径不可用，改用 `T.copy(C_L0, qk_ub)` 后 `T.tile.max(qk_ub, qk_ub, 0)` 或 `T.cast` + 比较
- **workspace_idx**: [6] (qk_workspace 在 main 函数签名中的位置)
- **fallback**：若 AUTO_CV_COMBINE 编译失败，回退到 v2 Expert 模式（§3.3 Kernel 1 Expert 代码保留）

### 8.8 sort 缓存优化设计（v7 新增，基于 AscendC arch22 `service_vector.h:372-407`）

> **v7 新增**：msprof 实测 BSND_BSND Vec0 利用率 48.4% vs AscendC 63.1%（低 14.7%），**sort 缓存优化缺失是次要瓶颈**。本节设计 S1≤4 场景的 SortedBasicBlock 缓存机制，减少 75% merge_sort 调用。
>
> **v7.2 关键修正（缺口 2 + 缺口 6）**：
> - **缺口 2 (UB 超限)**: v7.1 原方案 UB=216.4KB > 192KB ❌ 超限。v7.2 采用**方案 C (VID_S1=1 + buffer 复用)**: UB=184.4KB ✓
> - **缺口 6 (SparseTopK 截断)**: AscendC 用 `DataCopy(dst, tmp, topk * 2)` 显式截断到 BASE_TOPK=2048。v7.2 在 4-way merge_sort 后增加 `T.copy(merge_output[0:BASE_TOPK*2], global_topk_ub)` 显式截断, 防止 merge_sort 输出超出 global_topk_ub 分配。

**AscendC sort 缓存优化机制（源码验证 `service_vector.h:372-407`）**：

```cpp
// AscendC: actS1Size <= 4 时启用缓存优化
if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
    // 标准路径: 每次 S2 块都 SortAll + MergeSort
    LIServiceVec::SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign);
    LIServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, ...);
} else {
    // 缓存优化路径: S1≤4 时缓存 4 块再 4-way merge
    int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;
    Sort<float, true>(
        SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2 + globalTopkUbCacheIdx * s2BaseSize_ * 2],
        reduceOutBuff, ...);

    // 缓存满 4 块或 S2 结束时, 进行 4-way 精排
    if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
        if (info.s2Idx - blockS2StartIdx_ < 4) {
            // 前 4 块直接精排覆盖到 globalTopkUb_
            MrgBasicBlock(globalTopkUb_[...], tt, globalTopkUbCacheIdx + 1, s2BaseSize_);
        } else {
            // 后面缓存的块先精排, 再 merge 到 globalTopkUb_
            MrgBasicBlock(tmpSortBuf, tt, globalTopkUbCacheIdx + 1, s2BaseSize_);
            SparseTopK(globalTopkUb_[...], SortedBasicBlock_[...], tmpSortBuf, BASE_TOPK, ...);
        }
    }
}
```

**v7 TileLang sort 缓存优化设计**：

**适用条件**：
- `actS1Size <= 4`（推理场景，最常见）
- `sparse_count <= 2048`（非 isSparseCountOver2K 场景）
- 当 `actS1Size > 4` 或 `isSparseCountOver2K` 时走标准路径（每次 sort + merge_sort）

**UB buffer 规划（v7.1 修正: 1D 平铺 + sort_tmp, API 验证 2）**：

> **v7.1 关键修正（API 验证 2 发现）**：
> - `T.tile.sort` **不支持 BufferRegion 切片**作为 dst（签名仅 `Buffer`）→ v7 的 `T.tile.sort(sorted_cache_ub[cache_idx], ...)` 不可行
> - `T.tile.merge_sort` **不存在 `way` 参数**（路数由非 None 源 buffer 个数决定）→ v7 的 `way=cache_idx` 不可行
> - **v7.1 方案**：sort 到临时 buffer `sort_tmp` → `T.copy` 到 1D 平铺 `sorted_cache` 偏移 → `merge_sort` 用切片 src（支持 BufferRegion）+ if-elif 分支处理动态路数

| Buffer | 大小 | 用途 |
|--------|------|------|
| `sorted_cache` | [SORT_CACHE_SIZE × S2_VEC_BLOCK × 2] × 4B = 4 × 512 × 2 × 4 = **16KB** (1D 平铺) | 缓存 4 个 S2 块的 sort 结果 (value, index 对, 1D 连续) |
| `sort_tmp` | [S2_VEC_BLOCK × 2] × 4B = 512 × 2 × 4 = **4KB** | sort 临时 buffer (T.tile.sort 的 dst, 完整 Buffer) |
| `merge_output` | [SORT_CACHE_SIZE × S2_VEC_BLOCK × 2] × 4B = **16KB** | 4-way merge 输出 buffer |
| `global_topk_ub` | [BASE_TOPK, 2] × 4B = 2048 × 2 × 4 = **16KB** | 全局 topk 结果 (跨 S2 块归并) |
| **v7.1 sort 缓存小计** | **52KB** | / 192KB UB (v7 原方案 48KB, +4KB sort_tmp) |

> **v7.2 方案 C 完整 UB 占用（缺口 2 验证, `test_memory_budget.py`）**：
>
> | Buffer | v7.1 原方案 | v7.2 方案 C (VID_S1=1) | 变化 |
> |--------|------------|----------------------|------|
> | topk_a_ub (per-row topk 累加器) | (VID_S1=2, TA2) = 32KB | **(VID_S1=1, TA2) = 16KB** | -16KB |
> | sorted_cache | 16KB | 16KB (不变) | — |
> | sort_tmp | 4KB (独立) | **0KB (复用 cache_tmp_ub)** | -4KB |
> | merge_output | 16KB (独立) | **0KB (复用 merged_ub)** | -16KB |
> | global_topk_ub | 16KB | 16KB (不变) | — |
> | 其他 v6 buffer (qk_ub, weight_ub, scores_accum 等) | ~100KB | ~100KB | — |
> | **总计** | **216.4KB ❌ 超限** | **184.4KB ✓** | **-32KB** |
>
> **方案 C 关键调整**:
> - `VID_S1` 从 2 改为 1（仅 S1≤4 / S1_BLOCK=4 场景）: 每个 AIV 处理 1 行 S1 而非 2 行, 多 1 轮循环
> - `sort_tmp` 复用 v6 已有的 `cache_tmp_ub` (避免新增 4KB)
> - `merge_output` 复用 v6 已有的 `merged_ub` (避免新增 16KB)
> - 新增 `sorted_cache` 16KB (无法复用, 1D 平铺独立 buffer)
> - 净变化: +16KB (sorted_cache) - 16KB (topk_a_ub 缩小) = 0KB, UB 保持 184.4KB
>
> **性能影响分析（结合 Excel 37 个用例数据范围）**:
> - S1≤4 的用例占 46% (17/37) → 缓存路径, VID_S1=1 多 1 轮 S1 循环, 性能影响 <0.5%
> - S1>4 的用例占 54% (20/37) → 标准路径 (每次 sort + merge_sort), 不受 VID_S1 影响
> - AscendC arch22 `service_vector.h:372-407` 也是 `innerS1Idx` 循环串行处理 S1 行, 行为一致
> - `VID_S1` 是 buffer 容量维度, 不是并行度; sort/merge 都是逐行串行执行

> **v7.2 SparseTopK 截断（缺口 6 验证）**：
> - AscendC 用 `DataCopy(dst, tmp, topk * 2)` 显式截断到 BASE_TOPK=2048
> - v7.1 的 `T.tile.merge_sort(global_topk_ub, global_topk_ub, merge_output)` 可能不截断
> - **风险**: merge_sort 输出 = 输入大小之和 (global_topk_ub + merge_output), 可能超出 global_topk_ub 的 16KB 分配
> - **v7.2 方案**: merge_sort 后用 `T.copy(merge_output[0:BASE_TOPK*2], global_topk_ub)` 显式截断（见伪代码步骤 6）

> **v7.1 UB 容量验证**：v6 Kernel 1 UB=34KB + v7.1 sort 缓存 52KB = 86KB < 192KB ✓（余量 106KB）
>
> **注意**：sort 缓存优化在 **Kernel 2**（topk）中实施，不影响 Kernel 1 的 UB 占用。Kernel 2 的 UB 预算独立于 Kernel 1。

**v7.1 sort 缓存优化伪代码（Kernel 2 内, API 验证 2 修正）**：

```python
# v7.1: sort 缓存优化 (对齐 AscendC SortedBasicBlock 机制)
# 仅 actS1Size <= 4 且 sparse_count <= 2048 时启用
# v7.1 修正: sort→临时 buffer→T.copy 到 1D cache→merge_sort 用切片 src (验证 2)

SORT_CACHE_SIZE = 4         # 缓存 4 个 S2 块
S2_VEC_BLOCK = 512          # S2 分块大小 (Vector 侧)
BASE_TOPK = 2048            # 全局 topk 容量
SORT_BUF_SIZE = S2_VEC_BLOCK * 2  # sort 输出 = 2×输入 (value, index 对, 降序)

# v7.1: UB buffer (Kernel 2 内, 独立于 Kernel 1)
# v7.2 方案 C (缺口 2): VID_S1=1, sort_tmp 复用 cache_tmp_ub, merge_output 复用 merged_ub
# 1D 平铺 cache (T.tile.sort 不支持切片 dst, 用 T.copy 搬运到偏移)
VID_S1 = 1  # v7.2 方案 C: 每个 AIV 处理 1 行 S1 (原 v7.1: VID_S1=2), 缩小 topk_a_ub 16KB
sorted_cache = T.alloc_ub([SORT_CACHE_SIZE * SORT_BUF_SIZE], calc_dtype)  # [4×1024] = 16KB (1D, 新增)
sort_tmp = cache_tmp_ub  # v7.2 方案 C: 复用 v6 已有的 cache_tmp_ub (避免新增 4KB)
merge_output = merged_ub  # v7.2 方案 C: 复用 v6 已有的 merged_ub (避免新增 16KB)
global_topk_ub = T.alloc_ub([BASE_TOPK * 2], calc_dtype)                 # [4096] = 16KB
topk_a_ub = T.alloc_ub([VID_S1, _TA2], calc_dtype)                       # v7.2: [1, TA2] = 16KB (原 v7.1: [2, TA2] = 32KB, -16KB)

# v7.1: sort 缓存优化主循环
cache_idx = 0  # 当前缓存位置 (0-3)

for s2_block in T.serial(s2BaseNum):
    s2_start = s2_block * S2_VEC_BLOCK
    s2_valid = T.min(S2_VEC_BLOCK, act_k - s2_start)

    # 1. 加载当前 S2 块的 scores
    T.copy(Scores[b, 0, s1_idx, s2_start : s2_start + s2_valid], score_buf)

    # 2. mask (sparse_mode=3)
    if sparse_mode == 3:
        # apply rightDownCausal mask
        cutoff = act_k - act_q + s1_idx + 1
        # mask 位置 >= cutoff 的 score 设为 -inf
        ...

    # 2.5 v7.4: sort 前将无效 K 元素 score 设为 -inf（Phase 1 精度根因修复）
    # 对齐 AscendC service_vector.h:357-368: Duplicate(-inf) → Adds(有效元素) → Duplicate(-1) → Adds(有效index)
    # 解决 PA_BSND 输出 -1 无效 index 的 bug（cu_s2_len < S2_VEC_BLOCK 时尾部未填 -inf → 被选入 top-k）
    #
    # 计算当前 S2 块的有效元素数 cu_s2_len（考虑 causal mask + 尾块）
    if sparse_mode == 3:  # rightDownCausal: 有效 key 数受 act_q - s1_idx 约束
        cu_real_ac_seq = act_k[b] - (act_q[b] - s1_idx)  # 对齐 AscendC cuRealAcSeq
    else:
        cu_real_ac_seq = act_k[b]
    cu_s2_len = T.min(cu_real_ac_seq - s2_start, S2_VEC_BLOCK)  # 当前块有效元素数

    # 仅当尾块有效元素 < S2_VEC_BLOCK 时才需要填充（整块有效时跳过, 节省指令）
    if cu_s2_len < S2_VEC_BLOCK:
        # 无效 score 区间填 -inf（sort 后沉到底部, 不会被选入 top-k）
        T.tile.fill(score_buf[cu_s2_len:S2_VEC_BLOCK], -T.infinity(calc_dtype))
        # 无效 index 区间填 -1（对齐 AscendC Duplicate(sortIndiceUbInt, -1, ...)）
        T.tile.fill(index_blk_ub[cu_s2_len:S2_VEC_BLOCK], -1)
        T.pipe_barrier("v")  # 确保 fill 完成后再 sort

    # 3. v7.1: sort 到临时 buffer (T.tile.sort 不支持切片 dst, 必须用完整 Buffer)
    # 对齐 AscendC Sort<float, true>
    T.tile.sort(sort_tmp, score_buf, S2_VEC_BLOCK)

    # 4. v7.1: T.copy 从 sort_tmp 搬运到 1D sorted_cache 偏移
    T.copy(sort_tmp, sorted_cache[cache_idx * SORT_BUF_SIZE : (cache_idx + 1) * SORT_BUF_SIZE])

    cache_idx += 1

    # 5. v7.1: 缓存满 4 块或 S2 结束时, 进行 merge (对齐 AscendC MrgBasicBlock)
    # merge_sort 支持 BufferRegion 切片作为 src (验证 1 确认)
    # 动态路数用 if-elif 分支 (不存在 way 参数, 验证 1 确认)
    is_cache_full = (cache_idx == SORT_CACHE_SIZE)
    is_s2_end = (s2_block == s2BaseNum - 1)

    if is_cache_full or is_s2_end:
        if cache_idx == 4:
            # 4-way merge: 4 个切片 src → merge_output
            T.tile.merge_sort(
                merge_output,
                sorted_cache[0 * SORT_BUF_SIZE : 1 * SORT_BUF_SIZE],
                sorted_cache[1 * SORT_BUF_SIZE : 2 * SORT_BUF_SIZE],
                sorted_cache[2 * SORT_BUF_SIZE : 3 * SORT_BUF_SIZE],
                sorted_cache[3 * SORT_BUF_SIZE : 4 * SORT_BUF_SIZE],
            )
        elif cache_idx == 3:
            # 3-way merge: 3 个切片 src → merge_output
            T.tile.merge_sort(
                merge_output,
                sorted_cache[0 * SORT_BUF_SIZE : 1 * SORT_BUF_SIZE],
                sorted_cache[1 * SORT_BUF_SIZE : 2 * SORT_BUF_SIZE],
                sorted_cache[2 * SORT_BUF_SIZE : 3 * SORT_BUF_SIZE],
            )
        elif cache_idx == 2:
            # 2-way merge: 2 个切片 src → merge_output
            T.tile.merge_sort(
                merge_output,
                sorted_cache[0 * SORT_BUF_SIZE : 1 * SORT_BUF_SIZE],
                sorted_cache[1 * SORT_BUF_SIZE : 2 * SORT_BUF_SIZE],
            )
        elif cache_idx == 1:
            # 仅 1 块: 直接复制
            T.copy(sorted_cache[0 : SORT_BUF_SIZE], merge_output)

        # 6. merge_output 与 global_topk_ub 归并 (非首次)
        if s2_block < SORT_CACHE_SIZE:
            # 前 4 块: merge_output 直接作为 global_topk_ub
            # v7.2 缺口 6: 显式截断到 BASE_TOPK (AscendC SparseTopK 行为)
            T.copy(merge_output[0 : BASE_TOPK * 2], global_topk_ub)
        else:
            # 后续块: global_topk_ub + merge_output → global_topk_ub (2-way merge, 取前 BASE_TOPK)
            # v7.2 缺口 6: merge_sort 后必须显式截断, 防止输出 = 输入之和超出 global_topk_ub 分配
            T.tile.merge_sort(merge_output, global_topk_ub, merge_output)
            T.copy(merge_output[0 : BASE_TOPK * 2], global_topk_ub)  # v7.2: SparseTopK 显式截断

        cache_idx = 0  # 重置缓存

# 7. 最终 topk: 从 global_topk_ub 提取前 sparse_count 个
T.tile.topk(final_topk, global_topk_ub, sparse_count, BASE_TOPK)
```

> **v7.1 关键修正点**：
> 1. `T.tile.sort(sort_tmp, score_buf, ...)` — sort 到**完整 Buffer** `sort_tmp`（不支持切片 dst, 验证 2）
> 2. `T.copy(sort_tmp, sorted_cache[cache_idx * SORT_BUF_SIZE : ...])` — T.copy 搬运到 1D cache 偏移
> 3. `T.tile.merge_sort(dst, src0, src1, src2, src3)` — **位置参数**指定 4-way（不存在 `way` 参数, 验证 1）
> 4. `sorted_cache[0 * SORT_BUF_SIZE : 1 * SORT_BUF_SIZE]` — **切片作为 src** 可行（merge_sort 支持 BufferRegion, 验证 1）
> 5. 动态路数（1/2/3/4）用 **if-elif 分支**显式调用（num_ways 是编译期常量, 验证 1）
>
> **v7.2 关键修正点（缺口 2 + 缺口 6）**：
> 6. **VID_S1=1** (方案 C, 缺口 2): `topk_a_ub = T.alloc_ub([VID_S1=1, _TA2], ...)` — 缩小 16KB, UB 从 216.4KB 降到 184.4KB ✓
> 7. **buffer 复用** (方案 C, 缺口 2): `sort_tmp = cache_tmp_ub`, `merge_output = merged_ub` — 复用 v6 已有 buffer, 避免新增 20KB
> 8. **SparseTopK 显式截断** (缺口 6): `T.copy(merge_output[0:BASE_TOPK*2], global_topk_ub)` — merge_sort 后显式截断到 BASE_TOPK=2048, 防止输出超出 global_topk_ub 分配
> 9. **2-way merge_sort 改用 merge_output 作 dst** (缺口 6): `T.tile.merge_sort(merge_output, global_topk_ub, merge_output)` — 不能用 global_topk_ub 作 dst (输入输出同 buffer 会冲突), 改用 merge_output 作 dst 再截断复制
>
> **v7.4 关键修正点（Phase 1 精度根因, 见 §12.12）**：
> 12. **sort 前 -inf 填充** (Phase 1): sort 前对 `score_buf[cu_s2_len:S2_VEC_BLOCK]` 填 `-inf`, 对 `index_blk_ub[cu_s2_len:S2_VEC_BLOCK]` 填 `-1`. 对齐 AscendC `service_vector.h:357-368` 的 `Duplicate(-inf)→Adds(有效)→Duplicate(-1)→Adds(有效index)` 四步. **解决 PA_BSND li_default_a2 输出 -1 无效 index 的 bug**（cu_s2_len < S2_VEC_BLOCK 时尾部未填 -inf, 被错误选入 top-k）.
> 13. **cu_s2_len 动态计算** (Phase 1): `cu_s2_len = min(cu_real_ac_seq - s2_start, S2_VEC_BLOCK)`, 其中 `cu_real_ac_seq = (sparse_mode==3) ? act_k[b]-(act_q[b]-s1_idx) : act_k[b]`. 对齐 AscendC `cuRealAcSeq` (考虑 causal mask). 仅 `cu_s2_len < S2_VEC_BLOCK` 时才填充（整块有效跳过, 节省指令）.
> 14. **同时处理 TOP_K > S2** (Phase 3 边界 Case 4): 当 sparse_count > 有效 S2 时, 无效位置的 -inf score 经 sort 沉底后不被选入 top-k, 输出区自然由 topk 的 `T.tile.select(..., -1.0, "VSEL_TENSOR_SCALAR_MODE")` 填 -1. 需确认 `topk_a_ub` 初始化为 -inf（见 §10.2 Q27 修复方案）.
>
> **v7.3 关键更新（参考算子模式 6, MTE2∥V overlap, HISA 行 357-370）**：
> 10. **MTE2∥V overlap** — sort 期间 enqueue 下一个 S2 块的 GM→UB DMA，让 MTE2 与 V 并行
>     - 当前 v7.2: sort → T.copy(下一个 S2 块) → sort (串行, V 空闲等 MTE2)
>     - v7.3 改进: T.copy(下一个 S2 块, late DMA 入队) → set_flag("MTE2","V") → sort(当前块, early output) → wait_flag → 使用下一个块
>     - 预期收益: Vector 利用率提升 ≥ 5%（HISA 验证 MTE2∥V overlap 显著减少 V 空闲）
>     - ⚠️ Stage 2 验证: flag 时序正确性, 防止 sort 使用未加载完成的 late DMA 数据
>     - 伪代码示意（Stage 2 实施时整合到主循环）:
>       ```python
>       # late DMA: 下一个 S2 块的 scores 入队 (不立即等)
>       if s2_block + 1 < s2BaseNum:
>           next_s2_start = (s2_block + 1) * S2_VEC_BLOCK
>           T.copy(Scores[b, 0, s1_idx, next_s2_start : ...], score_buf_next)
>           T.set_flag("MTE2", "V", SIG_LATE_DMA)
>       # early output: 当前块的 sort + merge (与 late DMA 并行)
>       T.tile.sort(sort_tmp, score_buf, S2_VEC_BLOCK)
>       T.copy(sort_tmp, sorted_cache[...])
>       # ... merge_sort ...
>       # late compute: 等下一个块 DMA 完成后再使用
>       if s2_block + 1 < s2BaseNum:
>           T.wait_flag("MTE2", "V", SIG_LATE_DMA)
>           # swap score_buf ↔ score_buf_next (double buffer)
>       ```
> 11. **tail-fill 替代 mask** (参考算子模式 9, HISA 行 346-356) — 仅适用于尾块边界
>     - 当前 v7.2: 用 compare + select mask 处理 sparse_mode=3 causal mask
>     - v7.3 改进: 尾块边界（s2_valid < S2_VEC_BLOCK）用标量循环填 -inf, 减少 compare+select 指令数
>     - causal mask (sparse_mode=3) 仍用 compare+select (非尾块场景)
>     - ⚠️ Stage 2 验证: tail-fill 与 sort 的兼容性 (sort 要求连续 buffer)

**v7 sort 缓存优化的收益分析**：

| 场景 | v6 merge_sort 调用次数 | v7 merge_sort 调用次数 | 减少 |
|------|----------------------|----------------------|------|
| S2=3072, S2_VEC_BLOCK=512 (6 块) | 6 次（每块 1 次） | 2 次（4+2 块各 1 次 4-way merge） | **67%** |
| S2=2048, S2_VEC_BLOCK=512 (4 块) | 4 次 | 1 次（4 块 1 次 4-way merge） | **75%** |
| S2=8192, S2_VEC_BLOCK=512 (16 块) | 16 次 | 4 次（4×4 块各 1 次） | **75%** |

> **注意**：v7 sort 缓存优化仅适用于 `actS1Size <= 4` 场景（推理场景，最常见）。当 `actS1Size > 4` 时（训练场景），仍走 v6 的标准 sort + merge_sort 路径。

**v7.1 sort 缓存优化的 API 验证（已验证, 见 §12.8）**：

| API | 源码位置 | 使用验证 | 说明 |
|-----|---------|---------|------|
| `T.tile.sort` | `ascend_tile.py` | `api_tests/test_sort_cache_1d.py` ✅ | sort 到**完整 Buffer**（不支持切片 dst）, v7.1 用 sort_tmp 中转 |
| `T.tile.merge_sort` | `ascend_tile.py:320` | `api_tests/test_merge_sort_4way.py` ✅ | 2/3/4-way merge, **位置参数**指定路数（不存在 `way` 参数）, 支持 BufferRegion 切片 src |

> **⚠️ v7 Stage 2 验证项**（v7.1 已验证, 见 §12.8 API 验证结果）：
> 1. ~~`T.tile.merge_sort` 的 4-way merge 是否支持 `way=4` 参数~~ → **v7.1 验证结论：不存在 `way` 参数！** 4-way 通过位置参数 `merge_sort(dst, s0, s1, s2, s3)` 调用（验证 1 PASS）
> 2. ~~`sorted_cache_ub[cache_idx]` 的 3D buffer 切片是否被 T.tile.sort 支持~~ → **v7.1 验证结论：T.tile.sort 不支持切片 dst！** 改用 sort→sort_tmp→T.copy 到 1D cache 偏移（验证 2 PASS）
> 3. ~~缓存不满 4 块时的 `way=cache_idx`（如 way=2, way=3）是否正确处理~~ → **v7.1 方案：if-elif 分支显式调用 2/3/4-way merge_sort**（num_ways 是编译期常量）
> 4. fallback：若 4-way merge 不可用，退回 2-way merge — **v7.1 已内置 if-elif 分支, 无需 fallback**

---

## 9. 验证方案

### 9.1 Golden 函数

基于 `/home/tilelang-ascend/li_pytest/lightning_indexer_golden.py` 的 `GeneralizedLI` 类，核心计算逻辑 `cal_atten_per_batch_b16`：

```python
def golden_lightning_indexer(query, key, weights, actual_seq_q, actual_seq_k,
                              block_table, layout_query, layout_key,
                              sparse_count, sparse_mode):
    """基于 PyTorch 的参考实现 (简化版, 完整版见 lightning_indexer_golden.py)"""
    B, S1, N1, D = query.shape
    N2 = 1
    G = N1 // N2

    # 1. QK BMM: [B, N1, S1, D] × [B, N2, D, S2] → [B, N1, S1, S2]
    #    K 在 N2 维广播到 N1
    qk = torch.einsum("bsmd,btnd->bmnt", query.float(), key.float())  # [B, N1, S1, S2]

    # 2. ReLU
    qk_relu = qk.clamp_min(0.0)

    # 3. Weight mul: [B, N1, S1, S2] × [B, S1, N1, 1] → [B, N1, S1, S2]
    weighted = qk_relu * weights.float().permute(0, 2, 1).unsqueeze(-1)

    # 4. Group reduce: sum over N1 (=g*n2) → [B, N2, S1, S2]
    scores = weighted.reshape(B, N2, G, S1, S2).sum(dim=2)

    # 5. Mask (sparse_mode=3: rightDownCausal)
    if sparse_mode == 3:
        for b in range(B):
            act_q_b = actual_seq_q[b]
            act_k_b = actual_seq_k[b]
            for i in range(act_q_b):
                cutoff = act_k_b - act_q_b + i + 1
                if cutoff < 0:
                    scores[b, :, i, :] = float('-inf')
                else:
                    scores[b, :, i, int(cutoff):] = float('-inf')

    # 6. TopK (stable sort, descending)
    topk_vals, topk_indices = torch.sort(-scores, dim=-1, stable=True)
    topk_indices = topk_indices[..., :sparse_count]

    # Fill invalid positions with -1
    for b in range(B):
        act_q_b = actual_seq_q[b]
        act_k_b = actual_seq_k[b]
        valid_count = min(act_k_b, sparse_count)
        topk_indices[b, :, act_q_b:, :] = -1
        topk_indices[b, :, :act_q_b, valid_count:] = -1

    # Output: [B, S1, N2, sparse_count]
    return topk_indices.permute(0, 2, 1, 3).to(torch.int32)
```

### 9.2 L0 门槛测试计划

> 由 `tilelang-op-test-design`（场景 A）生成。仅 L0 门槛用例（规则 shape, block 整除），供 Stage 2 快速精度收敛。L1/L2/Boundary 由 Stage 2 在 L0 通过后扩展。
>
> **用例选型依据**：从完整测试用例集（37 例, `/home/tilelang-ascend/li_pytest/excel/test_cases.xlsx`）中选取 15 个代表性场景，覆盖 4 种 layout 组合、2 种 dtype、大 S2 分段路径、大 S1、大 B、任意 sparse_count、特殊用例。

**L0 测试用例（15 个）**：

| # | 用例名 | B | S1 | S2 | N1 | block_size | dtype | layout_q | layout_k | sparse_count | sparse_mode | act_q | act_k | return_value | topk 路径 | 说明 |
|---|--------|---|----|----|----|-----------|-------|----------|----------|-------------|-------------|-------|-------|-------------|----------|------|
| 1 | l0_li_default_a2_bf16 | 18 | 3 | 3072 | 16 | 128 | bfloat16 | BSND | PA_BSND | 2048 | 3 | [3]*18 | [3016]*17+[3072] | False | 单次 | 默认 A2 配置 (li_default_a2) |
| 2 | l0_li_default_a2_fp16 | 18 | 3 | 3072 | 16 | 128 | float16 | BSND | PA_BSND | 2048 | 3 | [3]*18 | [3016]*17+[3072] | False | 单次 | FP16 变体 |
| 3 | l0_li_small_bf16 | 1 | 1 | 256 | 8 | 128 | bfloat16 | BSND | PA_BSND | 128 | 3 | [1] | [256] | False | 单次 | 最小 shape, N1=8 |
| 4 | l0_li_small_fp16_nomask | 1 | 1 | 256 | 8 | 128 | float16 | BSND | PA_BSND | 128 | 0 | [1] | [256] | False | 单次 | sparse_mode=0 (无 mask) |
| 5 | l0_li_bsnd_bsnd | 2 | 4 | 512 | 16 | 128 | bfloat16 | BSND | BSND | 256 | 3 | [4,4] | [512,512] | False | 单次 | BSND_BSND layout |
| 6 | l0_li_tnd_tnd | 2 | 4 | 3072 | 24 | 256 | float16 | TND | TND | 2048 | 0 | [4,8] (前缀和) | [0,6144] (前缀和) | True | 单次 | **TND_TND layout** (act_k=0 特殊用例) |
| 7 | l0_li_tnd_pa_bsnd | 2 | 2048 | 8192 | 8 | 128 | bfloat16 | TND | PA_BSND | 2048 | 0 | [26,52] (前缀和) | [8192,8192] | False | 单次 | **TND_PA_BSND 混合 layout** |
| 8 | l0_li_large_s2_131072 | 1 | 2 | 131072 | 16 | 128 | float16 | BSND | PA_BSND | 2048 | 0 | [2] | [131072] | False | **分段** | **大 S2=131072 分段 topk** |
| 9 | l0_li_large_s2_32768 | 2 | 3 | 32768 | 64 | 128 | float16 | BSND | BSND | 2048 | 3 | [2,3] | [2,3] | True | **分段** | 大 S2=32768, N1=64, return_value |
| 10 | l0_li_large_s1_8192 | 2 | 8192 | 8192 | 8 | 256 | float16 | BSND | BSND | 2048 | 0 | [3250,1295] | [8192,8192] | True | 单次 | **大 S1=8192 S1 切分** |
| 11 | l0_li_large_b_48 | 48 | 4 | 3072 | 24 | 128 | float16 | BSND | BSND | 1774 | 3 | [4]*48 | [3072]*48 | False | 单次 | **大 B=48**, 非幂次 sparse_count=1774 |
| 12 | l0_li_n1_64 | 16 | 5 | 3072 | 64 | 128 | float16 | BSND | BSND | 2048 | 3 | [5]*16 | [3072]*16 | False | 单次 | **N1=64 大 group** (G=64) |
| 13 | l0_li_sparse_count_153 | 2 | 1 | 2048 | 8 | 128 | float16 | BSND | PA_BSND | 153 | 3 | [1,1] | [2048,2048] | False | 单次 | **非幂次 sparse_count=153** |
| 14 | l0_li_block_size_768 | 2 | 13 | 7168 | 64 | 768 | float16 | BSND | BSND | 2048 | 3 | [13,13] | [7168,7168] | False | 单次 | **block_size=768** (G=64 需 inner split) |
| 15 | l0_li_return_value | 2 | 3 | 32768 | 64 | 256 | float16 | BSND | BSND | 2048 | 3 | [2,3] | [2,3] | True | **分段** | **return_value=True** + 大 S2 分段 |

**覆盖矩阵**：

| 覆盖维度 | 命中用例 # | 状态 |
|---------|-----------|------|
| layout BSND_BSND | 5, 9, 10, 11, 12, 14, 15 | ✓ |
| layout BSND_PA_BSND | 1, 2, 3, 4, 8, 13 | ✓ |
| layout TND_TND | 6 | ✓ |
| layout TND_PA_BSND | 7 | ✓ |
| dtype bfloat16 | 1, 3, 5, 7 | ✓ |
| dtype float16 | 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15 | ✓ |
| sparse_mode=0 | 4, 6, 7, 8, 10, 13 | ✓ |
| sparse_mode=3 | 1, 2, 3, 5, 9, 11, 12, 14, 15 | ✓ |
| 大 S2 分段路径 (>16384) | 8, 9, 15 | ✓ |
| 单次 topk 路径 (≤16384) | 1-7, 10-14 | ✓ |
| 非幂次 sparse_count | 11 (1774), 13 (153) | ✓ |
| block_size=768 | 14 | ✓ |
| block_size=256 | 6, 7, 10, 15 | ✓ |
| block_size=128 | 1-5, 8, 9, 11, 12, 13 | ✓ |
| N1=64 (G=64) | 9, 12, 14, 15 | ✓ |
| N1=8 | 3, 4, 7, 10, 13 | ✓ |
| return_value=True | 6, 9, 10, 15 | ✓ |
| act_k=0 特殊用例 | 6 | ✓ |
| 大 S1=8192 | 10 | ✓ |
| 大 B (≥48) | 11 | ✓ |

**L0 golden 草案**：
- 基于 `/home/tilelang-ascend/li_pytest/lightning_indexer_golden.py` 的 `GeneralizedLI` 类
- 测试调用流程：构造 query/key/weights/block_table/actual_seq → 调用 `GeneralizedLI.forward()` → 获取 CPU golden indices → 与 NPU kernel 输出对比
- **TND 用例**：act_seq 以**前缀和**形式传入（与 golden 的 `trans_tnd_actseq` 一致），host wrapper 负责转换

**L0 精度对比方法**（**非标准逐元素对比**，基于 `result_compare_method.py` 的 `check_result` 函数）：

1. **排序比较**：对 CPU 和 NPU 输出的每行（per b, s1, n2）分别排序后比较集合
2. **集合相同**：直接通过（无需值比较）
3. **集合不同**：对差异元素，查找对应的 `topk_value`（注意力分数），检查相对误差：
   - `npu_re = |npu_value - benchmark_value| / |benchmark_value|` < thres (0.0001)
   - `cpu_re = |cpu_value - benchmark_value| / |benchmark_value|` < thres (0.0001)
   - 两者均 < thres → 通过；否则失败
4. **最终判定**：所有差异行均通过值误差检查 → 整体通过

> **注意**：此方法容忍 TopK 稳定性差异（相同 value 时索引顺序不同），只要被选中的位置集合一致或差异位置的分数值接近即可。

> **v7.4 测试代码修正与边界 Case 测试计划（Phase 1+3 验证, 见 §12.12）**：
>
> **测试代码 torch.sort bug 修正（Phase 1）**：
> - **bug**: 测试代码在调用 `check_result` 前对 `cpu_result`/`npu_result` 执行 `torch.sort`, 破坏了 check_result 期望的 score 降序顺序, 导致 `value_bm = topk_value[b, n2, s1, cur_cpu[-1]]` 取错值（cur_cpu[-1] 不再是最小 topk value 的 index）.
> - **影响**: 1/8 PASS → 修正后 7/8 PASS（仅 li_default_a2 仍失败, 由 sort 前 -inf 填充缺失导致, 见 §8.8 步骤 2.5）.
> - **修复**: 去掉测试代码中的 `torch.sort(cpu_result)` / `torch.sort(npu_result)`. **注**: `check_result` 内部的 `np.sort(cpu_reshape, axis=1)` 是 index 集合对比所需（排序 index 后比集合）, **不是 bug**, 保留.
>
> **边界 Case 测试计划（Phase 3, Stage 2 在 L0 通过后扩展为 L1/Boundary 套件）**：
>
> | 边界 Case | 参数 | 验证点 | 预期 | 优先级 |
> |----------|------|--------|------|--------|
> | block_size=16 | B=20,S1=3,S2=512,bs=16,PA_BSND,BF16 | BLOCK_N 缩小后不超时 + 精度正确 | 不超时 + 7/8+ PASS | 🔴 P0 |
> | sparse_count=1 | TOP_K=1 | TOP_K_ALIGNED=64, UB=74.9KB | ✅ 无问题（已验证） | 🟢 — |
> | sparse_count=8192 (Over2K) | TOP_K=8192 | S1_BLOCK 缩小 + virTopK=sparse_count, UB 不超限 | UB<192KB + 精度正确 | 🔴 P0 |
> | S2=128 (< S2_VEC_BLOCK) | TOP_K=2048 > S2=128 | -inf 填充后输出 S2 个有效 + (TOP_K-S2) 个 -1 | 精度正确 + -1 填充 | 🟡 P1 |
> | S1=1 | 最小 S1 | 无特殊问题 | ✅ 无问题（已验证） | 🟢 — |
>
> **check_result 两步对比逻辑澄清**（`result_compare_method.py:174-221`, `compare_topk_valid`）：
> 1. 第一步: `np.sort(cpu_row) == np.sort(npu_row)` 排序 index 后比集合 → 集合相同直接通过
> 2. 第二步（仅集合不同时）: `value_bm = topk_value[b, n2, s1, cur_cpu[-1]]`（golden 最小 topk value）, 比较差异 index 的 value 相对误差（thres=0.0001）
> - **关键**: `cur_cpu[-1]` 依赖 cpu_result 的 score 降序排列, 故测试代码**不能**预先 torch.sort（会破坏降序）.

### 9.3 精度标准

> **采用用户指定的自定义精度标准**（来自 `result_compare_method.py` 和用户明确要求），非模板默认值。
> weights **绝对禁止 float32**（用户明确），query/key/weights dtype 必须一致。
> indices 输出为 int32, 采用**集合匹配 + value 相对误差**方法（非标准逐元素容差）。
> **v5 新增**（缺陷 #7）：sparse_values 的 FP16/BF16 精度标准。

| dtype | atol | rtol | max_abs_error_limit | required_matched_ratio | 对比方法 |
|-------|------|------|---------------------|------------------------|---------|
| float16 | 2.5e-5 (0.000025) | 5e-3 (0.005) | 0.1 | 0.99 | 标准混合容差 (value 对比时) |
| bfloat16 | 1e-4 (0.0001) | 5e-3 (0.005) | 1.0 | 0.99 | 标准混合容差 (value 对比时) |
| **float16 (sparse_values)** (v5 新增) | 2.5e-5 (0.000025) | 5e-3 (0.005) | 0.1 | 0.99 | 标准混合容差 (return_value=True 时) |
| **bfloat16 (sparse_values)** (v5 新增) | 1e-4 (0.0001) | 5e-3 (0.005) | 1.0 | 0.99 | 标准混合容差 (return_value=True 时) |
| int32 (indices) | 0 | 0 | 0 | 1.0 | **集合匹配 + value 相对误差 < 1e-4** (非逐元素) |

**int32 indices 特殊对比逻辑**（来自 `result_compare_method.py:174-221`，`compare_topk_valid` 函数）：

> **v7.2 缺口 3 修正说明**：
> - sort 稳定性差异（相同 value 时索引顺序不同）是**正常行为**, AscendC 也有相同行为
> - **不需要在 sort 后做稳定化处理**, check_result 已正确处理 index 顺序差异
> - 真正的精度问题在 **score 计算精度**（GEMM float32 累加、Weight mul cast 路径、ReLU 精度）, 不是 sort
> - 精度失败的根因是 score 值有微小差异, 导致 kernel 选了不同的 top-k index

- **第一步**: 排序 index 后比集合（`set(npu_row) == set(cpu_row)`）→ 集合相同直接通过
- **第二步**（仅集合不同时）: 比较差异 index 的 value 与 golden 最小 topk value 的相对误差
  - `value_bm = topk_value[b, n2, s1, cpu_topk[-1]]` — golden 最小 topk value
  - `npu_re = abs(npu_value - value_bm) / value_bm` — NPU 差异 index 的相对误差
  - `cpu_re = abs(cpu_value - value_bm) / value_bm` — CPU 差异 index 的相对误差
  - 阈值 `thres = 0.0001`, 误差 ≤ 阈值则通过, 否则失败
- 此方法不适用于标准 `check_precision` 函数, 需在 `test_lightning_indexer.py` 中实现专用 `check_result_li()` 函数

**sparse_values 精度对比逻辑**（v5 新增，缺陷 #7）：
- return_value=True 时，kernel 输出 sparse_values（FP32 注意力分数 Cast 到 FP16/BF16）
- 对比方法：标准混合容差（`torch.testing.assert_close` 或 `check_precision`）
- sparse_values 的 value 来自 topk 的分数输出，经 `T.tile.cast(dst, src, "CAST_ROUND", K)` 从 FP32 转为 FP16/BF16
- **无效位置**：FP16 填 -inf (0xFC00)，BF16 填 -inf (0xFF80)
- **L0 测试用例 #6/9/10/15**（return_value=True）需验证 sparse_values 精度

---

## 10. 风险点与注意事项

### 10.1 技术约束检测结论

| 约束项 | 检测结论 | 详情 |
|--------|---------|------|
| **三维 Kernel** | ✅ 已处理 | v6: Fixed Core 模式 `T.Kernel(min(total_tasks,动态核数), threads=2)`, 每核多任务循环展开 B×S1 维度 |
| **threads 参数** | ✅ 已处理 | v5: Kernel 1 `threads=2` (AIC:AIV=1:2), 配合 AUTO_CV_COMBINE + T.Pipelined |
| **动态边界** | ✅ 已处理 | act_q/act_k 为运行时 tensor; 使用 `if s1 < act_q` 条件判断 + `T.ceildiv(MAX_S2, BLOCK_N)` 编译期静态循环 + 运行时 if 条件; v5: TND 前缀和作为数据访问索引（**act_seq 标量读取需 Stage 2 验证, 缺陷 #6**）|
| **L0C 容量** (v5 修正) | ✅ 已验证 | v5: L0 inner split (M_L0=128, N_L0=128) 后 C_L0=64KB < 128KB ✓ **所有场景安全**（v4 的 [BLOCK_M,BLOCK_N] 全尺寸超限已修正） |
| **UB 容量 (Kernel 1)** (v6 修正) | ✅ 已验证 | v6: row_expand_mul + reduce_sum merge 后 UB=34KB < 192KB ✓ **所有 N1 场景安全**（v5: 68KB → v6: 34KB） |
| **UB 容量 (Kernel 2 大 S2)** | ✅ 已处理 | S2≤16384: 单次 topk 峰值 ≤160KB ✓; S2>16384: 分段 topk 峰值 ~96KB ✓ (§4.5) |
| **GEMM 非整除** | ✅ 已处理 | S2 不被 BLOCK_N 整除时用 T.ceildiv + valid_n 尾块处理; block_size=768 整除场景同此 |
| **TND layout** | ✅ 已处理 | v5: kernel 直接消费 TND strided DMA (§4.9), 消除 host wrapper 转置 (**act_seq 标量读取需 Stage 2 验证**) |
| **T.Pipelined UB ring-buffer** | ✅ 已处理 (v6 修正) | num_stages=2 (保守安全), UB buffer 在 pipeline body 外分配不被 ring-buffer; v6 UB=34KB (v5: 68KB), 可尝试 num_stages=3-4 |
| **Fixed Core 尾块** | ✅ 已处理 (v5 修正) | `if task_id < total_tasks:` 条件保护; v5: launch_core_num=min(total_tasks,NUM_CORES) 小任务不空转 (缺陷 #5); v6: NUM_CORES 动态获取 |
| **L0 inner split API** (v5 新增, v7.1 已验证) | 🔴 验证发现不可行 (v7.1 修正) | **`T.gemm_v0(transpose_B=True)` + L1 行切片 → 精度错误 (diff=66.5, 验证 3)**. v7.1 改用独立 L1 buffer 方案 (§5.2.1/§8.5), 无切片, 精度正确 (diff=0.0156) |
| **groupInner broadcast** (v5 新增, v6 降级) | ✅ API 已验证 (fallback) | T.tile.broadcast(dst=[16,512], src=[16], axis=1) 参数确认 (ascend_tile.py:2031). **v6: 降级为 fallback, 主路径用 row_expand_mul_experiment** |
| **T.tile.add/fill** (v5 新增, v6 add 降级) | ✅ API 已验证 | T.tile.add (ascend_tile.py:856) — **v6: 降级为 fallback, reduce_sum(clear=False) 替代 add**; T.tile.fill (ascend_tile.py:221) 仍用 |
| **row_expand_mul_experiment** (v6 新增) | ✅ API 已验证 | ascend_tile.py:2353-2372, 使用验证 xattention.py:901-913. ⚠️ Stage 2 验证 dst=src0 原地乘 |
| **reduce_sum(clear=False)** (v6 新增) | ✅ API 已验证 | api-compute.md:119-121 merge 语义. ⚠️ Stage 2 验证 scores_accum 初始化 |
| **pipe_barrier("v")** (v6 新增, v7.1 已验证) | ✅ v7.1 验证通过 (验证 5) | xattention.py:898, HISA:320. **v7.1 验证 (验证 5 PASS): 精度与 barrier_all 一致 (diff=0.0312)**. 跨管线同步仍需 barrier_all |
| **T.Pipelined num_stages=3-4** (v6 新增, v7.1 已验证) | ✅ v7.1 验证通过 (验证 4) | **v7.1 验证 (验证 4 PASS): num_stages=3/4 均编译通过且精度正确 (diff=0.0312)**. 关键: buffer 须在 body 外分配 |
| **threads=2 解包** (v6 新增, v7.1 已验证) | ✅ v7.1 验证通过 (验证 4) | **v7.1 验证: `as (cid)` 1 个值, 非 `as (cid, _)`** (验证 4 确认) |
| **动态核数 cube_core_num** (v6 新增) | ✅ API 已验证 | xattention.py:24. ⚠️ Stage 2 验证 A2/A3 跨芯片 |
| **v7 Cube 3 层 tiling** (v7 新增, v7.1 修正) | ✅ v7.1 修正完成 (验证 3) | M_L1=256, S2_L1=256 对齐 AscendC. **v7.1: 独立 L1 buffer 方案修正 transpose_B+切片精度问题 (验证 3), L1 占用 128KB < 512KB ✓** |
| **v7 sort 缓存优化** (v7 新增, v7.1 修正, v7.2 方案 C) | ✅ v7.2 方案 C 修正完成 (缺口 2) | SortedBasicBlock 4-way merge. v7.1: 1D 平铺+sort_tmp+T.copy+切片 src (验证 1+2 PASS). **v7.2: 方案 C (VID_S1=1 + buffer 复用) UB 从 216.4KB 降到 184.4KB ✓ (缺口 2 验证 `test_memory_budget.py`)** |
| **v7 TND_PA_BSND block_size=16** (v7 新增, v7.4 修复) | ✅ v7.4 已处理 (Phase 3) | msprof 实测超时. **根因**: PA + block_size=16 时 BLOCKS_PER_TILE=256/16=16, PA gather 过多. **v7.4 修复**: `is_pa and block_size<64 → BLOCK_N=min(BLOCK_N,128)`, BLOCKS_PER_TILE 降至 8 (§5.2 边界 Case 1). ⚠️ Stage 2 实测验证 |
| **v7.2 L0C→UB 跨级访问** (缺口 1 新增) | ✅ v7.2 已处理 | L0C 是 Cube 专用 accumulator, Vector 核不能直接访问. **正确路径: L0C → GM workspace → UB → Vector 处理 → GM output**. §8.5 伪代码已用 `T.copy(C_L0, qk_workspace, enable_relu=True)` 经 GM 中转. 参考 `sparse_flash_attn_pa_no_cv_pipeline.py` |
| **v7.2 Expert T.Scope("V") 解包** (缺口 1 新增) | ✅ v7.2 已标注 | T.Scope("V") 内 codegen 引用 `vid`, 必须用 `as (cid, vid)` 不能用 `as (cid, _)`. §8.5 已标注. 当前 Developer 模式 threads=2 用 `as (cid)`, Expert fallback 用 `as (cid, vid)` |
| **v7.2 SparseTopK 显式截断** (缺口 6 新增) | ✅ v7.2 已处理 | AscendC 用 `DataCopy(dst, tmp, topk*2)` 截断到 BASE_TOPK=2048. v7.2 在 merge_sort 后用 `T.copy(merge_output[0:BASE_TOPK*2], global_topk_ub)` 显式截断, 防止输出超出 global_topk_ub 分配 (§8.8) |
| **v7.2 sort 稳定性** (缺口 3 修正) | ✅ 不是问题 | sort 稳定性差异是正常行为, AscendC 也有. check_result 两步对比已处理 (先比集合, 再比 value 误差 thres=0.0001). **不需要在 sort 后做稳定化处理**. 真正精度问题在 score 计算精度 (§9.3) |
| **v7.2 score 计算精度** (缺口 3 新增) | 🟡 Stage 2 验证 | 精度失败的根因是 score 值微小差异, 导致 kernel 选了不同的 top-k index. 需要提高 score 计算精度: GEMM float32 累加 ✓ (已用 calc_dtype=float32), Weight mul cast 路径, ReLU 精度 |
| **v7.3 4×L0C 容量** (参考算子模式 1) | ✅ 已验证 | 4×L0C 总占用 4×64KB=256KB < 512KB ✓ (Ascend910B3 单 AIC L0C 上限). 每个子 GEMM 独占一个 L0C fragment, 消除 FIX↔M flag 争用 (§4.6, §8.5). ⚠️ Stage 2 验证: 编译后检查 4 fragment 是否独立 (get_kernel_source) |
| **v7.3 annotate_layout 兼容性** (参考算子模式 2) | 🟡 Stage 2 验证 | `T.annotate_layout({A_L1: make_zn_layout, B_L1: make_nz_layout})` 与 v7.1 独立 L1 buffer 方案的兼容性未验证. v7.1 仅验证独立 buffer 精度 (验证 3 PASS), layout 注解可能影响 codegen. 参考 `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py:105-112` |
| **v7.3 DMA 重排 flag 时序** (参考算子模式 3) | 🟡 Stage 2 验证 | Wave 0 (K[0]+Q 优先) + Wave 1+ (K[i] 与 staging overlap) 的 flag 时序需验证. 编排复杂度增加, 错误时序可能导致 DMA 数据竞争. 参考 `examples/HISA/paged_block_sparse_mqa_attn_expert.py` Wave pipeline |
| **v7.3 MTE2∥V overlap 同步复杂度** (参考算子模式 6) | 🟡 Stage 2 验证 | late DMA 入队 + set_flag("MTE2","V") + early output + wait_flag 的时序正确性. 错误时序会导致 sort 使用未加载完成的数据. 需 double buffer (score_buf ↔ score_buf_next). 参考 HISA 行 357-370 |
| **v7.3 brcb_experiment API 稳定性** (参考算子模式 7) | 🟡 Stage 2 验证 | `T.tile.brcb_experiment` + `T.tile.row_expand_mul_experiment` 三步模式的组合精度未验证. 当前 v6 row_expand_mul_experiment 已验证 (验证 5 PASS), brcb 前缀可能影响 flag 时序. 参考 `examples/xattention/xattention.py:891-916` |
| **v7.3 cross_interval=2 已采纳** (参考算子模式 5) | ✅ 已采纳 | v7.2 已用 `T.Pipelined(s2BaseNum, num_stages=2, cross_interval=2)`. v7.1 验证 cross_interval 参数可用 (验证 4 PASS). 无新增风险 |
| **v7.4 sort 前 -inf 填充** (Phase 1 精度根因) | ✅ v7.4 已处理 | sort 前未将无效 K 元素 score 设为 -inf → PA_BSND 输出 -1 无效 index (li_default_a2). **修复**: sort 前对 `score_buf[cu_s2_len:S2_VEC_BLOCK]` 填 -inf + `index_blk_ub[...]` 填 -1, 对齐 AscendC `service_vector.h:357-368` (§8.8 步骤 2.5). ⚠️ Stage 2 验证 `cu_s2_len` 动态计算正确性 (causal mask 分支) |
| **v7.4 Over2K UB 超限** (Phase 3 边界 Case 3) | 🟡 Stage 2 验证 | sparse_count>2048 (如 8192) 时 UB=586KB 严重超限. **v7.4 修复**: Over2K 时 S1_BLOCK=max(2, 8192//sparse_count*2) + virTopK=sparse_count + 走标准路径, 对齐 AscendC `kernel.h:183-185` (§5.2 边界 Case 3). ⚠️ Stage 2 实测实际 UB 占用 |
| **v7.4 TOP_K > S2** (Phase 3 边界 Case 4) | ✅ v7.4 已处理 | sparse_count=2048 > S2=128 时无法选满. **修复**: §8.8 步骤 2.5 的 -inf 填充使无效位置沉底, topk 后由 `T.tile.select(-1)` 填 -1. 需确认 `topk_a_ub` 初始化为 -inf (§10.2 Q27) |
| **v7.4 测试代码 torch.sort bug** (Phase 1) | ✅ 已修复 | 测试中 `torch.sort(cpu_result/npu_result)` 破坏 check_result 期望的 score 降序, 导致 value_bm 取错值 (1/8→7/8 PASS). **修复**: 去掉 torch.sort (§9). 注: check_result 内部的 `np.sort` 是 index 集合对比所需, 非 bug |

### 10.2 关键设计问题回答

**Q2: PA_BSND block_table 间接寻址**
- ✅ 已确认支持：`T.copy(Key[block_table[b, s2_block], :, 0, :], K_L1)` 使用运行时索引访问 GM tensor
- 参考：`examples/sparse_flash_attention/example_sparse_flash_attn_mask_pa.py` 中 `T.copy(KV[block_i, block_inter, 0, :D], kv_ub)` 确认此模式可用
- block_table[b, s2_block] 先读取为标量, 再用作 Key 的索引
- 设置 BLOCK_N = block_size, 实现 s2_block 到 KV block 的 1:1 映射

**Q3: rightDownCausal mask 生成与应用**
- **在线生成**（Kernel 2 内）, 非 host 预处理
- **应用时机**：group reduce 之后、topk 之前（与 golden 代码一致）
- **cutoff 计算**：`cutoff = act_k[b] - act_q[b] + s1 + 1`, key 位置 j >= cutoff 置 -inf
- **实现**：T.tile.createvecindex + T.tile.compare("GE") + T.tile.select(-inf)
- 参考：`examples/seer_attention/block_sparse_attn.py` 的 causal mask 实现模式

**Q4: TopK 稳定性** (v7.2 缺口 3 修正)

> **v7.2 关键修正（缺口 3 验证）**：
> - ~~sort 稳定性是问题~~ → **sort 稳定性差异是正常行为, AscendC 也有相同行为**
> - **不需要在 sort 后做稳定化处理**, check_result 已正确处理 index 顺序差异
> - 真正的精度问题在 **score 计算精度**（value 误差）, 不是 sort

- T.tile.topk / T.tile.sort / T.tile.merge_sort 硬件排序**不保证**与 PyTorch stable=True 完全一致（相同 value 时索引顺序可能不同）
- **v7.2 验证结论（缺口 3, `test_sort_stability.py`）**: 两条路径（标准路径 + sort 缓存路径）的 sort 稳定性差异**完全相同**, 说明不是 sort 缓存优化引入的问题, 而是 Sort32 硬件指令的固有行为
- **精度对比方法兼容**：`result_compare_method.py:174-221` 的 `compare_topk_valid` 函数使用两步对比:
  1. 第一步: 排序 index 后比集合 (`set(npu_row) == set(cpu_row)`) → 集合相同直接通过
  2. 第二步（仅集合不同时）: 比较差异 index 的 value 与 golden 最小 topk value 的相对误差 (thres=0.0001)
- **容忍稳定性差异**, 无需额外处理
- **精度问题正确方向**: 真正的精度失败根因是 **score 值有微小差异**, 导致 kernel 选了不同的 top-k index。差异 index 的 score 不完全相同（否则 value 误差 = 0 会通过）。需要提高 score 计算精度:
  - GEMM float32 累加 ✓ (已用 calc_dtype=float32, C_L0 是 float32)
  - Weight mul cast 路径: 确保 Weights 从 fp16/bf16 cast 到 float32 后再乘
  - ReLU 精度: L0C→GM 时 enable_relu 融合, 精度由硬件保证

**Q6: 大 S2 UB 容量规划（修正版，含 topk sort_tmp）**

> **前版错误**：前版 DESIGN.md 遗漏了 T.tile.topk 内部注入的排序临时 buffer（`allocate_tmp_buffer.cc` pass），UB 预算被低估为 67KB。本版已修正。

- **topk sort_tmp 大小**（源码验证 `src/transform/allocate_tmp_buffer.cc:597-619`）：
  - float32 (calc_dtype): `aligned_count × 6` 字节，`aligned_count = ((MAX_S2+31)//32)×32`
  - half (fp16/bf16): `aligned_count × 16` 字节（含 cast-to-float pool）
  - 本算子 calc_dtype=float32，故 sort_tmp = MAX_S2 × 6 字节
- **单次 topk 路径 UB 峰值**（含 MEMORY_PLANNING 地址复用）：
  - MAX_S2=8192: score_accum(32KB) + sort_tmp(48KB) = **80KB** < 192KB ✓
  - MAX_S2=16384: score_accum(64KB) + sort_tmp(96KB) = **160KB** < 192KB ✓
  - MAX_S2=20480: score_accum(80KB) + sort_tmp(120KB) = **200KB** > 192KB ✗
- **分段 topk 路径**（S2 > 16384）：
  - score_accum 放 GM workspace，UB 仅需段 topk + merge_sort 峰值 ~96KB ✓
  - SEGMENT_SIZE=8192，sort_tmp=48KB，与 S2 无关
- **结论**：MAX_S2_UB = 16384 为单次/分段路径分界线（§4.5 详述）
- T.tile.topk 要求 src 静态 shape: 使用 MAX_S2 (单次) 或 SEGMENT_SIZE (分段) 编译期常量, 运行时传 actual_num
- T.tile.topk 内部自动对齐到 32 元素, 无需手动 padding

**Q7: TND layout 如何处理？（v4 更新）**

- **v4 方案**：kernel 直接消费 TND（strided DMA），消除 host wrapper 转置，详见 §4.9
- **v3 方案（已废弃）**：host wrapper 转换（TND→BSND 输入，BSND→TND 输出）
- **v4 改进原因**：msprof 实测 TND 场景（24% 用例）的转置 kernel 开销 15-30%
- **kernel 内 TND 访问**：`T.copy(Query[t_start + offset : t_start + offset + s1BaseSize, :, :], Q_L1)`
  - 切片起始动态（t_start 从 act_seq 前缀和读取），切片长度固定（s1BaseSize × N1 × D）
  - D 是最内层连续维，T.copy 原生支持 strided DMA（参考 `deepseek_v4/sparse_attention.py`）
- **"循环边界不能依赖 tensor 值"约束**：此约束针对循环次数，不针对数据访问索引。`t_start` 作为切片起始是数据访问，非循环控制
- **TND_PA_BSND 混合**：query 用 TND strided DMA，key 用 PA_BSND block_table 间接寻址
- **TND_TND**：query 和 key 均用 TND strided DMA
- **act_seq 传递**：前缀和 GM tensor 直接传 kernel，kernel 内 `actual_seq_q[b]` 读取为标量（Q18 详述）

**Q8: S2=131072 时 UB 规划方案？**

- **选定方案**：Strategy B+D 混合（GM workspace + 分段 topk + merge_sort）
- **四种方案分析**：

| 方案 | UB 占用 | 性能 | 实现复杂度 | 采用 |
|------|---------|------|-----------|------|
| A: 分段 topk + 合并 | 固定（段大小决定） | 中（多次 topk + 合并） | 高（合并逻辑复杂） | ✗ |
| B: score_accum 放 GM | UB 固定 | 中低（增加 GM 访问） | 中 | 部分 |
| C: 按 MAX_S2 分档编译 | 每档最优 | 高（每档最优 UB） | 低（但编译次数多） | 部分 |
| D: GM workspace + 流式 topk | UB 仅与 k 相关 | 中（流式扫描） | 高（流式 topk 复杂） | 部分 |

- **最终方案 B+D**：score_accum 放 GM（方案 B）+ 分段 topk + merge_sort 合并（方案 D 的简化版）
  - score_accum_gm: [MAX_S2] float32 在 GM（workspace_idx）
  - 分段：SEGMENT_SIZE=8192，每段 topk → k 个 (value, local_index) 对
  - 合并：T.tile.merge_sort（2-way，参考 `examples/sort/example_merge_sort.py`）
  - local_index → global_index: 段 topk 后对 index 位加 seg_start
  - UB 峰值 ~96KB < 192KB ✓（与 S2 无关）
- **方案 C 作为补充**：编译两个 Kernel 2 变体（单次/分段），host dispatch 根据 MAX_S2 选择
- **不选纯方案 A 的原因**：合并逻辑需自定义，merge_sort 已提供硬件加速的 2/3/4-way merge
- **不选纯方案 D 的原因**：流式 topk（维护大小 k 的最小堆）在 TileLang 中实现复杂

**Q9: block_size=768 是否被 T.gemm_v0 支持？**

- **源码验证**（`tilelang/language/ascend.py:343-448`）：
  - T.gemm_v0 仅要求 `kL0Size % 16 == 0` 且 `kL0Size <= 4095`
  - **无 block_size (N 维) 必须为 2 的幂次约束**
  - 768 = 48 × 16，满足 16 对齐 ✓
- **L0C 容量验证**（§4.6）：
  - G=8, block_size=768: C_L0 = 8×768×4 = 24KB < 128KB ✓
  - G=16, block_size=768: C_L0 = 16×768×4 = 48KB < 128KB ✓
  - G=24, block_size=768: C_L0 = 24×768×4 = 72KB < 128KB ✓
  - G=32, block_size=768: C_L0 = 32×768×4 = 96KB < 128KB ✓
  - G=64, block_size=768: C_L0 = 64×768×4 = 192KB > 128KB ✗ → 需 inner split (BLOCK_N_inner=512)
- **测试用例验证**：block_size=768 出现在 N1=64 (G=64) 场景 → 需 inner split N
- **结论**：block_size=768 可用，G=64 时需 inner split

**Q10: T.tile.topk 是否支持任意 sparse_count（非 2 的幂次）？**

- **源码验证**（`tilelang/language/ascend_tile.py:419-462`）：
  - K 参数为 PrimExpr（可为运行时值或编译期常量），**无 2 的幂次约束**
  - dst 需 ≥ 2×K 元素，K 为任意正整数
  - actual_num 也为 PrimExpr，可运行时
- **测试用例验证**：sparse_count 分布含 153/308/315/393/596/822/1000/1774 等非幂次值
- **结论**：sparse_count ∈ [1, 2048] 任意整数均支持 ✓

**Q16: T.Pipelined 与 threads=2 + AUTO_CV_COMBINE 如何组合？（v4 新增）**

- **源码验证结论**：T.Pipelined **可以**与 threads=2 + AUTO_CV_COMBINE 组合使用
- **API 签名**（`pipeline.py:11`）：`Pipelined(start, stop, num_stages, order, stage, sync, group, cross_interval=1)`
- **已验证可运行的参考实现**：
  - `examples/pipeline/matmul_add_pipeline.py:46`：`for k in T.Pipelined(loop_k, num_stages=3)` + AUTO_CV_COMBINE + (cid, vid) 模式
  - `examples/pipeline/sparse_flash_attn_gqa_pipeline.py:128`：`for i_i in T.Pipelined(NI, num_stages=2)` + AUTO_CV_COMBINE + Expert 风格 alloc_L1/ub
- **UB ring-buffer 风险**（`flash_attn_bhsd_auto_pipeline_h16_d128.py` 文档记录）：
  - num_stages=8 时编译器对 pipeline body 内 UB buffer 做 ring-buffer，导致 UB 超限
  - **本设计对策**：num_stages=2（保守安全），UB buffer 在 pipeline body 外分配
  - 参考 `matmul_add_pipeline.py`：buffer 在 pipeline body 外分配，num_stages=3 可运行
- **cross_interval=2 的作用**：每 2 次迭代同步一次，减少 50% 跨核同步开销
- **组合用法**：
  ```python
  pass_configs = {
      TL_ASCEND_AUTO_CV_COMBINE: True,
      TL_ASCEND_AUTO_CV_SYNC: True,
      TL_ASCEND_AUTO_SYNC: True,
  }
  with T.Kernel(NUM_CORES, threads=2, is_npu=True) as (cid, vid):
      for s2_block in T.Pipelined(s2BaseNum, num_stages=2, cross_interval=2):
          # Cube: GEMM → workspace (编译器自动插 set_cross_flag)
          # Vector: workspace → mul+reduce (编译器自动插 wait_cross_flag)
  ```

**Q17: Fixed Core 模式下 workspace 如何分配？（v4 新增）**

- **workspace 按 NUM_CORES × num_stages 分配**（非按 total_tasks）：
  ```python
  qk_workspace: T.Tensor([NUM_CORES, num_stages, BLOCK_M, BLOCK_N], calc_dtype)
  # 访问: qk_workspace[cid, s2_block % num_stages, :, :]
  ```
- **L2 cache 友好性**：
  - v4 workspace 大小：24 × 2 × 128 × 128 × 4B = 3MB << 192MB L2 cache → 高命中率
  - v3 workspace 大小：108 × 128 × 128 × 4B = 6MB > 单行 L2 cache → 低命中率
- **内存节省**：v4 固定 3MB，v3 随 B×S1×S2_blocks 线性增长（最坏 6MB+）
- **参考**：`flash_attn_optimize.md` §7 "workspace 按 NUM_CORES 和 num_stages 分配"

**Q18: kernel 直接消费 TND 时，act_seq 前缀和如何传递？（v4 新增）**

- **act_seq 是 GM tensor**，kernel 内通过 `actual_seq_q[b]` 读取为 int32 标量
- **前缀和读取方式**：
  ```python
  # b 为当前 batch 索引 (compile-time loop variable)
  t_start = T.if_then_else(b == 0, 0, actual_seq_q[b - 1])  # 前缀和: 前 b 个 batch 的 token 总数
  act_q_batch = actual_seq_q[b] - t_start                    # per-batch token count
  ```
- **性能影响**：每个 task 读取 1-2 个 int32 标量（8B GM read），相比 GEMM 计算（MB 级数据）可忽略
- **"循环边界不能依赖 tensor 值"约束不冲突**：
  - 约束针对**循环次数**（`for i in T.serial(act_q)` 禁止）
  - `t_start` 作为**数据访问索引**（`T.copy(Query[t_start, :, :], Q_L1)` 允许）
  - v3 已有此模式：`act_q = actual_seq_q[b]` 读取为标量用于 if 条件判断
- **TND strided DMA 正确性**：
  - 切片长度固定（s1BaseSize × N1 × D），dst buffer 静态 shape ✓
  - 切片起始动态（t_start + offset），T.copy 原生支持（参考 PA_BSND `block_table[b, s2_block]` 运行时索引）
  - D 是最内层连续维，N·D 跨步的 strided DMA 硬件原生支持
- **尾块越界保护**：最后一个 batch 的最后一个 s1_tile 可能越过 T1 → `if s1_tile * s1BaseSize < act_q_batch:` 条件跳过

**Q19: broadcast + 整 tile mul 的 UB 占用如何变化？（v5 修正）**

> **v5 关键修正**：v4 用 [BLOCK_M, BLOCK_N] 全尺寸 broadcast，UB 超限。v5 缩小 broadcast 范围为 [groupInner, BLOCK_N]。

- **v5 UB 占用（groupInner=16 分块后）**：

| Buffer | v4 大小 (超限) | v5 大小 (修正) | 修正原因 |
|--------|---------------|---------------|---------|
| qk_ub | [BLOCK_M=512, BLOCK_N=512]=1MB | **[groupInner=16, BLOCK_N=512]=32KB** | groupInner 分块 |
| weight_ub | [BLOCK_M=512]=2KB | **[groupInner=16]=64B** | groupInner 分块 |
| **weight_2d** (broadcast) | [BLOCK_M=512, BLOCK_N=512]=1MB | **[groupInner=16, BLOCK_N=512]=32KB** | broadcast 范围缩小 |
| scores_partial (v5 新增) | — | **[BLOCK_N=512]=2KB** | 单次 groupInner reduce |
| scores_accum | [BLOCK_N=512]=2KB | **[BLOCK_N=512]=2KB** | 不变 |
| **总计** | ~2MB (超限 ❌) | **~68KB** ✓ | v5 groupInner=16 分块 |

- **v5 UB 容量验证**：68KB < 192KB ✓。即使 T.Pipelined ring-buffer（68KB×2=136KB < 192KB ✓）也安全。
- **v5 broadcast 范围**：`T.tile.broadcast(weight_2d=[16,512], weight_ub=[16], axis=1)` — 1D [16] → 2D [16, 512]
- **v5 循环开销**：s1BaseSize × outerG 次循环（如 8×4=32 次），每次 broadcast + mul + reduce + add = 4 条指令 → 128 条指令
  - v3 逐行 mul：BLOCK_M=512 次 mul + 1 次 reduce = 513 条指令
  - v5 比 v3 少 4 倍指令，比 v4（3 条指令但 UB 超限）安全
- **AscendC 参考**：`lightning_indexer_service_vector.h:198` `groupInner_ = 16`，`tmpBuf_ = (16*512 + 512)*2*4 = 68KB`

> **v6 更新**（优化 #1+#2）：v5 的 broadcast+mul 两步 + reduce+add 两步，被 `row_expand_mul_experiment`（融合 Brcb+Mul）+ `reduce_sum(clear=False)`（融合 reduce+累加）替代：
> - 移除 weight_2d (32KB) → UB 从 68KB 降到 34KB
> - 移除 scores_partial (2KB) → UB 进一步降到 34KB
> - 每次 iteration 指令数从 6 条 (copy qk + copy weight + broadcast + mul + reduce + add) 降为 4 条 (copy qk + copy weight + row_expand_mul + reduce_sum merge)
> - v5 的 broadcast 方案降级为 fallback（若 row_expand_mul 不可用时回退）
> - 详见 Q23 (row_expand_mul) 和 Q24 (reduce_sum clear=False)

**Q20: Vector 侧 groupInner=16 分块如何工作？（v5 新增）**

- **问题背景**：v4 的 qk_ub=[BLOCK_M, BLOCK_N] 全尺寸，N1≥16 时 UB 超限。v5 引入 groupInner=16 分块。

- **groupInner=16 的含义**：
  - groupInner 是 Vector 侧 G 维的内层分块大小（固定 16）
  - BLOCK_M = s1BaseSize × G（如 8×64=512），qk_workspace 仍是 [BLOCK_M, BLOCK_N]（GM 无容量限制）
  - Vector 侧每次从 qk_workspace 读取 [groupInner, BLOCK_N] 切片，循环处理

- **Vector 侧循环结构**：
  ```python
  groupInner = 16
  outerG = T.ceildiv(G, groupInner)  # G=64→4, G=16→1

  T.tile.fill(scores_accum, 0)  # 初始化累加器

  for s1_inner in T.serial(s1BaseSize):        # 8 次
      for g_idx in T.serial(outerG):            # 4 次 (G=64)
          m_start = s1_inner * G + g_idx * groupInner  # qk_workspace 行偏移
          # 读取 [groupInner, BLOCK_N] 切片
          T.copy(qk_workspace[cid, slot, m_start:m_start+groupInner, :], qk_ub)
          # 读取权重 [groupInner]
          T.copy(Weights[b, s1_tile*s1BaseSize+s1_inner, g_idx*groupInner:(g_idx+1)*groupInner], weight_ub)
          # broadcast + mul + reduce + accumulate
          T.tile.broadcast(weight_2d, weight_ub, axis=1)  # [16] → [16, 512]
          T.tile.mul(qk_ub, qk_ub, weight_2d)              # [16, 512]
          T.reduce_sum(qk_ub, scores_partial, dim=0)        # [16, 512] → [512]
          T.tile.add(scores_accum, scores_accum, scores_partial)
  ```

- **BLOCK_M 排列方式**：BLOCK_M = s1BaseSize × G，排列为 s1 在外、G 在内（reshape(s1BaseSize * G, D)）
  - row m 对应 s1 = m // G, g = m % G
  - m_start = s1_inner * G + g_idx * groupInner（连续 16 行 = 同一 s1 的 16 个 group）

- **scores_accum 累加逻辑**：
  - scores_accum 跨所有 s1_inner × outerG 次迭代累加
  - 最终 scores_accum = sum over (s1_inner, g) of (qk_relu[s1,g,:] × weight[s1,g]) = 完整分数 [BLOCK_N]
  - scores_accum shape = [BLOCK_N] = 2KB，固定大小（与 G 无关）

- **API 验证**：
  - T.tile.broadcast(dst=[16,512], src=[16], axis=1) — ascend_tile.py:2031, 1D→2D axis=1 ✓
  - T.tile.mul(dst, src0, src1) — ascend_tile.py:878 ✓
  - T.reduce_sum(buffer=[16,512], out=[512], dim=0) — reduce_ascend.py:391 ✓
  - T.tile.add(dst, src0, src1) — ascend_tile.py:856 ✓
  - T.tile.fill(buffer, value) — ascend_tile.py:221 ✓

**Q21: Cube 侧 L0 inner split 如何工作？（v5 新增）**

- **问题背景**：v4 的 C_L0=[BLOCK_M, BLOCK_N] 全尺寸，N1≥16 时 L0C 超限（如 [512,512]×4=1MB>>128KB）。v5 引入 L0 inner split。

- **L0 inner split 的含义**：
  - M_L0=128, N_L0=128（L0 级 GEMM 的块大小，固定值）
  - C_L0 = [M_L0, N_L0] = [128, 128] × 4B = 64KB < 128KB ✓
  - Q_L1/K_L1 仍是 [BLOCK_M, D]/[BLOCK_N, D]（L1 级别，512KB 容量足够）
  - GEMM 内部分 M 维和 N 维两重循环

- **Cube 侧循环结构**：
  ```python
  M_L0 = 128
  N_L0 = 128

  for m_l0 in T.serial(T.ceildiv(BLOCK_M, M_L0)):    # 512/128=4 次 (G=64)
      for n_l0 in T.serial(T.ceildiv(BLOCK_N, N_L0)):  # 512/128=4 次
          # T.gemm_v0 + BufferRegion 切片
          # M, N 从 C_L0 shape 推导 = [M_L0, N_L0] = [128, 128]
          T.gemm_v0(
              Q_L1[m_l0*M_L0:(m_l0+1)*M_L0, :],  # L1 切片 [M_L0, D]=[128, 128]
              K_L1[n_l0*N_L0:(n_l0+1)*N_L0, :],  # L1 切片 [N_L0, D]=[128, 128]
              C_L0,                                # [M_L0, N_L0]=[128, 128]
              transpose_B=True,
              init=(n_l0 == 0)                     # N 维首次 init, 后续累加
          )
      # C_L0 完成一整行 [M_L0, BLOCK_N], copy 到 workspace (enable_relu)
      T.copy(C_L0, qk_workspace[cid, slot, m_l0*M_L0:(m_l0+1)*M_L0, :], enable_relu=True)
  ```

- **T.gemm_v0 的 M, N 推导**（ascend.py:413）：
  - `M, N = C_shape[-2], C_shape[-1]` — 从 C_L0 shape 推导
  - A/B 的 shape 仅用于推导 K（ascend.py:414）
  - 因此 C_L0=[128,128] 决定 M=128, N=128，即使 Q_L1=[512,128] 也可用

- **init 参数的累加逻辑**：
  - n_l0==0 时 init=True：C_L0 清零后计算 Q_L1[m_slice] × K_L1[n_slice]
  - n_l0>0 时 init=False：C_L0 += Q_L1[m_slice] × K_L1[n_slice]（累加）
  - 一个 m_l0 迭代内，N 维全部累加完后 C_L0 = [M_L0, BLOCK_N] 的完整结果
  - m_l0 切换时无需特殊处理（新切片重新 init）

- **⚠️ Stage 2 验证项**（v7.1 已验证, 见 §12.8 API 验证结果）：
  1. ~~`T.gemm_v0(Q_L1[m_l0*M_L0:(m_l0+1)*M_L0, :], ...)` 的 2D L1 行切片是否被支持~~ → **v7.1 验证结论 (验证 3): `transpose_B=True` + L1 行切片 → 精度错误 (diff=66.5), 不可行!** 改用独立 L1 buffer (§5.2.1 / §8.5 v7.1 伪代码)
     - examples 中仅有 3D 第一维切片先例（`chunk_gated_delta_rule.py:132` `w_chunk_l1[pid, :, :]`）
     - 2D 行切片 + transpose_B 组合经验证不可行
  2. ~~**fallback 方案**：若 2D 行切片不可用，改为 `Q_L1 = T.alloc_shared((M_L0, D), input_dtype)`~~ → **v7.1 已采纳此方案**: 独立 buffer A_L1_0/A_L1_1 各 [M_L0, D], L1 占用 128KB < 512KB ✓
  3. `T.copy(C_L0, qk_workspace[..., m_slice, :], enable_relu=True)` — L0C→GM 部分写入 + enable_relu — **待 Stage 2 验证**

> **⚠️ v7.1 重要说明**：以上 Q21 的 L0 inner split + L1 切片方案（v5/v6）已被 v7.1 的独立 L1 buffer 方案替代（§5.2.1 / §8.5）。Q21 内容保留作为历史参考和 fallback 分析, **实际实现以 §8.5 v7.1 伪代码为准**。

- **AscendC 参考**：`lightning_indexer_service_cube.h:54-65`
  ```cpp
  M_BASIC_BLOCK = 256;      // L1 级别
  S2_BASIC_BLOCK = 256;
  M_BASIC_BLOCK_L0 = 128;   // L0 级别 GEMM
  S2_BASIC_BLOCK_L0 = 128;
  // C_L0 = 128×128×4B = 64KB ✓
  ```

**Q22: T.Pipelined + groupInner + L0 inner split 三层嵌套如何协调？（v5 新增）**

- **三层嵌套结构**：
  ```
  T.Pipelined(s2BaseNum, num_stages=2)  ← 最外层：S2 核间流水
    └─ for m_l0 in T.serial(...)         ← Cube 侧：L0 M 维 inner split
    │    └─ for n_l0 in T.serial(...)    ← Cube 侧：L0 N 维 inner split
    │         └─ T.gemm_v0(Q_L1 切片, K_L1 切片, C_L0)
    │    └─ T.copy(C_L0 → qk_workspace[m_slice])
    └─ for s1_inner in T.serial(...)     ← Vector 侧：s1 维循环
         └─ for g_idx in T.serial(...)   ← Vector 侧：groupInner 分块
              └─ T.copy(qk_workspace[m_slice] → qk_ub)
              └─ T.copy(Weights → weight_ub)
              └─ T.tile.row_expand_mul_experiment(qk_ub, qk_ub, weight_ub)  ← v6: 融合 Brcb+Mul
              └─ T.reduce_sum(qk_ub, scores_accum, dim=0, clear=False)     ← v6: 融合 reduce+累加
              └─ T.pipe_barrier("v")                                        ← v6: Vector 管线轻量同步
    └─ T.copy(scores_accum → Scores)
  ```

- **T.Pipelined 的作用范围**：
  - T.Pipelined 包裹整个 S2 循环体（Cube + Vector 部分）
  - 编译器自动识别 GEMM（T.gemm_v0）→ AIC，其余 → AIV
  - 迭代 k 的 Cube（L0 inner split GEMM）与迭代 k-1 的 Vector（groupInner mul+reduce）重叠

- **Cube/Vector 在 T.Pipelined 内的分工**：
  | 阶段 | 执行核 | 操作 | 数据依赖 |
  |------|--------|------|---------|
  | Cube | AIC | GM→L1(Q,K) + L0 inner split GEMM + L0C→GM(workspace) | 读 GM Q/K, 写 GM qk_workspace |
  | Vector | AIV | GM→UB(qk_workspace 切片) + row_expand_mul + reduce_sum(clear=False) + pipe_barrier("v") + UB→GM(Scores) | 读 GM qk_workspace + Weights, 写 GM Scores |

- **同步点**：
  - Cube 写 qk_workspace → 编译器自动插 set_cross_flag
  - Vector 读 qk_workspace → 编译器自动插 wait_cross_flag
  - cross_interval=2：每 2 次 S2 迭代同步一次（减少同步开销）
  - workspace ring-buffer：`qk_workspace[cid, s2_block % num_stages, :, :]`，2 slot 交替

- **UB/L1 buffer 在 pipeline body 外**：
  - Q_L1, K_L1, C_L0, qk_ub, weight_ub, scores_accum 均在 `with T.Kernel` 内、`for s2_block in T.Pipelined` 外分配
  - **v6 移除**: weight_2d, scores_partial（由 row_expand_mul + reduce_sum(clear=False) 融合）
  - 编译器对 body 外 buffer 不做 ring-buffer（参考 `matmul_add_pipeline.py`）
  - 仅 GM workspace 做 ring-buffer（GM 无容量限制）

- **性能考量**：
  - L0 inner split 增加 Cube 侧循环次数（M×N = 4×4=16 次 GEMM），但每次 GEMM 更小（128×128），L0A/L0B 利用率更高
  - groupInner 增加 Vector 侧循环次数（s1×outerG = 8×4=32 次），但每次数据更小（[16,512]），UB 利用率更高
  - T.Pipelined 的 CV overlap 隐藏了这些内层循环的延迟
  - **v6**: row_expand_mul + reduce_sum merge 后, 每次 iteration 从 6 条指令降为 4 条, Vector 侧延迟进一步降低

**Q23: `T.tile.row_expand_mul_experiment` 如何替代 broadcast + mul？（v6 新增）**

- **API 签名**（源码验证: `ascend_tile.py:2353-2372`）：
  ```python
  def row_expand_mul_experiment(dst, src0, src1, tmp=None):
      """dst[i,j] = src0[i,j] * src1[i]
      AscendC: brcb(src1→tmp) + mul_mask(dst, src0, tmp)
      PTO: TROWEXPANDMUL_row_vec(dst, src0, src1)"""
  ```

- **参数对齐**（lightning_indexer 场景）：
  | 参数 | v5 broadcast+mul | v6 row_expand_mul | 说明 |
  |------|-------------------|-------------------|------|
  | dst | qk_ub (被 mul 覆盖) | qk_ub (原地乘) | 2D [groupInner=16, BLOCK_N=512] |
  | src0 | qk_ub (mul 输入) | qk_ub (与 dst 同 buffer) | 2D [16, 512] |
  | src1 | weight_2d (broadcast 后 2D) | weight_ub (1D 行向量) | v5: [16,512] broadcast → v6: [16] 原始 |
  | tmp | 无 | None (可选, 内部管理) | AscendC 后端自动分配 |

- **等价性证明**：
  - v5: `broadcast(weight_2d, weight_ub, axis=1)` → weight_2d[i,j] = weight_ub[i]; 然后 `mul(qk_ub, qk_ub, weight_2d)` → qk_ub[i,j] = qk_ub[i,j] × weight_2d[i,j] = qk_ub[i,j] × weight_ub[i]
  - v6: `row_expand_mul_experiment(qk_ub, qk_ub, weight_ub)` → dst[i,j] = src0[i,j] × src1[i] = qk_ub[i,j] × weight_ub[i]
  - **数学等价** ✓

- **收益**：
  - 移除 weight_2d buffer（-32KB UB）
  - 移除 broadcast 指令（-1 条）
  - 移除 mul 指令（-1 条，row_expand_mul 内部融合）
  - PTO 后端单条 TROWEXPANDMUL 指令
  - AscendC 后端自动生成 brcb + mul_mask（与 AscendC lightning_indexer 完全对齐）

- **使用验证**：`examples/xattention/xattention.py:901-913` 已使用同款 API（dst=src0 原地乘模式）

**Q24: `T.reduce_sum(clear=False)` 的 merge 语义如何工作？（v6 新增）**

- **API 语义**（源码验证: `api-compute.md:119-121`）：
  ```
  clear=True (默认): out = reduced_result              (覆盖)
  clear=False:        new_out = old_out + reduced_result  (merge/累加)
  ```

- **参数对齐**（lightning_indexer 场景）：
  | 参数 | v5 reduce+add 两步 | v6 reduce_sum(clear=False) | 说明 |
  |------|---------------------|----------------------------|------|
  | src | qk_ub [16,512] | qk_ub [16,512] | 不变 |
  | dst | scores_partial [512] | scores_accum [512] | v6: 直接累加到最终目标 |
  | dim | 0 | 0 | 不变 |
  | clear | True (默认) | **False** | v6: merge 语义 |

- **等价性证明**：
  - v5: `reduce_sum(qk_ub, scores_partial, dim=0)` → scores_partial = sum(qk_ub, dim=0); 然后 `add(scores_accum, scores_accum, scores_partial)` → scores_accum = scores_accum + scores_partial
  - v6: `reduce_sum(qk_ub, scores_accum, dim=0, clear=False)` → new_scores_accum = old_scores_accum + sum(qk_ub, dim=0)
  - **数学等价** ✓（前提: scores_accum 在循环前已 fill(0) 初始化）

- **收益**：
  - 移除 scores_partial buffer（-2KB UB）
  - 移除 add 指令（-1 条）
  - reduce + 累加融合为单条指令

- **使用验证**：`examples/HISA/paged_block_sparse_mqa_attn_expert.py:343` 使用 `reduce_sum(..., clear=True)`（对照），clear=False merge 语义在 api-compute.md:119-121 确认

- **注意**：
  - scores_accum 必须在循环前用 `T.tile.fill(scores_accum, 0)` 初始化（已有，v5 保留）
  - clear=False 的 merge 语义依赖 dst 的旧值，确保 dst 不被其他操作并发写入

**Q25: v6 UB 34KB 能支持多大 num_stages？（v6 新增）**

- **v6 UB buffer 清单**（per core, pipeline body 外）：
  | Buffer | 大小 | 说明 |
  |--------|------|------|
  | qk_ub [16,512] | 32KB | row_expand_mul dst=src0 |
  | weight_ub [16] | 64B | row_expand_mul src1 |
  | scores_accum [512] | 2KB | reduce_sum(clear=False) dst |
  | **总计** | **~34KB** | / 192KB UB |

- **num_stages 容量分析**：
  | num_stages | UB ring-buffer? | GM workspace slots | 可行? | 说明 |
  |-----------|-----------------|-------------------|-------|------|
  | 2 | 否 (body 外) | 2 | ✅ | v6 基线, 保守 |
  | 3 | 否 (body 外) | 3 | ✅ | pipeline 重叠更好, 推荐尝试 |
  | 4 | 否 (body 外) | 4 | ✅ | 仍有收益, diminishing returns |
  | 5 | 否 (body 外) | 5 | ⚠️ | 理论可行, 收益递减, 需实测 |

- **关键前提**：UB buffer (qk_ub, weight_ub, scores_accum) 在 `with T.Kernel` 内、`for s2_block in T.Pipelined` 外分配（Q22 详述），编译器对 body 外 buffer 不做 ring-buffer。仅 GM workspace 做 ring-buffer（GM 无容量限制）。

- **v5 的错误推理（v6 修正）**：
  - v5 说 "num_stages=3 时即使 ring-buffer：68KB×3=204KB > 192KB（有风险）"
  - **v6 修正**：UB buffer 在 body 外不被 ring-buffer，上述推理不适用。num_stages 不受 UB×num_stages 限制，仅受 GM workspace slots 和 pipeline 有效性限制
  - 参考 `flash_attn_optimize.md` 的 "KNOWN BROKEN" 教训：UB buffer 在 pipeline body 内分配时才会 ring-buffer 超限

- **v6 建议**：num_stages=2（保守基线）→ Stage 2 实测后尝试 3-4。参考 `matmul_add_pipeline.py:46`（num_stages=3 已验证可运行）

**Q26: v7 Cube 3 层 tiling 对齐 AscendC 如何工作？（v7 新增）**

- **问题背景**：msprof 实测 BSND_BSND Cube 时间 98.83us vs AscendC 60.14us（慢 64%）。分析发现 v6 的 M_L1=S1_BLOCK×G=512 导致 L1 超限（Q_L1+K_L1=640KB > 512KB）。

- **v7 3 层 tiling 结构**（对齐 AscendC `service_cube.h:148-204`, v7.1 独立 buffer 修正）：

  ```
  L1 层 S2 循环 (S2_L1=256)        ← AscendC: s2GmOffset
    └─ B_L1_0, B_L1_1: GM → 独立 L1 buffer (各 N_L0=128 行, 无切片)  ← v7.1 修正
    └─ L1 层 M 循环 (M_L1=256)     ← AscendC: s1gGmOffset
         └─ A_L1_0, A_L1_1: GM → 独立 L1 buffer (各 M_L0=128 行, 无切片)  ← v7.1 修正
         └─ 4 个子 GEMM (2×2 展开, 独立 buffer, 无 L1 切片)
              └─ T.gemm_v0(A_L1_x, B_L1_y, C_L0, transpose_B=True, init=True)
              └─ T.copy(C_L0 → qk_workspace, enable_relu=True)
  ```

- **v6 vs v7.1 对比**：

  | 参数 | v6 | v7 (原方案) | v7.1 (独立 buffer) | 变化 |
  |------|----|----|----|------|
  | tiling 层数 | 2 层 (L0 inner split) | 3 层 (L1→L0→MMA) | **3 层 (L1→4 子 GEMM 展开)** | v7.1 无内层 L0 循环 |
  | M_L1 | S1_BLOCK×G=512 | 256 | 256 (不变) | 对齐 AscendC |
  | S2_L1 | s2BaseSize=512 | 256 | 256 (不变) | 对齐 AscendC |
  | L1 buffer 策略 | 大 buffer + 行切片 | 大 buffer + 行切片 | **独立 buffer, 无切片** | v7.1 修正 transpose_B 精度 |
  | A_L1 + B_L1 占用 | 640KB > 512KB ❌ | 320KB < 512KB ✓ | **128KB < 512KB ✓** | v7.1 最优 |
  | transpose_B 精度 | 未验证 | ❌ 精度错误 (diff=66.5) | **✅ 精度正确 (diff=0.0156)** | v7.1 验证 3 修正 |

- **L1 容量验证**：
  - v6: Q_L1(2×128KB) + K_L1(3×128KB) = 256KB + 384KB = **640KB > 512KB ❌**
  - v7 原方案: Q_L1(2×64KB) + K_L1(3×64KB) = 128KB + 192KB = **320KB < 512KB ✓** (但 transpose_B+切片不可行)
  - v7.1: A_L1_0+A_L1_1+B_L1_0+B_L1_1 = 4×32KB = **128KB < 512KB ✓** (独立 buffer, 精度正确)

- **预期收益**：BSND_BSND Cube 时间 98.83us → ~65-70us（-30~35%），主要来自 L1 容量修正和流水效率提升

- **⚠️ Stage 2 验证项**（v7.1 已部分验证, 见 §12.8）：
  1. ~~3 层嵌套循环（s2_l1 → m_l1 → s2_l0 → m_l0）编译通过~~ → **v7.1 改为 2 层循环 + 2×2 展开, 验证 3 已确认独立 buffer 方案精度正确**
  2. `T.copy(Query[m_l1 * M_L1 : ..., ...], A_L1_0)` 部分行加载（非全量加载）编译通过 — **待 Stage 2 验证**
  3. `T.copy(C_L0, qk_workspace[..., m_slice, s2_slice, :], enable_relu=True)` 2D 部分写入编译通过 — **待 Stage 2 验证**

**Q27: TND_PA_BSND block_size=16 超时问题如何修复？（v7 新增）**

- **问题背景**：msprof 实测 TND_PA_BSND（B=20, S1=3, S2=512, N1=64, block_size=16, BF16）超时（>120s），AscendC 仅需 30.44us。

- **可能原因分析**：

  | 原因 | 分析 | 可能性 |
  |------|------|--------|
  | block_size=16 导致 PA 寻址过于碎片化 | block_size=16 时 S2=512 需要 32 个 block，block_table 查询次数 32 倍于 block_size=512 | 🔴 高 |
  | BF16 dtype 的 GEMM/Vector 路径异常 | 其他 BF16 用例（BSND_PA_BSND）正常，但 block_size=16 可能触发不同路径 | 🟡 中 |
  | B=20 大 batch + block_size=16 小块导致核间负载不均 | total_tasks=20×1=20，launch_core_num=min(20,20)=20，每核 1 任务但 PA 寻址开销大 | 🟡 中 |
  | sort 缓存优化缺失（S1=3 ≤ 4） | S1=3 应触发 sort 缓存路径，但 v6 无此优化 | 🟢 低（sort 不是超时原因） |
  | TND 前缀和 + PA block_table 交互问题 | TND query + PA key 混合 layout 可能触发边界问题 | 🟡 中 |

- **v7.4 修复方案（Phase 3 验证后确定, 见 §12.12）**：

  **根因确认**: PA + block_size=16 时 `BLOCKS_PER_TILE = BLOCK_N/block_size = 256/16 = 16`, 每个 tile 需 16 次 PA gather, 寻址开销过大 → 超时. 附加问题: TOP_K=2048 > S2=512, 需从 512 个元素选 2048 个.

  | 修复项 | v7.4 方案 | 对齐 AscendC |
  |--------|----------|-------------|
  | block_size<64 超时 | `if is_pa and block_size<64: BLOCK_N=min(BLOCK_N,128)` → BLOCKS_PER_TILE=8 (原 16) | `S2_BASE_SIZE=512` 固定, PA gather 逐 block 处理 |
  | TOP_K > S2 | §8.8 步骤 2.5 的 -inf 填充 + `topk_a_ub` 初始化为 -inf, 输出区由 `T.tile.select(-1)` 填 -1 | `Duplicate(sortScoreUb, NEG_INF, cuS2LenVecAlign)` |
  | cu_s2_len 动态计算 | `cu_s2_len = min(cu_real_ac_seq - s2_start, S2_VEC_BLOCK)`, causal mask 时 `cu_real_ac_seq = act_k-(act_q-s1_idx)` | `cuRealAcSeq = actS2Size-(actS1Size-cuS1BeginIdxPerAiv)` |

  ```python
  # v7.4: block_size < 64 时减小 BLOCK_N (Phase 3 边界 Case 1)
  is_pa = (layout_key == "PA_BSND")
  if is_pa and block_size < 64:
      BLOCK_N = min(BLOCK_N, 128)  # BLOCKS_PER_TILE = 128/16 = 8

  # v7.4: topk_a_ub 初始化为 -inf, 确保 TOP_K > S2 时未选位置输出 -1
  T.tile.fill(topk_a_ub, -T.infinity(calc_dtype))
  ```

- **⚠️ Stage 2 验证项**：
  1. 单独运行 TND_PA_BSND 用例（block_size=16），确认超时已修复
  2. 确认 `topk_a_ub` 初始化为 -inf 后, TOP_K > S2 场景输出正确（有效 index + -1 填充）
  3. 检查 `cu_s2_len` 动态计算在 causal mask (sparse_mode=3) 和无 mask (sparse_mode=0) 两个分支的正确性

### 10.3 已知约束

1. ~~一期不支持 TND layout~~ → **TND 已纳入一期**（v4: kernel 直接消费 TND strided DMA, §4.9），支持 BSND/TND × BSND/PA_BSND/TND 全组合
2. **weights dtype 限制**：仅 bfloat16/float16, **绝对禁止 float32**（用户明确）
3. **N2 固定为 1**：key 仅有 1 个 head, 通过 GQA group (G=N1/N2) 广播
4. **block_table 仅 PA 场景必传**：BSND/TND key 不需要 block_table
5. **return_value 一期支持**：sparse_values 输出（仅在 layout_key != PA_BSND 时）
6. **S2 范围**：1 ~ 131072；S2 ≤ 16384 用单次 topk 路径，S2 > 16384 用分段 topk 路径
7. **S1 范围**：1 ~ 8192（扩展，前版 4096 不足）
8. **block_size 范围**：16 ~ 1024（16 整数倍），768/1024 均支持（G=64 时需 inner split）
9. **sparse_count 范围**：1 ~ 2048（任意整数，非 2 的幂次亦可）
10. **TND act_seq**：必须为前缀和（非递减），host wrapper 负责转换

### 10.4 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| **UB 溢出 (v4 BLOCK_M×BLOCK_N 全尺寸)** | v4 qk_ub=[512,512]×4=1MB 或 weight_2d=[512,512]×4=1MB | 编译失败/segfault | **v5 已修正**: groupInner=16 分块, qk_ub/weight_2d=[16,512]=32KB (§4.5) |
| **L0C 溢出 (v4 BLOCK_M×BLOCK_N 全尺寸)** | v4 C_L0=[512,512]×4=1MB 或 [128,512]×4=256KB | 编译失败/segfault | **v5 已修正**: L0 inner split (M_L0=128,N_L0=128), C_L0=[128,128]=64KB (§4.6) |
| UB 溢出 (单次 topk) | MAX_S2 > 16384 + score_accum + sort_tmp 超 192KB | 编译失败 | 切换分段 topk 路径 (§4.5/§5.2) |
| UB 溢出 (topk sort_tmp 遗漏) | 仅计 score_accum, 忘计 sort_tmp | 运行时 segfault | sort_tmp = MAX_S2 × 6B (float32), 必须计入 UB 预算 (§4.5) |
| **UB 溢出 (T.Pipelined ring-buffer)** | T.Pipelined num_stages>2 + UB buffer 在 pipeline body 内分配 | 运行时 segfault | UB buffer 在 pipeline body 外分配, num_stages=2 (v6: row_expand_mul 后 34KB, body 外不被 ring-buffer, 可尝试 3-5) |
| **T.gemm_v0 2D 行切片 + transpose_B 精度错误** (v5 风险, v7.1 已修正) | `T.gemm_v0(Q_L1[m:m+128, :], ..., transpose_B=True)` 精度错误 (diff=66.5) | 结果错误 (非编译错误) | **v7.1 已修正**: 改用独立 L1 buffer (A_L1_0/A_L1_1/B_L1_0/B_L1_1), 无切片, 精度正确 (diff=0.0156, 验证 3) |
| **L0C→GM 部分写入不支持** (v5 风险) | `T.copy(C_L0, qk_workspace[..., m_slice, :], enable_relu=True)` 失败 | 编译报错 | fallback: C_L0 完整 copy 到临时 UB, 再切片 copy 到 GM (增加一次 UB 中转) |
| **act_seq 标量读取不支持** (v5 缺陷 #6) | `actual_seq_q[b-1]` GM tensor 标量读取失败 | 编译报错 | fallback: host wrapper 预处理前缀和→per-batch 偏移 (§4.9.3) |
| **v6: row_expand_mul dst≠src0** (v6 风险) | `row_expand_mul_experiment(qk_ub, other_buf, weight_ub)` dst≠src0 | 结果错误/编译失败 | 确保 dst=src0 (原地乘), 参考 xattention.py:901-913 同款用法 |
| **v6: reduce_sum clear=False 未初始化 dst** (v6 风险) | scores_accum 未 fill(0) 就用 clear=False | 结果错误 (累加脏数据) | 循环前必须 `T.tile.fill(scores_accum, 0)` (§8.5) |
| **v6: cube_core_num 不可用** (v6 风险) | `torch.npu.get_device_properties("npu").cube_core_num` 失败 | host 报错 | fallback: 硬编码 NUM_CORES=24 (A2) 或通过环境变量传入 |
| topk src 动态 shape | score_accum 用动态 S2 | 编译报错 | 用 MAX_S2 (单次) 或 SEGMENT_SIZE (分段) 编译期常量, actual_num 传运行时值 |
| block_table 间接寻址失败 | block_table[b,s2_block] 返回 -1 | 越界访问 | Kernel 内 if k_block_id >= 0 判断 |
| cross_flag 死锁 | set/wait flag ID 不匹配 | Kernel hang | v5: 使用 AUTO_CV_SYNC 自动管理; fallback 确保 set/wait 相同 flag ID |
| **TND 前缀和越界** | 最后 batch 的最后 s1_tile 越过 T1 | 越界访问 | `if s1_tile * s1BaseSize < act_q_batch:` 条件跳过; host 确保 T1 padding (§4.9.3) |
| **TND strided DMA 切片长度动态** | `T.copy(Query[t_start:t_end, ...])` t_end 动态 | 编译报错 | 切片长度固定 s1BaseSize, 起始动态: `Query[t_start : t_start+s1BaseSize, ...]` (§4.9.3) |
| **Fixed Core 小任务空转** (v5 缺陷 #5) | total_tasks < NUM_CORES, 空闲核空转 | 性能浪费 | v5: launch_core_num = min(total_tasks, NUM_CORES) (§5.2) |
| **Fixed Core 尾块越界** | task_id >= total_tasks | 越界访问 | `if task_id < total_tasks:` 条件保护 (§5.2) |
| TND_PA_BSND 混合 layout | query TND key PA_BSND 处理不一致 | 结果错误 | kernel 内 layout_query/layout_key 分支处理 (§4.9.5) |
| 分段 topk 索引未转 global | segment_topk 的 local_index 直接输出 | 索引错误 | 段 topk 后 local_index += seg_start (§5.2 路径 B) |
| merge_sort 输入未排序 | merge_sort 输入非降序 | 合并结果错误 | 确保每段 topk 输出已降序（topk 保证） |
| T.copy enable_relu 无效 | 用 T.copy (非 npu_copy_v2) | ReLU 不执行 | T.copy 已导出为 npu_copy_v2 (__init__.py:53), enable_relu 可用 |
| block_size 非整除 S2 | 如 block_size=768, S2=20481 | 尾块越界 | T.ceildiv + valid_n = min(BLOCK_N, act_k - s2_start) |
| **v7: Cube L1 超限** (v6 隐患) | v6 Q_L1+K_L1=640KB > 512KB L1 | L1 数据驱逐到 GM, Cube 慢 64% | **v7 已修正**: M_L1=256, S2_L1=256, L1 总占用 320KB < 512KB (§5.2.1) |
| **v7.2: L0C→UB 直接跨级访问** (缺口 1) | `T.copy(C_L0, ub_buffer)` 尝试 L0C → UB 直搬 | codegen 错误 / 运行时 segfault | **禁止**: L0C 是 Cube 专用 accumulator, Vector 核不能直接访问. 正确路径: L0C → GM workspace → UB (§8.5, 缺口 1) |
| **v7.2: T.Scope("V") 用 `as (cid, _)`** (缺口 1) | Expert 模式 `with T.Kernel(blocks, is_npu=True) as (cid, _):` + T.Scope("V") | codegen 报 `Find undefined Variable _` | **正确**: 用 `as (cid, vid)`, T.Scope("V") 内 codegen 引用 vid (§8.5, 缺口 1) |
| **v7.2: sort 缓存 UB 超限** (缺口 2) | v7.1 原方案 UB=216.4KB > 192KB | 编译失败 / 运行时 segfault | **v7.2 方案 C**: VID_S1=1 + buffer 复用, UB=184.4KB ✓ (§8.8, 缺口 2) |
| **v7.2: SparseTopK 未截断** (缺口 6) | `T.tile.merge_sort(global_topk_ub, global_topk_ub, merge_output)` 输出 = 输入之和 | 输出超出 global_topk_ub 的 16KB 分配, 越界写入 | **v7.2**: merge_sort 后用 `T.copy(merge_output[0:BASE_TOPK*2], global_topk_ub)` 显式截断 (§8.8, 缺口 6) |
| **v7.2: sort 稳定性误判为问题** (缺口 3) | 误以为 sort 稳定性差异是精度失败根因, 在 sort 后加稳定化处理 | 不必要的复杂度, 不解决真正问题 | **正确理解**: sort 稳定性差异是正常行为, check_result 已处理. 真正问题是 score 计算精度 (§9.3, 缺口 3) |
| **v7: sort 缓存 UB 不足** (v7 新增) | sort 缓存 48KB + Kernel 2 其他 buffer 超 192KB | 编译失败/segfault | 缓存大小 SORT_CACHE_SIZE=4 固定, S2_VEC_BLOCK=512 固定, 总 48KB < 192KB ✓ |
| **v7: 4-way merge_sort 不可用** (v7 风险, v7.1 已修正) | ~~T.tile.merge_sort 不支持 way=4 或 3D buffer 切片~~ | ~~编译报错~~ | **v7.1 已修正**: way 参数不存在, 改用位置参数 + if-elif 分支; sort 切片 dst 不可行, 改用 sort_tmp + T.copy + 切片 src (验证 1+2 PASS) |
| **v7: TND_PA_BSND block_size=16 超时** (v7 新增) | block_size=16 时 PA 寻址碎片化 | 运行超时 | 排查 PA 寻址 DMA 效率, 考虑多 block 聚合加载 (§10.2 Q27) |
| **v7: 3 层 tiling 循环编译失败** (v7 风险, v7.1 已修正) | ~~4 层嵌套循环 (s2_l1→m_l1→s2_l0→m_l0) 编译器不支持~~ | ~~编译报错~~ | **v7.1 已修正**: 改为 2 层循环 (s2_l1→m_l1) + 2×2 子 GEMM 展开 (独立 buffer, 无 L1 切片), 验证 3 PASS |
| **v7.3: 4×L0C fragment 合并** (参考算子模式 1) | 4 个 `T.alloc_fragment` 被编译器合并为共享 fragment, 退化为单 L0C | 性能退化 (无 4×L0C 收益) | Stage 2 验证: get_kernel_source 检查 4 fragment 独立; 若合并, 改用 `T.alloc_L0C` (Expert 模式) 显式分配 (§4.6, §8.5) |
| **v7.3: annotate_layout 与独立 buffer 冲突** (参考算子模式 2) | `T.annotate_layout({A_L1_0: make_zn_layout})` 与 v7.1 独立 L1 buffer 方案冲突 | codegen 错误 / 精度退化 | Stage 2 验证: 先编译无 annotate_layout 基线, 再加 annotate_layout 对比精度; 若冲突, 退回 v7.2 无 layout 方案 (§8.5) |
| **v7.3: DMA 重排 flag 时序错误** (参考算子模式 3) | Wave 0 (K[0]+Q) 与 Wave 1+ (K[i]+staging) 的 flag 时序不匹配 | DMA 数据竞争 / 结果错误 | Stage 2 验证: 用 T.dump_tensor 检查 L1 buffer 内容; 若错误, 退回顺序 DMA (v7.2 基线) |
| **v7.3: MTE2∥V overlap late DMA 数据竞争** (参考算子模式 6) | late DMA 未完成时 sort 使用了 score_buf_next 的脏数据 | 结果错误 (非编译错误) | Stage 2 验证: 确保 set_flag/wait_flag 时序正确; 用 double buffer (score_buf ↔ score_buf_next); 若错误, 退回串行 DMA (v7.2 基线) |
| **v7.3: brcb_experiment 破坏 row_expand_mul 精度** (参考算子模式 7) | brcb 前缀改变了 row_expand_mul_experiment 的输入 layout | 精度退化 (diff > 0.0312) | Stage 2 验证: 先测单 row_expand_mul (v7.2 基线), 再加 brcb 前缀对比; 若退化, 保留 v7.2 单 row_expand_mul |

### 10.5 特殊场景处理

- **act_q=0（空 query）**：Kernel 跳过该 batch, 输出全 -1
- **act_k=0（空 key）**：topk 无有效元素, 输出全 -1
- **act_k < sparse_count**：topk 输出不足 k 个, 剩余填 -1
- **block_table 含 -1（无效 block）**：Kernel 内 if 判断跳过

---

## 11. 交付清单

### 11.1 目录结构

```
examples/lightning_indexer/
├── lightning_indexer.py          # 单 Kernel + CV 分离 (T.Scope("C")/"V"), @tilelang.jit (v7.4 Phase 2 确认)
├── test_lightning_indexer.py     # from lightning_indexer import ... + golden + L0 测试 + main
├── proto.yaml                    # 算子接口规格 (dtype/attr)
├── DESIGN.md                     # 本设计文档 (v7)
├── COMPARISON.md                 # TileLang vs AscendC arch22 对比分析 (v7 新增)
├── perf_benchmark_ascendc.py     # AscendC 性能基准采集脚本 (v7 新增)
├── perf_benchmark_tilelang.py    # TileLang 性能基准采集脚本 (v7 新增)
├── perf_msprof_ascendc_full/     # AscendC msprof 原始数据 (v7 新增)
├── perf_msprof_tilelang_full/    # TileLang msprof 原始数据 (v7 新增)
├── debug_log.md                  # 调试日志
├── example_lightning_indexer.py  # 旧版参考实现 (保留, 不作为交付物)
├── example_lightning_indexer_dynamic_shape.py  # 旧版动态 shape 参考
└── history_version/              # 历史备份
    ├── example_lightning_indexer_original.py.bak
    ├── example_lightning_indexer_dynamic_shape_original.py.bak
    └── design_v6_pre_perf_update.md  # v6 性能更新前备份
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档 (本文件, v7 — 性能优化更新, 基于 msprof op 实测数据) |
| `proto.yaml` | ✅ 已完成 | 算子接口规格, 覆盖门禁用 (v7: 接口不变, 同 v6) |
| `COMPARISON.md` | ✅ 已完成 | TileLang vs AscendC arch22 实现对比分析 (v7 新增) |
| `perf_benchmark_ascendc.py` | ✅ 已完成 | AscendC 性能基准采集脚本 (v7 新增) |
| `perf_benchmark_tilelang.py` | ✅ 已完成 | TileLang 性能基准采集脚本 (v7 新增) |
| `perf_msprof_ascendc_full/` | ✅ 已完成 | AscendC msprof 原始数据 (v7 新增) |
| `perf_msprof_tilelang_full/` | ✅ 已完成 | TileLang msprof 原始数据 (v7 新增) |
| `lightning_indexer.py` | ✅ 已实现 | 单 Kernel + CV 分离 (T.Scope("C")/"V") + 手动 set_flag/wait_flag, 参考 sparse_flash_attn_pa_no_cv_pipeline.py (v7.4 Phase 2 确认; v6 实现, v7 待优化) |
| `test_lightning_indexer.py` | ✅ 已实现 | import kernel + Golden (GeneralizedLI) + L0 用例 + main |

### 11.3 命名规范

- 目录名: `lightning_indexer` (snake_case)
- kernel 文件: `lightning_indexer.py`
- 测试文件: `test_lightning_indexer.py` (顶部 `from lightning_indexer import lightning_indexer`)

### 11.4 实现顺序

1. ✅ 设计文档 (DESIGN.md) + proto.yaml + L0 门槛测试计划 (本文件 §9.2)
2. ✅ kernel 实现 (`lightning_indexer.py`, 单 Kernel + CV 分离 @tilelang.jit, T.Scope("C")/"V") + 手动同步) — v6 已实现, v7.4 Phase 2 确认架构
3. ✅ 测试文件 (`test_lightning_indexer.py`): import kernel + Golden (GeneralizedLI) + L0 用例 + main — 已实现
4. ✅ L0 门槛测试通过 (精度收敛, 使用 check_result_li 集合+value 误差对比) — 已通过
5. ✅ msprof op 性能数据采集 (4 种 layout 场景) — v7 已完成
6. ✅ AscendC arch22 深度对比分析 (COMPARISON.md) — v7 已完成
7. ⬜ v7 性能优化实施 (Cube 3 层 tiling + sort 缓存 + TND_PA_BSND 修复)
8. ⬜ v7 优化后性能验证 (4/4 达标 80% 目标)
9. ⬜ 扩展分层套件 (L1 功能 / L2 异常 / Boundary 特殊值, 由 tilelang-op-test-design 场景 B 生成)
10. ⬜ 全量套件运行 (L0/L1 须通过; L2/Boundary 失败仅记录不阻塞)

### 11.5 算子 proto.yaml

见同目录 `proto.yaml` 文件。

**一致性约束**：
- `inputs[].dtype` 与 §9.3 精度表的 dtype 行一致（float16, bfloat16 for query/key/weights; int32 for actual_seq/block_table）
- `outputs[].dtype`: sparse_indices = int32; sparse_values = **bfloat16/float16**（v3 修正: 与 query/key 一致, 非 int32）
- `attrs[].name` 覆盖所有影响计算路径的属性: layout_query, layout_key, sparse_count, sparse_mode, pre_tokens, next_tokens, return_value
- weights dtype **不含 float32**（用户明确禁止）
- v4: layout_query/layout_key 的 TND 已从"二期"提升到"一期"（kernel 直接消费 TND）

---

## 12. 性能优化预期（v4 新增）

### 12.1 4 项架构级优化 + 5 项 v6 API 优化 + 6 项 v7 性能优化的预期收益

| 优化 | 影响场景 | 预期收益 | 收益来源 | 验证方法 |
|------|---------|---------|---------|---------|
| 1: T.Pipelined 核间流水 | 所有 Kernel 1 场景 | 15-25% | Cube/Vector 重叠隐藏延迟, cross_interval=2 减少同步 | msprof: Cube/Vector pipe utilization 重叠度 |
| 2: Fixed Core 任务分配 | 大 B×S1×S2 场景 | 5-10% | workspace 固定 3MB (v3 6MB+), L2 cache 命中率提升; **v5: min(total_tasks,NUM_CORES) 小任务不空转** | msprof: L2 cache hit rate, workspace 访问延迟 |
| 3: kernel 直接消费 TND | TND 场景 (24% 用例) | 15-30% (TND 场景) | 消除 host wrapper Transpose/Copy kernel | msprof: main_kernel 外无 Transpose kernel |
| 4: broadcast + 整 tile mul | 所有 Kernel 1 场景 | 10-15% | **v5: groupInner=16 分块后, s1×outerG 次循环 (如 32 次) vs v3 的 BLOCK_M=500 次逐行 mul** | msprof: aiv_vec_ratio 提升 (v3 基准 65%) |
| **5: row_expand_mul_experiment** (v6) | 所有 Kernel 1 场景 | **5-10%** | **v6: 融合 Brcb+Mul, 移除 broadcast+mul 两步, 移除 weight_2d(32KB UB)** | msprof: aiv_vec_ratio 提升, UB 占用降低 |
| **6: reduce_sum(clear=False)** (v6) | 所有 Kernel 1 场景 | **3-5%** | **v6: 融合 reduce+累加, 移除 add 步骤, 移除 scores_partial(2KB UB)** | msprof: aiv_vec_ratio 提升 |
| **7: 动态核数 cube_core_num** (v6) | A3/950 等非 A2 芯片 | **适配性** | **v6: 适配 A3(20核)/950, 不再硬编码 24** | 跨芯片测试 |
| **8: pipe_barrier("v")** (v6) | 所有 Kernel 1 场景 | **2-3%** | **v6: Vector 管线轻量同步, 不等 Cube/MTE2/MTE3** | msprof: 同步开销降低 |
| **9: Cube 3 层 tiling 对齐 AscendC** (v7) | BSND_BSND, TND_TND | **Cube 时间 30-35%** | **v7: M_L1=256, S2_L1=256 对齐 AscendC, 修正 v6 L1 超限(640KB→320KB), 提升流水效率** | msprof: Cube 时间降低, L1 容量验证 |
| **10: sort 缓存优化** (v7) | S1≤4 推理场景 | **Vec0 时间 10-15%** | **v7: SortedBasicBlock 4-way merge, 减少 75% merge_sort 调用** | msprof: Vec0 利用率提升 (v6: 48.4% → v7: ~58%) |
| **11: TND_PA_BSND block_size=16 修复** (v7) | TND_PA_BSND | **功能补齐** | **v7: 排查 PA 寻址碎片化, 修复超时问题** | msprof: TND_PA_BSND 不超时 |
| **12: num_stages 提升** (v7) | 所有 Kernel 1 场景 | **5-10%** | **v7: num_stages=3-4, 提升 Cube/Vector 流水深度** | msprof: pipeline 重叠度提升 |
| **合计预期** | — | **v6: 35-55% + v7: 20-35%** | — | 对比 v3 实测 vs v7 实测 |

> **⚠️ v6 num_stages 建议（优化 #2 后更新）**：
> - **v6 UB=34KB**（v5: 68KB），buffer 在 pipeline body 外不被 ring-buffer
> - num_stages 不受 UB×num_stages 限制，仅受 GM workspace ring-buffer slots 和 pipeline 有效性限制
> - **v6 建议**: num_stages=2（保守基线）→ Stage 2 实测后尝试 **3-4**
> - num_stages=3 参考 `matmul_add_pipeline.py:46`（已验证可运行）
> - 参考 `flash_attn_optimize.md` 的 "KNOWN BROKEN" 教训：UB buffer 不在 pipeline body 内分配（v6 已确保）

### 12.2 各 msprof 基准场景的实测与预期表现（v7 更新，基于 msprof op 实测数据）

| Case | AscendC(us) | 80%目标(us) | v6 实测(us) | v6 达标? | v7 预期(us) | v7 达标? | 主要优化 |
|------|------------|------------|------------|---------|------------|---------|---------|
| BSND_BSND (B=16,S1=5,S2=3072,N1=64,G=64) | 74.26 | 92.83 | 111.64 | ❌ (1.50x) | ≤85 | ✅ | v7#9 Cube tiling + v7#10 sort 缓存 + v7#12 num_stages |
| BSND_PA_BSND (B=2,S1=1,S2=2048,N1=8,G=8) | 20.20 | 25.25 | **18.78** | ✅ (0.93x) | ~18-19 | ✅ | 已达标, 保持不退化 |
| TND_TND (B=8,S1=5,S2=3072,N1=24,G=24) | 40.54 | 50.67 | 55.10 | ❌ (1.36x) | ≤48 | ✅ | v7#9 Cube tiling + v7#10 sort 缓存 |
| TND_PA_BSND (B=20,S1=3,S2=512,N1=64,G=64,bs=16) | 30.44 | 38.05 | 超时 | ⏳ | ≤35 | ✅(待验证) | v7#11 block_size=16 修复 |

> **v7 关键改进**：
> - BSND_BSND: v6 慢 1.50x → v7 预期达标（Cube tiling 对齐 + sort 缓存优化）
> - TND_TND: v6 慢 1.36x → v7 预期达标（同上）
> - BSND_PA_BSND: v6 已达标（0.93x），v7 保持不退化
> - TND_PA_BSND: v6 超时 → v7 修复 block_size=16 问题后预期达标

> **BSND_PA_BSND 场景分析**：B=2, S1=1, N1=8 数据量极小，TileLang 用 8 核（自适应）vs AscendC 20 核，task 分配更合理，已比 AscendC 快 7%。v7 优化对此场景收益有限，保持不退化即可。

### 12.3 实现级优化候选（Stage 2 实施时评估）+ Stage 3 fallback 参考

**Stage 2 候选**（Developer 模式内优化）：

| 优化 | 来源 | 预期收益 | 实施条件 | 风险 |
|------|------|---------|---------|------|
| 5: T.tile.mul_add_dst 融合 | performance-antipatterns.md §"mul+add" | 3-5% | **v6: 已被 row_expand_mul + reduce_sum(clear=False) 替代, 降级为 fallback** | 改变舍入路径, 需重新精度验证 |
| 6: T.annotate_layout (ZN/NZ 分形) | flash_attn_optimize.md §1 | 5-10% | L1 分形优化, 减少 bank 冲突 | 需验证 GEMM 分形对齐 |
| 7: S1==1 decode 窄块 | flash_attn_optimize.md §9(b) | 50%+ (decode 场景) | S1=1 场景用 block_M=2 或 4 | 需额外 kernel 变体 |
| 8: num_stages=3-4 | v6 UB=34KB 余量 | 5-10% | buffer 在 body 外, GM workspace 多 slot | 需实测 pipeline 有效性 |

**Stage 3 fallback 参考**（HISA/xattention Expert 模式优化, v6 标注）：

> 以下优化来自 `examples/HISA/paged_block_sparse_mqa_attn_expert.py` 和 `examples/xattention/xattention.py` 的 Expert 模式实现。**v6 Developer 模式不实施**，作为 Stage 3 性能未达标时的 fallback 路径。

| Expert 优化 | 来源 | 预期收益 | 实施条件 | 风险 |
|-------------|------|---------|---------|------|
| **E1: 4×K L1 slots** | HISA | 10-15% | K 数据四缓冲, 比 AscendC 三缓冲更多 | L1 容量紧张, 需重新规划 |
| **E2: 4×L0C slots** | HISA | 5-10% | 每个 MMA 独立 L0C, 无 FIX↔M 争用 | L0C 容量紧张 (4×64KB=256KB>128KB, 需更小 C_L0) |
| **E3: MTE2 ∥ V overlap** | HISA | 10-15% | Late DMA 在 early mask+output 之前入队 | 需 Expert 模式手动 set_flag/wait_flag |
| **E4: Tail-fill replaces mask** | HISA | 5-10% | 尾块填充替代 mask pipeline | 需额外 tail-fill 逻辑 |
| **E5: 核分组** | xattention | 5-10% | UNSHARED_CORES + SHARED_CORES 分组 | 需 Expert 模式核间通信 |
| **E6: MEMORY_PLANNING=False** | xattention | 5-10% | 完全手动内存管理 | 编译复杂度高, UB 地址手动管理 |
| **E7: brcb_experiment 单独广播** | ascend_tile.py:742 | 2-3% | 需单独广播场景 (如 mask 广播) | row_expand_mul 已融合 Brcb, 仅特定场景 |

> **Stage 3 触发条件**：若 v6 Developer 模式实测性能未达 AscendC 80% 目标（4 场景中 ≥1 个未达标），评估 E1-E7 中收益最高的 1-2 项。参考 `examples/HISA/paged_block_sparse_mqa_attn_expert.py`（4×K L1 + 4×L0C + MTE2∥V overlap 完整 Expert 实现）。

### 12.4 性能验证计划

1. **Stage 2 完成后**：对 4 个 msprof 基准场景采集 v4 实测数据
2. **对比指标**：task_duration, aic_cube_ratio, aiv_vec_ratio, aiv_mte2_ratio, UB 带宽
3. **达标标准**：task_duration ≤ 80% 目标（4 场景中至少 3 个达标）
4. **未达标处理**：
   - 若 BSND_PA_BSND 未达标（launch overhead 主导）：评估 decode 窄块（优化 7）
   - 若其他场景未达标：评估 mul_add_dst（优化 5）和 annotate_layout（优化 6）
   - 若 T.Pipelined UB 超限：回退 num_stages=1（等同 T.serial）或减小 BLOCK_M

### 12.5 v6 优化的 Stage 2 验证清单

**v4 保留项**：
- [ ] T.Pipelined(num_stages=2, cross_interval=2) 编译通过
- [ ] T.Pipelined 运行时 UB 未超限（get_kernel_source 检查 ring-buffer）
- [ ] Fixed Core 模式 `T.Kernel(min(total_tasks,动态核数), threads=2)` 编译通过
- [ ] Fixed Core 尾块 `if task_id < total_tasks:` 正确处理
- [ ] TND strided DMA `T.copy(Query[t_start:...], Q_L1)` 编译通过
- [ ] 4 个 msprof 基准场景精度通过（L0 用例）
- [ ] 4 个 msprof 基准场景性能达标（≤ 80% 目标）

**v5 保留项**（groupInner + L0 inner split）：
- [ ] **v5: T.gemm_v0 + 2D L1 行切片** `T.gemm_v0(Q_L1[m_l0*M_L0:(m_l0+1)*M_L0, :], ...)` 编译通过
- [ ] **v5 fallback**: 若 2D 行切片不可用, Q_L1=[M_L0, D] 方案编译通过
- [ ] **v5: T.copy(C_L0, qk_workspace[..., m_slice, :], enable_relu=True)** L0C→GM 部分写入编译通过
- [ ] **v5: T.tile.fill(scores_accum, 0)** 编译通过
- [ ] **v5: groupInner=16 分块后 UB 占用 ≤ 192KB**（get_kernel_source 验证）
- [ ] **v5: L0 inner split 后 C_L0=64KB ≤ 128KB**（get_kernel_source 验证）
- [ ] **v5: groupInner=16 分块后 N1=64 场景精度通过**（v4 UB 超限的场景）
- [ ] **v5: act_seq_q[b-1] 标量读取**编译通过（缺陷 #6）
- [ ] **v5 fallback**: 若标量读取不可用, host wrapper 预处理方案编译通过
- [ ] **v5: launch_core_num=min(total_tasks,NUM_CORES)** 小任务场景不空转
- [ ] **v5: sparse_values 精度验证**（return_value=True 用例, FP16/BF16 容差）

**v6 新增项**（row_expand_mul + reduce_sum merge + 动态核数 + pipe_barrier）：
- [ ] **v6: T.tile.row_expand_mul_experiment(qk_ub, qk_ub, weight_ub)** 编译通过（优化 #1）
- [ ] **v6: row_expand_mul dst=src0 原地乘** 结果正确（精度验证）
- [ ] **v6: T.reduce_sum(qk_ub, scores_accum, dim=0, clear=False)** 编译通过（优化 #2）
- [ ] **v6: reduce_sum(clear=False) merge 语义** 结果正确（scores_accum 累加验证）
- [ ] **v6: 移除 weight_2d 和 scores_partial 后 UB=34KB**（get_kernel_source 验证）
- [ ] **v6: torch.npu.get_device_properties("npu").cube_core_num** host 侧获取成功（优化 #3）
- [ ] **v6: 动态核数** 在 A2(24核) 和 A3(20核) 上均正确运行
- [ ] **v6: T.pipe_barrier("v")** 编译通过（优化 #4）
- [ ] **v6: pipe_barrier("v")** 不导致死锁（Vector 管线内同步验证）
- [ ] **v6: Vector 侧指令数 4 条/iteration**（get_kernel_source 验证, v5 是 6 条）
- [ ] **v6: num_stages=3 实测**（UB=34KB, body 外 buffer, GM workspace 3 slot）
- [ ] **v6 fallback**: 若 row_expand_mul 不可用, 回退 v5 broadcast+mul 方案编译通过
- [ ] **v6 fallback**: 若 reduce_sum(clear=False) 不可用, 回退 v5 reduce+add 方案编译通过

### 12.6 v7 性能基准与瓶颈分析（msprof op 实测数据驱动）

> **v7 新增**：基于 msprof op 工具采集的 AscendC 和 TileLang kernel 级性能数据，驱动 v7 性能优化设计。数据采集时间 2026-07-21，设备 Ascend910B3，CANN 9.0。

#### 12.6.1 msprof op 采集方式

```bash
# AscendC 官方算子
msprof op --kernel-name="Lightning" --output=./perf_msprof_ascendc_full/<case> \
  --application="python3 perf_benchmark_ascendc.py --case=<case> --iters=30"

# TileLang 实现
msprof op --kernel-name="main" --output=./perf_msprof_tilelang_full/<case> \
  --application="python3 perf_benchmark_tilelang.py --case=<case> --iters=30"
```

性能脚本：`examples/lightning_indexer/perf_benchmark_ascendc.py` / `perf_benchmark_tilelang.py`

#### 12.6.2 Task Duration 对比

| 用例 | B | S1 | S2 | N1 | G | block_size | dtype | mode | AscendC (us) | TileLang v6 (us) | 比值 | 80% 目标 (us) | 达标 |
|------|---|----|----|----|---|-----------|-------|------|-------------|------------------|------|-------------|------|
| BSND_BSND | 16 | 5 | 3072 | 64 | 64 | 128 | FP16 | 3 | 74.26 | 111.64 | 1.50x | ≤92.83 | ❌ |
| BSND_PA_BSND | 2 | 1 | 2048 | 8 | 8 | 128 | FP16 | 3 | 20.20 | **18.78** | 0.93x | ≤25.25 | ✅ |
| TND_TND | 8 | 5 | 3072 | 24 | 24 | 256 | FP16 | 0 | 40.54 | 55.10 | 1.36x | ≤50.67 | ❌ |
| TND_PA_BSND | 20 | 3 | 512 | 64 | 64 | 16 | BF16 | 0 | 30.44 | 超时 | — | ≤38.05 | ⏳ |

**达标情况**：1/3 达标（BSND_PA_BSND），2/3 未达标，1 个超时。

#### 12.6.3 ArithmeticUtilization 详细数据

| 用例 | 实现 | Cube (us) | Vec0 (us) | Vec1 (us) | Cube% | Vec0% | Vec1% |
|------|------|----------|----------|----------|-------|-------|-------|
| BSND_BSND | AscendC | 60.14 | 69.26 | 67.98 | 20.5% | 63.1% | 43.5% |
| | TileLang v6 | 98.83 | 102.49 | 102.50 | 21.1% | 48.4% | 33.1% |
| BSND_PA_BSND | AscendC | 9.16 | 14.61 | 14.38 | 0.5% | 13.2% | 11.0% |
| | TileLang v6 | 10.80 | 12.19 | 12.23 | 6.7% | 17.4% | 7.2% |
| TND_TND | AscendC | 26.56 | 33.74 | 33.06 | 7.1% | 43.7% | 31.0% |
| | TileLang v6 | 42.80 | 47.04 | 47.04 | 10.4% | 36.0% | 26.0% |

#### 12.6.4 核数对比

| 用例 | AscendC Block Dim | AscendC Mix Block Dim | TileLang Block Dim | TileLang Mix Block Dim |
|------|-------------------|----------------------|-------------------|----------------------|
| BSND_BSND | 20 | 40 | 20 | 40 |
| BSND_PA_BSND | 20 | 40 | 8 | 16 |
| TND_TND | 20 | 40 | 20 | 40 |

> **BSND_PA_BSND 核数差异**：TileLang 用 8 核 vs AscendC 20 核，自适应核数分配更合理（数据量小，20 核空转），这是 TileLang 在此用例上比 AscendC 快的原因。

#### 12.6.5 瓶颈分析

**BSND_BSND（主要瓶颈，差 37.38 us）**：

| 指标 | AscendC | TileLang v6 | 差距 | 原因分析 |
|------|---------|-------------|------|---------|
| Cube 时间 | 60.14 us | 98.83 us | +38.69 us (慢 64%) | **主要差距来源**：v6 M_L1=512 导致 L1 超限(640KB>512KB)，数据驱逐到 GM |
| Vec0 时间 | 69.26 us | 102.49 us | +33.23 us (慢 48%) | sort 缓存优化缺失，更多 merge_sort 空闲 |
| Vec0 利用率 | 63.1% | 48.4% | -14.7% | sort 缓存缺失 + 同步开销 |
| Cube 利用率 | 20.5% | 21.1% | +0.6% | 利用率相近，但 tiling/流水线效率差异导致时间差距大 |

**TND_TND（差 14.56 us）**：

| 指标 | AscendC | TileLang v6 | 差距 | 原因分析 |
|------|---------|-------------|------|---------|
| Cube 时间 | 26.56 us | 42.80 us | +16.24 us (慢 61%) | 同 BSND_BSND，L1 超限 + tiling 差异 |
| Vec0 时间 | 33.74 us | 47.04 us | +13.30 us (慢 39%) | sort 缓存缺失 |

**BSND_PA_BSND（已达标，快 1.42 us）**：

| 指标 | AscendC | TileLang v6 | 差距 | 原因分析 |
|------|---------|-------------|------|---------|
| Cube 时间 | 9.16 us | 10.80 us | +1.64 us | 数据量小，Cube 差距不大 |
| Vec0 时间 | 14.61 us | 12.19 us | -2.42 us | **TileLang 更快**：8 核 vs 20 核，task 分配更合理 |
| 核数 | 20 | 8 | -12 | TileLang 自适应核数更优 |

#### 12.6.6 v7 性能优化方向

| 优先级 | 优化项 | 预期收益 | 影响用例 | 实施章节 |
|--------|--------|---------|---------|---------|
| P0 | Cube GEMM tiling 对齐 AscendC — M_L1=256, S2_L1=256 | Cube 时间 ~30-35% | BSND_BSND, TND_TND | §5.2.1, §8.5 |
| P0 | sort 缓存优化 — S1≤4 时缓存 4 块再 4-way merge (v7.2 方案 C: VID_S1=1) | merge_sort 次数 75% | BSND_BSND, TND_TND | §8.8 |
| P0 | TND_PA_BSND 排查 — block_size=16 超时 | 功能补齐 | TND_PA_BSND | §10.2 Q27 |
| P0 | SparseTopK 显式截断 (v7.2 缺口 6) | 防止越界写入 | 全部 sort 缓存路径 | §8.8 |
| P1 | Vector 同步优化 — 减少 barrier_all | Vector 空闲时间 | 全部 | §7, §8.5 |
| P1 | 数据范围特化 — G=64 优先 | G=64 场景优化 | BSND_BSND, TND_PA_BSND | §5.2, §8.5 |
| P1 | **MTE2∥V double buffer** (v7.2 缺口 6 遗漏 5) | Kernel 1 Vector 利用率提升 | BSND_BSND, TND_TND | §8.5 (Stage 3 候选) |
| P1 | **MergeSort 3072 阈值自适应** (v7.2 缺口 6 遗漏 3) | Over2K 场景性能 (virTopK 最大 8192) | sort 缓存路径 | §8.8 (Stage 3 候选) |
| P1 | **SortedBasicBlock_ 与 globalTopkUb_ 内存复用** (v7.2 缺口 6 遗漏 2) | 省 16KB UB | sort 缓存路径 | §8.8 (方案 E, 可选) |
| P2 | num_stages 提升 — 2→3-4 | Cube/Vector 流水深度 | 全部 | §6, §12.6 |
| P2 | **ifExhaustedSuspension=false 行为验证** (v7.2 缺口 6 遗漏 4) | 不等长队列 merge 性能 | sort 缓存路径 | §8.8 (Stage 3 候选) |
| **v7.3 P0** | **4×L0C 消除 MMA/copy 争用** (参考算子模式 1, HISA) | Cube 时间 ~15-20% (消除 FIX↔M flag 争用) | BSND_BSND, TND_TND | §4.6, §8.5, §12.11 |
| **v7.3 P0** | **make_zn/make_nz_layout** (参考算子模式 2, fa_opt) | Cube 时间 ~5-10% (减少 L1 bank conflict) | BSND_BSND, TND_TND | §8.5, §12.11 |
| **v7.3 P0** | **DMA 重排 K[0]+Q 优先** (参考算子模式 3, HISA Wave 0) | Cube 时间 ~5-10% (减少 Wave 0 DMA 延迟) | BSND_BSND, TND_TND | §8.5, §12.11 |
| **v7.3 P1** | **cross_interval=2 减少跨核同步** (参考算子模式 5, fa_opt) | Vector 空闲时间 ~50% (跨核同步减半) | 全部 | §6.3, §8.5 (✅ 已采纳) |
| **v7.3 P1** | **MTE2∥V overlap** (参考算子模式 6, HISA) | Vector 利用率提升 ≥ 5% | BSND_BSND, TND_TND | §8.8, §12.11 |
| **v7.3 P1** | **brcb+row_expand_mul 三步模式** (参考算子模式 7, xattention) | pipe_barrier 数量减少 | 全部 | §8.5, §12.11 |
| **v7.3 P2** | **reduce_sum clear=True 变体** (参考算子模式 8, HISA) | 单次归约场景指令数减少 | — | §8.5 (✅ 已整合说明) |
| **v7.3 P2** | **tail-fill 替代 mask** (参考算子模式 9, HISA) | 尾块边界指令数减少 | sort 缓存路径 | §8.8, §12.11 |
| **v7.3 P2** | **signal ID 命名规范** (参考算子模式 10, HISA) | 代码可维护性提升 | 全部 | §7.2, §12.11 |
| **v7.3 P2** | **3-slot PRE_LAUNCH 流水** (参考算子模式 12, xattention_paged) | Cube/Vector 流水深度 (Stage 3 候选) | 全部 | §6.3, §12.11 (Stage 3 候选) |

> **v7.2 缺口 6 新增优化方向说明（AscendC 代码级对照, 6 个遗漏点）**：
>
> | # | 遗漏点 | 优先级 | v7.2 处理 | 说明 |
> |---|--------|--------|----------|------|
> | 1 | SparseTopK 显式截断 | 🔴 高 | ✅ 已处理 (§8.8) | merge_sort 后 `T.copy(merge_output[0:BASE_TOPK*2], global_topk_ub)` |
> | 2 | SortedBasicBlock_ 与 globalTopkUb_ 内存复用 | 🟡 中 | 🟡 方案 E (可选) | 用 `T.annotate_address` 实现 buffer alias, 省 16KB UB. v7.2 方案 C 已解决 UB 超限, 此项为 Stage 3 可选优化 |
> | 3 | MergeSort 3072 阈值自适应 | 🟡 中 | 🟡 Stage 3 候选 | AscendC `mrgDstNum > 3072` 时拆 3 段做 4-way merge. v7.2 未实现, 影响 Over2K 场景 (virTopK 最大 8192) 标准路径性能 |
> | 4 | ifExhaustedSuspension = false | 🟡 中 | 🟡 Stage 3 候选 | AscendC 所有 MrgSort 均设此参数. v7.2 的 T.tile.merge_sort 是否有等效行为未验证 |
> | 5 | Vector 侧 MTE2∥V double buffer | 🟡 中 | 🟡 Stage 3 候选 | AscendC outerG 循环内 pingpong double buffer (MTE2∥V 并行). v7.2 的 T.Pipelined 是 CV overlap, 不覆盖 Vector 内部 MTE2∥V. 参考 `examples/HISA/paged_block_sparse_mqa_attn_expert.py` |
> | 6 | ProcessLD 跨核归约 | 🟢 低 | ✅ 不是问题 | v7.2 双 Kernel 架构天然避免跨核归约, 非遗漏 |

#### 12.6.7 v7 预期性能目标

| 用例 | v6 实测 (us) | v7 优化项 | v7 预期 (us) | 80% 目标 (us) | 预期达标 |
|------|------------|---------|------------|------------|---------|
| BSND_BSND | 111.64 | Cube tiling + sort 缓存 + num_stages | ≤85 | ≤92.83 | ✅ |
| TND_TND | 55.10 | Cube tiling + sort 缓存 | ≤48 | ≤50.67 | ✅ |
| BSND_PA_BSND | 18.78 | 保持（已达标） | ~18-19 | ≤25.25 | ✅ |
| TND_PA_BSND | 超时 | block_size=16 修复 | ≤35 | ≤38.05 | ✅（待验证） |

> **v7 目标**：4/4 用例达到 AscendC 80% 性能目标。

#### 12.6.8 性能数据文件

- AscendC msprof 原始数据: `examples/lightning_indexer/perf_msprof_ascendc_full/<case>/`
- TileLang msprof 原始数据: `examples/lightning_indexer/perf_msprof_tilelang_full/<case>/`
- 性能基准脚本: `examples/lightning_indexer/perf_benchmark_ascendc.py` / `perf_benchmark_tilelang.py`
- 实现对比分析: `examples/lightning_indexer/COMPARISON.md`

### 12.7 v7 优化的 Stage 2 验证清单（v7.1 更新: API 验证后修正）

**v7.1 已验证项**（API 验证 1-5, 见 §12.8, ✅ = 已通过）：
- [x] **v7.1: 独立 L1 buffer 方案** — `T.gemm_v0(A_L1_x, B_L1_y, transpose_B=True)` 无切片, 精度正确 (验证 3 PASS, diff=0.0156)
- [x] **v7.1: T.tile.merge_sort 4-way** — 位置参数调用 `merge_sort(dst, s0, s1, s2, s3)`, 不存在 way 参数 (验证 1 PASS)
- [x] **v7.1: merge_sort 切片 src** — 1D 平铺 buffer 的切片作为 src 可行 (验证 1+2 PASS)
- [x] **v7.1: sort→sort_tmp→T.copy** — T.tile.sort 不支持切片 dst, 用 sort_tmp 中转 + T.copy 到 1D cache (验证 2 PASS)
- [x] **v7.1: T.Pipelined num_stages=3-4** — 编译通过且精度正确 (验证 4 PASS, diff=0.0312), buffer 须在 body 外
- [x] **v7.1: threads=2 解包** — `as (cid)` 1 个值 (验证 4 确认)
- [x] **v7.1: pipe_barrier("v")** — 精度与 barrier_all 一致 (验证 5 PASS, diff=0.0312)

**v7.1 待 Stage 2 验证项**（设计层已修正, 需 Stage 2 编译+精度验证）：
- [ ] **v7.1: 独立 L1 buffer 编译** — A_L1_0/A_L1_1/B_L1_0/B_L1_1 各 [128,128], 4 个 alloc_shared 编译通过
- [ ] **v7.1: L1 容量验证** — 4×32KB = 128KB < 512KB（get_kernel_source 验证）
- [ ] **v7.1: T.copy(Query[m_l1*M_L1:..., ...], A_L1_0)** — 部分行加载编译通过
- [ ] **v7.1: T.copy(C_L0, qk_workspace[..., m_slice, s2_slice, :], enable_relu=True)** — 2D 部分写入编译通过
- [ ] **v7.1: sort 缓存优化编译** — 1D 平铺 + sort_tmp + T.copy + merge_sort 切片 src 编译通过（§8.8）
- [ ] **v7.1: sort 缓存 UB 占用** — 52KB < 192KB（get_kernel_source 验证）
- [ ] **v7.1: 动态路数 if-elif 分支** — 1/2/3/4-way merge_sort 分支编译通过
- [ ] **v7: TND_PA_BSND block_size=16** — 超时问题修复（§10.2 Q27, 仍待排查）
- [ ] **v7: 4 个 msprof 基准场景性能达标** — ≤ 80% 目标（4/4 达标）

### 12.8 API 验证结果（v7.1 新增）

> **验证时间**: 2026-07-20 | **设备**: Ascend910B3 | **CANN**: 9.0
> **验证脚本**: `examples/lightning_indexer/api_tests/test_*.py`（5 个）
> **验证结果汇总**: `examples/lightning_indexer/api_tests/VERIFICATION_RESULTS.md`
> **API 调研笔记**: `examples/lightning_indexer/api_tests/api_research_notes.md`

#### 12.8.1 验证清单

| # | 验证目标 | 结果 | 关键发现 |
|---|---------|------|---------|
| 1 | `T.tile.merge_sort` 4-way merge | ✅ PASS | 4-way 通过位置参数调用, 切片作为 src 可行 |
| 2 | sort 缓存优化 1D 平铺方案 | ✅ PASS | 1D 平铺 + T.copy 搬运 + 切片 src 可行 |
| 3 | 3 层 tiling + L1 行切片 | ⚠️ 部分通过 | **transpose_B=True + L1 行切片不可行** (diff=66.5) |
| 4 | num_stages=3-4 + threads=2 + CV_COMBINE | ✅ PASS | num_stages=3-4 可行, buffer 须在 body 外 |
| 5 | `T.pipe_barrier("v")` 替代 `barrier_all` | ✅ PASS | pipe_barrier("v") 精度与 barrier_all 一致 |

#### 12.8.2 验证 3 详细结果（关键发现: transpose_B + L1 切片不可行）

```
3 层 tiling (L1 切片):        ❌ FAIL (diff=66.5156)
4 层 serial 嵌套 (L1 切片):   ❌ FAIL (diff=66.5156)
独立 buffer (无切片):          ✅ PASS (diff=0.0156)
单层 tiling 基线:              ✅ PASS (diff=0.0156)
```

**结论**: `T.gemm_v0(transpose_B=True)` + L1 行切片 → 精度错误
**v7.1 修正**: 为每个子 tile 分配独立的 (128, 128) L1 buffer, L1 占用 128KB < 512KB ✓

#### 12.8.3 已验证 API 用法

| API | 正确用法 | 错误用法 (v7 原方案) |
|-----|---------|---------------------|
| `T.tile.merge_sort` 4-way | `merge_sort(dst, s0, s1, s2, s3)` 位置参数 | ~~`merge_sort(..., way=4)`~~ 不存在 way 参数 |
| `T.tile.merge_sort` 切片 src | `merge_sort(dst, cache[0:N], cache[N:2N], ...)` ✅ | — |
| `T.tile.sort` | `sort(dst_buffer, src_buffer, num)` dst 须完整 Buffer | ~~`sort(cache[cache_idx], ...)`~~ 不支持切片 dst |
| `T.gemm_v0` + transpose_B | `gemm_v0(A_L1, B_L1, C, transpose_B=True)` 独立 buffer ✅ | ~~`gemm_v0(Q_L1[m:m+128,:], ..., transpose_B=True)`~~ L1 切片精度错误 |
| `T.Pipelined` num_stages | `for k in T.Pipelined(loop, num_stages=3)`, buffer 在 body 外 ✅ | buffer 在 body 内 → ring-buffer 放大 UB |
| `T.Kernel` threads=2 | `as (cid)` 1 个值 ✅ | ~~`as (cid, _)`~~ threads=2 时解包错误 |
| `T.pipe_barrier` | `pipe_barrier("v")` Vector 管线内部同步 ✅ | 跨管线同步仍需 `barrier_all` |

#### 12.8.4 API 使用注意事项

**Kernel 解包规则**:
| threads | 解包方式 | 示例 |
|---------|---------|------|
| 1（默认） | `as (cid, _)` | `with T.Kernel(blocks, is_npu=True) as (cid, _):` |
| 2 | `as (cid)` | `with T.Kernel(blocks, threads=2, is_npu=True) as (cid):` |

**T.tile.sort vs T.tile.merge_sort 切片支持**:
| API | BufferRegion 切片支持 | 用法 |
|-----|---------------------|------|
| `T.tile.sort` | ❌ 不支持（dst 仅 Buffer） | `T.tile.sort(dst_buffer, src_buffer, actual_num)` |
| `T.tile.merge_sort` | ✅ 支持（src 可切片） | `T.tile.merge_sort(dst, s0, s1, s2, s3)` |

**T.gemm_v0 + transpose_B 可行性**:
| 场景 | 可行性 |
|------|--------|
| 完整 buffer + transpose_B=True | ✅ |
| 独立 buffer + transpose_B=True | ✅ (v7.1 采用) |
| **L1 行切片 + transpose_B=True** | ❌ 精度错误 (验证 3) |

### 12.9 v7.1 修正记录（API 验证后修正）

> **修正时间**: 2026-07-21 | **基于**: API 验证结果（§12.8）+ `VERIFICATION_RESULTS.md`
> **修正原则**: 只修正 API 误用部分, 保留 v7 性能数据和分析

#### 12.9.1 修正清单

| # | 章节 | v7 原方案（错误） | v7.1 修正方案 | 验证依据 |
|---|------|-----------------|-------------|---------|
| 1 | §5.2.1 / §8.5 Cube GEMM tiling | L1 行切片传 `gemm_v0(transpose_B=True)` | **独立 L1 buffer** (A_L1_0/1, B_L1_0/1), 无切片 | 验证 3: 独立 buffer diff=0.0156 ✅ |
| 2 | §8.8 sort 缓存优化 | `T.tile.sort(cache[cache_idx], ...)` 切片 dst | **sort→sort_tmp→T.copy 到 1D cache** | 验证 2: 1D 平铺 PASS ✅ |
| 3 | §8.8 sort 缓存优化 | `merge_sort(..., way=cache_idx)` | **位置参数 + if-elif 分支** (2/3/4-way) | 验证 1: 不存在 way 参数 ✅ |
| 4 | §6.3 流水线 | num_stages=2 (保守, 待 Stage 2 验证 3-4) | **标注 num_stages=3-4 已验证** | 验证 4: PASS ✅ |
| 5 | §8.5 同步 | pipe_barrier("v") 待 Stage 2 验证 | **标注 pipe_barrier("v") 已验证** | 验证 5: PASS ✅ |
| 6 | §5.2.1 / §6 内存规划 | L1 占用 320KB (Q_L1 128KB + K_L1 192KB) | **L1 占用 128KB** (4×32KB 独立 buffer) | v7.1 独立 buffer 核算 |
| 7 | §8.8 UB 规划 | 48KB (3D sorted_cache_ub) | **52KB** (1D sorted_cache + sort_tmp + merge_output) | v7.1 1D 平铺核算 |

#### 12.9.2 修正后的内存占用汇总

| 层级 | v7 原方案 | v7.1 修正 | 变化 |
|------|---------|---------|------|
| L1 (Cube) | 320KB (Q_L1 128KB + K_L1 192KB) | **128KB** (4×32KB 独立 buffer) | -60% |
| UB (Kernel 1 Vector) | 34KB (不变) | 34KB (不变) | — |
| UB (Kernel 2 sort 缓存) | 48KB (3D cache) | **52KB** (1D cache + sort_tmp) | +4KB (sort_tmp) |
| UB (Kernel 2 总计) | ~96KB | ~100KB | +4KB |

#### 12.9.3 相对 v7 的关键调整

1. **Cube GEMM tiling**: 从 "L1 大 buffer + 行切片" 改为 "独立小 buffer + 无切片"
   - 原因: `transpose_B=True` + L1 行切片导致精度错误（验证 3, diff=66.5）
   - 收益: L1 占用 320KB → 128KB（-60%）, 精度正确（diff=0.0156）
   - 代价: GM→L1 加载次数增加（每 L1 tile 4 次独立加载 vs 2 次大 buffer 加载）

2. **sort 缓存优化**: 从 "3D cache + sort 切片 dst + way 参数" 改为 "1D 平铺 cache + sort_tmp 中转 + 位置参数 + if-elif"
   - 原因: `T.tile.sort` 不支持切片 dst（验证 2）, `merge_sort` 不存在 way 参数（验证 1）
   - 收益: API 用法正确, 编译可通过
   - 代价: UB +4KB（sort_tmp 临时 buffer）

3. **流水线/同步**: 从 "待 Stage 2 验证" 改为 "已验证可行"
   - num_stages=3-4: 验证 4 PASS, 精度正确
   - pipe_barrier("v"): 验证 5 PASS, 精度与 barrier_all 一致
   - threads=2 解包: `as (cid)` 1 个值

#### 12.9.4 为何不会再犯同一错误

- **transpose_B + L1 切片**: v7.1 验证脚本 `test_3layer_tiling.py` 永久保留在 `api_tests/` 目录, 后续设计可直接引用验证结论
- **sort 切片 dst**: v7.1 在伪代码注释中明确标注 "T.tile.sort 不支持切片 dst, 必须用完整 Buffer"
- **way 参数**: v7.1 在伪代码注释中明确标注 "不存在 way 参数, 位置参数指定路数"
- **buffer 在 body 内**: v7.1 在 §6.3 明确标注 "buffer 必须在 T.Pipelined body 外部分配"（验证 4 确认）

### 12.10 缺口验证结果（v7.2 新增）

> **验证时间**: 2026-07-21 | **设备**: Ascend910B3 | **CANN**: 9.0
> **验证结果汇总**: `examples/lightning_indexer/api_tests/GAP_VERIFICATION_RESULTS.md`
> **验证脚本**: `examples/lightning_indexer/api_tests/test_memory_budget.py`, `test_sort_stability.py`, `test_3layer_tiling.py`
> **修正原则**: 基于 5 个缺口验证结果修正 v7.1, 保留 v7.1 的 API 验证修正

#### 12.10.1 缺口验证清单

| # | 缺口 | 优先级 | 状态 | 关键发现 | 修正章节 |
|---|------|--------|------|---------|---------|
| 1 | Expert 模式组合验证 | P1 | ✅ 完成 | T.mma+独立 buffer 可行; L0C→UB 不能直搬; T.Scope("V") 用 as (cid, vid) | §8.5, §10.1, §10.4 |
| 2 | 实际数据规模下的内存验证 | P0 | ✅ 完成 | v7.1 UB 超限 (216.4KB), 方案 C (VID_S1=1) 可行 (184.4KB) | §5.3, §8.8, §10.1 |
| 3 | sort 稳定性问题的验证方案 | P1 | ✅ 完成 | sort 稳定性不是问题, 真正问题是 score 计算精度 | §9.3, §10.1, §10.2 Q4 |
| 4 | 增量实现策略 | P0 | ✅ 完成 | 4 阶段增量路线 | §12.10.3 |
| 6 | AscendC 关键优化的代码级对照 | P2 | ✅ 完成 | 6 个遗漏点, 最关键是 SparseTopK 截断 | §8.8, §10.1, §10.4, §12.6.6 |

#### 12.10.2 v7.2 修正清单

| # | 章节 | v7.1 现状 | v7.2 修正方案 | 验证依据 |
|---|------|---------|-------------|---------|
| 1 | §8.5 Cube 数据路径 | 未明确 L0C→UB 禁止 | 标注 L0C→GM→UB 路径 (L0C 不能直接到 UB) | 缺口 1: 硬件内存层级约束 |
| 2 | §8.5 Expert 模式解包 | 未标注 T.Scope("V") 要求 | 标注 `as (cid, vid)` 不能用 `as (cid, _)` | 缺口 1: codegen 报 `Find undefined Variable _` |
| 3 | §5.3 / §8.8 sort 缓存内存 | v7.1 UB=216.4KB > 192KB ❌ | 方案 C: VID_S1=1 + buffer 复用, UB=184.4KB ✓ | 缺口 2: `test_memory_budget.py` |
| 4 | §8.8 SparseTopK 截断 | merge_sort 可能不截断 | merge_sort 后 `T.copy(merge_output[0:BASE_TOPK*2], global_topk_ub)` | 缺口 6: AscendC `DataCopy(dst, tmp, topk*2)` |
| 5 | §9.3 / §10.2 Q4 sort 稳定性 | 误认为 sort 稳定性是问题 | 修正: sort 稳定性是正常行为, check_result 已处理; 真正问题是 score 计算精度 | 缺口 3: `test_sort_stability.py` + `result_compare_method.py:174-221` |
| 6 | §10.1 / §10.4 风险点 | 缺 L0C→UB / SparseTopK / sort 稳定性风险 | 新增 4 个风险项 + 4 个常见错误 | 缺口 1+2+3+6 |
| 7 | §12.6.6 性能优化方向 | 缺 MTE2∥V / MergeSort 3072 等 | 新增 5 个 Stage 3 候选优化 | 缺口 6: AscendC 6 个遗漏点 |

#### 12.10.3 增量实现策略（4 阶段, 缺口 4）

> **策略**: 分 4 个 Phase 渐进式实现, 每个 Phase 都有明确的验证标准和回退点。避免一次性引入所有优化导致精度问题难以定位。

| Phase | 内容 | 验证标准 | 回退点 |
|-------|------|---------|--------|
| **Phase 1** | 只修正 API 误用 (独立 buffer + sort 1D 平铺), 不改变性能策略 | 精度不退化 (diff ≤ 0.0312, 与 v7.1 验证一致) | 回退到 v7.1 (已验证 PASS) |
| **Phase 2** | 加 pipe_barrier("V") 优化 (v7.1 已验证 PASS) | 精度与 Phase 1 一致 + 性能不退化 | 回退到 Phase 1 |
| **Phase 3** | 加 num_stages=3 (v7.1 已验证 PASS) | 精度与 Phase 2 一致 + 性能提升 | 回退到 Phase 2 (num_stages=2) |
| **Phase 4** | 加 sort 缓存优化 (方案 C: VID_S1=1 + SparseTopK 截断) | 精度与 Phase 3 一致 + S1≤4 场景性能提升 | 回退到 Phase 3 (无 sort 缓存) |

**Phase 4 的关键验证项**:
- VID_S1=1 不影响精度 (topk_a_ub 缩小, 但循环多 1 轮)
- SparseTopK 截断正确 (merge_output[0:BASE_TOPK*2] 复制到 global_topk_ub)
- sort 缓存路径与标准路径精度一致 (缺口 3 已验证: 两条路径差异完全相同)
- UB 占用 184.4KB < 192KB ✓ (缺口 2 验证)

#### 12.10.4 缺口 1 详细结果: Expert 模式组合验证

**3 个关键发现**:

1. **L0C 不能直接到 UB** (硬件内存层级约束):
   - ❌ 错误: `T.copy(C_L0, c_ub)` — L0C→UB 直接跨级, 硬件不支持
   - ✅ 正确: `L0C(float32) → GM workspace(cast to float16, enable_relu) → UB → Vector 处理 → GM output`
   - 依据: L0C 是 Cube 专用 accumulator, Vector 核不能直接访问
   - 参考: `examples/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa_no_cv_pipeline.py`

2. **T.Scope("V") 用 as (cid, vid)**:
   - ❌ 错误: `with T.Kernel(blocks, is_npu=True) as (cid, _):` — T.Scope("V") 内 codegen 引用 vid
   - ✅ 正确: `with T.Kernel(blocks, is_npu=True) as (cid, vid):`
   - 依据: codegen 报 `Find undefined Variable _`, `lightning_indexer.py:383` 用 `as (cid, vid)`

3. **Expert T.mma + 独立 L1 buffer 可行** (✅ 测试通过, diff=0.0156):
   - 4 个独立 (128,128) L1 buffer + T.copy(transpose=True) + T.mma 组合正确
   - L1 占用 128KB < 512KB ✓

#### 12.10.5 缺口 2 详细结果: 实际数据规模下的内存验证

**v7.1 sort 缓存优化 UB 超限**:
```
v7.1 原始方案: UB=216.4KB > 192KB ❌ 超限
方案 C (VID_S1=1): UB=184.4KB ✅ 可行
```

**4 种优化方案对比**:
| 方案 | UB 占用 | 可行 | 说明 |
|------|---------|------|------|
| A (复用 sort_tmp+merge_output) | 200.4KB | ❌ | 仍差 8.4KB |
| B (2 块缓存+复用) | 192.4KB | ❌ | 仍差 0.4KB |
| **C (VID_S1=1+4 块缓存+复用)** | **184.4KB** | **✅** | 省 16KB topk_a_ub |
| D (S2_VEC=256+4 块缓存+复用) | 192.4KB | ❌ | 仍差 0.4KB |

**数据范围支撑 (Excel 37 个用例)**:
- S1 ≤ 4: 17 个 (46%) → 缓存路径, VID_S1=1 多 1 轮循环, 影响 <0.5%
- S1 > 4: 20 个 (54%) → 标准路径, 不受影响

#### 12.10.6 缺口 3 详细结果: sort 稳定性

**结论修正**:
- ~~sort 稳定性是问题~~ → **sort 稳定性差异是正常行为, AscendC 也有**
- **check_result 已有正确的对比逻辑** (`result_compare_method.py:174-221`):
  1. 第一步: 排序 index 后比集合 → 集合相同直接通过
  2. 第二步: 集合不同时, 比较差异 index 的 value 与 golden 最小 topk value 的相对误差 (thres=0.0001)
- **不需要在 sort 后做稳定化处理**
- 真正的精度问题在 **value 误差** (score 计算精度), 不是 sort

**精度问题正确方向**:
- 精度失败的根因是 **score 值有微小差异**, 导致 kernel 选了不同的 top-k index
- 差异 index 的 score 不完全相同 (否则 value 误差 = 0, 会通过)
- 需要提高 score 计算精度: GEMM float32 累加 ✓ (已用), Weight mul cast 路径, ReLU 精度

#### 12.10.7 缺口 6 详细结果: AscendC 代码级对照 (6 个遗漏点)

| # | 遗漏点 | 优先级 | v7.2 处理 | 章节 |
|---|--------|--------|----------|------|
| 1 | SparseTopK 显式截断 | 🔴 高 | ✅ 已处理 | §8.8 (merge_sort 后 T.copy 截断) |
| 2 | SortedBasicBlock_ 与 globalTopkUb_ 内存复用 | 🟡 中 | 🟡 方案 E (Stage 3 可选) | §12.6.6 |
| 3 | MergeSort 3072 阈值自适应 | 🟡 中 | 🟡 Stage 3 候选 | §12.6.6 |
| 4 | ifExhaustedSuspension = false | 🟡 中 | 🟡 Stage 3 候选 | §12.6.6 |
| 5 | Vector 侧 MTE2∥V double buffer | 🟡 中 | 🟡 Stage 3 候选 | §12.6.6 |
| 6 | ProcessLD 跨核归约 | 🟢 低 | ✅ 不是问题 | 双 Kernel 架构天然避免 |

#### 12.10.8 v7.2 相对 v7.1 的关键调整

1. **Cube 数据路径约束** (缺口 1):
   - 调整: 明确标注 L0C→GM→UB 数据路径, 禁止 L0C→UB 直搬
   - 原因: 硬件内存层级约束 (L0C 是 Cube 专用, Vector 不能直接访问)
   - 影响: §8.5 伪代码已有 L0C→GM (T.copy enable_relu), v7.2 仅补充说明

2. **Expert 模式解包** (缺口 1):
   - 调整: 标注 T.Scope("V") 用 `as (cid, vid)` 不能用 `as (cid, _)`
   - 原因: codegen 在 T.Scope("V") 内引用 vid
   - 影响: 当前 Developer 模式 threads=2 不受影响 (用 `as (cid)`), 仅 Expert fallback 受影响

3. **sort 缓存方案 C** (缺口 2):
   - 调整: VID_S1 从 2 改为 1, sort_tmp/merge_output 复用 v6 buffer
   - 原因: v7.1 UB=216.4KB > 192KB 超限
   - 收益: UB 降到 184.4KB ✓, 性能影响 <0.5%

4. **SparseTopK 截断** (缺口 6):
   - 调整: merge_sort 后增加 `T.copy(merge_output[0:BASE_TOPK*2], global_topk_ub)` 显式截断
   - 原因: 防止 merge_sort 输出超出 global_topk_ub 分配
   - 收益: 避免越界写入

5. **sort 稳定性修正** (缺口 3):
   - 调整: 删除 "sort 稳定性是问题" 的判断, 改为 "sort 稳定性是正常行为"
   - 原因: check_result 已处理 index 顺序差异, 真正问题是 score 计算精度
   - 收益: 避免不必要的 sort 稳定化处理, 聚焦真正的精度问题

#### 12.10.9 为何不会再犯同一错误

- **L0C→UB 跨级**: v7.2 在 §8.5 伪代码注释中明确标注 "L0C → GM workspace 数据路径 (L0C 不能直接到 UB, 硬件内存层级约束)"
- **T.Scope("V") 解包**: v7.2 在 §8.5 标注 "Expert 模式 T.Scope("V") 解包要求... 不能用 as (cid, _)"
- **sort 缓存 UB 超限**: v7.2 验证脚本 `test_memory_budget.py` 永久保留, 后续设计可直接核算 UB 占用
- **SparseTopK 截断**: v7.2 在 §8.8 伪代码中明确标注 "v7.2 缺口 6: 显式截断到 BASE_TOPK"
- **sort 稳定性误判**: v7.2 在 §9.3 和 §10.2 Q4 明确标注 "sort 稳定性差异是正常行为, 不需要稳定化处理"
- **增量实现**: v7.2 在 §12.10.3 提供 4 阶段增量路线, 每个 Phase 都有回退点, 避免一次性引入所有优化

### 12.11 参考算子设计模式整合（v7.3 新增）

> **新增时间**: 2026-07-21 | **来源**: 5 个参考算子深度分析（`api_tests/REFERENCE_PATTERNS.md`）
> **目的**: 将 5 个参考算子的 12 个关键设计模式整合到 lightning_indexer v7.3 设计中，按 P0/P1/P2 优先级分批落地。
> **保留原则**: v7.3 保留 v7.2 的所有修正（独立 L1 buffer、方案 C sort 缓存、SparseTopK 截断等），仅在此基础上叠加参考算子优化模式。

#### 12.11.1 参考算子清单与核心价值

| # | 参考算子文件 | 核心参考价值 |
|---|-------------|-------------|
| 1 | `examples/sparse_flash_attention/bench_sfa/sparse_flash_attn_pa_no_cv_pipeline.py` | Cube→Vector workspace 中转、MTE2∥V pingpong、PA gather |
| 2 | `examples/xattention/xattention_paged.py` | PA block_table、brcb+row_expand_mul、3-slot PRE_LAUNCH |
| 3 | `examples/xattention/xattention.py` | row_expand_mul 完整用法、reduce_sum 累积、flag 初始化 |
| 4 | `examples/HISA/paged_block_sparse_mqa_attn_expert.py` | 4×L0C、MTE2∥V overlap、tail-fill、wave pipeline |
| 5 | `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py` | num_stages、cross_interval、T.Pipelined 不可用教训、make_zn/make_nz_layout |

#### 12.11.2 12 个设计模式与整合计划

**P0 — 直接性能影响（针对 BSND_BSND Cube 慢 64%）**

| # | 模式 | 参考来源 | 当前 v7.2 状态 | v7.3 整合方案 | 实施章节 |
|---|------|---------|---------------|-------------|---------|
| 1 | **4×L0C 消除 MMA/copy 争用** | HISA 行 156-159 | 单 C_L0（64KB），MMA 与 L0C→GM copy 串行 | 4 个独立 L0C fragment（C_L0_0/1/2/3 各 64KB），每个子 GEMM 独占一个，消除 FIX↔M flag 争用；总 L0C = 4×64KB = 256KB < 512KB ✓ | §4.6, §8.5 |
| 2 | **make_zn_layout / make_nz_layout** | fa_opt 行 105-112 | 未用 annotate_layout | K L1 加 `make_nz_layout`（转置友好，B^T 访问连续），Q L1 加 `make_zn_layout`；通过 `T.annotate_layout` 声明 | §8.5 |
| 3 | **DMA 重排 K[0]+Q 优先** | HISA Wave 0 编排 | K 块顺序加载 | 首个 K 块 + Q 优先 DMA，后续 K 块与 staging/MMA overlap；Wave 0 = K[0]+Q，Wave 1+ = K[i]+ staging | §8.5 |
| 4 | **4-way merge_sort** | AscendC SortedBasicBlock | v7.1 已采用 4-way merge_sort（验证 1 PASS） | ✅ 已整合，保留 v7.2 方案 C 实现 | §8.8 |

**P1 — 性能提升**

| # | 模式 | 参考来源 | 当前 v7.2 状态 | v7.3 整合方案 | 实施章节 |
|---|------|---------|---------------|-------------|---------|
| 5 | **cross_interval=2 减少跨核同步** | fa_opt 行 189, 269, 292 | 每 S2 块都跨核同步（cross_interval=1） | `T.Pipelined(s2BaseNum, num_stages=2, cross_interval=2)`，每 2 个 S2 块同步一次；v7.1 已验证 cross_interval 可用 | §6.3, §8.5 |
| 6 | **MTE2∥V overlap** | HISA 行 357-370 | sort 期间不 enqueue 下一个 S2 块的 DMA | late DMA 入队后不立即等，先做 early output（sort/merge_sort），让 MTE2 与 V 并行；用 set_flag("MTE2","V") + wait_flag 实现 | §8.8 |
| 7 | **brcb_experiment + row_expand_mul_experiment** | xattention 行 891-916 | v6 已用 row_expand_mul_experiment（无 brcb 前缀） | 三步模式：scalar → `brcb_experiment` → lane broadcast → `row_expand_mul_experiment` per chunk；减少 pipe_barrier 数量 | §8.5 |

**P2 — 代码质量/微优化**

| # | 模式 | 参考来源 | 当前 v7.2 状态 | v7.3 整合方案 | 实施章节 |
|---|------|---------|---------------|-------------|---------|
| 8 | **reduce_sum(dim=0) 硬件归约** | HISA 行 343 | v6 已用 reduce_sum(clear=False) merge 语义 | ✅ 已整合；v7.3 补充说明 clear=True 变体（用于单次归约场景，非累加） | §8.5 |
| 9 | **tail-fill 替代 mask** | HISA 行 346-356 | 用 compare + select mask（sparse_mode=3 causal mask） | 标量循环填 -inf，减少指令数；仅适用于尾块边界，causal mask 仍用 compare+select | §8.8 |
| 10 | **signal ID 命名规范** | HISA 行 88-114 | flag ID 较分散（2,3,30,31,20,21,22,40,41） | 用命名常量（SIG_Q_L1=0, SIG_K_L1_0=1, SIG_K_L1_1=2, ...）替代魔法数字 | §7.2, §8.5 |
| 11 | **T.Pipelined 不可用确认** | fa_opt KNOWN BROKEN | ✅ 已用 expert 手动 flag 方式（Developer 模式 T.Pipelined 在 body 外 buffer） | ✅ 已整合，保持 v7.2 策略：UB 密集型 kernel 用 T.serial + 手动 flag，T.Pipelined 仅用于 CV 流水（body 外 buffer） | §6.3 |
| 12 | **3-slot PRE_LAUNCH 流水** | xattention_paged 行 49-50 | pp_slots=2（2-slot 流水） | 🟡 可选：PRE_LAUNCH=2 的 3-slot 流水，Cube 提前 2 步生产；Stage 3 候选优化，v7.3 仅记录不强制 | §6.3（Stage 3 候选） |

#### 12.11.3 整合优先级与实施路线图

**Stage 2 实施（P0 + P1 强制）**：
- **Phase A（P0 基线）**: 模式 1（4×L0C）+ 模式 2（annotate_layout）+ 模式 3（DMA 重排）
  - 验证标准：精度不退化（diff ≤ 0.0312），Cube 时间下降 ≥ 20%
  - 回退点：单 C_L0 + 无 annotate_layout + 顺序 DMA（v7.2 基线）
- **Phase B（P1 性能）**: 模式 5（cross_interval=2）+ 模式 6（MTE2∥V overlap）+ 模式 7（brcb+row_expand_mul）
  - 验证标准：精度不退化 + Vector 利用率提升 ≥ 5%
  - 回退点：Phase A（无 cross_interval overlap + 单 row_expand_mul）

**Stage 3 候选（P2 可选）**：
- 模式 8（reduce_sum clear=True 变体）+ 模式 9（tail-fill）+ 模式 10（signal 命名）+ 模式 12（3-slot PRE_LAUNCH）
- 这些是代码质量/微优化，不影响精度，可在性能达标后按需采纳

#### 12.11.4 v7.3 相对 v7.2 的关键调整

1. **4×L0C 替代单 C_L0**（模式 1）:
   - 调整: `C_L0` 单 fragment → `C_L0_0/1/2/3` 4 个独立 fragment
   - 原因: 消除 MMA 与 L0C→GM copy 的 FIX↔M flag 争用（HISA 验证收益显著）
   - 风险: L0C 占用从 64KB → 256KB（< 512KB ✓）

2. **annotate_layout 显式声明 L1 layout**（模式 2）:
   - 调整: 新增 `T.annotate_layout({A_L1_0: make_zn_layout, B_L1_0: make_nz_layout, ...})`
   - 原因: K L1 用 NZ layout 使 B^T 访问连续，提升 GEMM 效率（fa_opt 验证）
   - 风险: Stage 2 需验证 annotate_layout 与独立 L1 buffer 方案的兼容性

3. **DMA 重排: K[0]+Q 优先**（模式 3）:
   - 调整: 首个 K 块与 Q 一起 DMA，后续 K 块与 staging/MMA overlap
   - 原因: 减少 Wave 0 的 DMA 延迟（HISA Wave pipeline 验证）
   - 风险: 编排复杂度增加，需 Stage 2 验证 flag 时序

4. **cross_interval=2 减少跨核同步**（模式 5）:
   - 调整: `T.Pipelined(s2BaseNum, num_stages=2, cross_interval=2)`
   - 原因: 每 2 个 S2 块同步一次，减少 50% 跨核同步开销（fa_opt 验证）
   - 风险: 无（v7.1 已验证 cross_interval 参数可用）

5. **MTE2∥V overlap**（模式 6）:
   - 调整: sort 期间 enqueue 下一个 S2 块的 GM→UB DMA，用 set_flag/wait_flag 实现
   - 原因: 让 MTE2 与 V 并行，提升 Vector 利用率（HISA 验证收益显著）
   - 风险: 同步复杂度增加，需 Stage 2 验证 flag 时序

6. **brcb_experiment + row_expand_mul_experiment 三步模式**（模式 7）:
   - 调整: 在 row_expand_mul_experiment 前加 brcb_experiment 预处理 scalar → lane broadcast
   - 原因: 减少 pipe_barrier 数量（xattention 验证）
   - 风险: brcb_experiment API 稳定性需 Stage 2 验证

#### 12.11.5 为何不会再犯同一错误

- **4×L0C 容量风险**: v7.3 在 §4.6 显式验证 4×64KB=256KB < 512KB L0C 上限，并在 §10.1 新增风险项追踪
- **annotate_layout 兼容性风险**: v7.3 在 §10.1 标注 "Stage 2 需验证 annotate_layout 与独立 L1 buffer 方案的兼容性"
- **DMA 重排 flag 时序风险**: v7.3 在 §8.5 伪代码注释中明确标注 "Wave 0: K[0]+Q 优先 DMA, Wave 1+: K[i] 与 staging overlap"
- **MTE2∥V 同步复杂度风险**: v7.3 在 §8.8 伪代码注释中明确标注 "late DMA 入队后不立即等, 先做 early output"
- **brcb_experiment API 稳定性风险**: v7.3 在 §10.1 标注 "Stage 2 验证 brcb_experiment 与 row_expand_mul_experiment 的组合精度"
- **增量实现**: v7.3 在 §12.11.3 提供 Phase A/B 分阶段路线，每个 Phase 都有回退点

---

### 12.12 Phase 1-3 验证结果（v7.4 新增）

> **验证时间**: 2026-07-21 | **基于**: `api_tests/PHASE1_3_RESULTS.md` + `api_tests/GAP_VERIFICATION_RESULTS.md` + AscendC arch22 源码对照
>
> v7.3 完成 API 验证、缺口验证、参考算子模式整合后, 进一步执行 Phase 1-3 验证, 定位精度根因、确认单 Kernel 架构、分析边界 Case. 本节记录三项验证结果及 v7.4 修复方案.

#### 12.12.1 Phase 1: 精度根因定位 ✅

**重大发现**: 精度问题的主要根因是**测试代码 bug** + **sort 前 -inf 填充缺失**, 不是 score 计算精度问题.

| 问题 | 根因 | 影响 | 修复 |
|------|------|------|------|
| 7/8 用例精度失败 | 测试代码 `torch.sort(cpu_result/npu_result)` 破坏 check_result 期望的 score 降序 | `value_bm = topk_value[b,n2,s1,cur_cpu[-1]]` 取错值 | ✅ 去掉 torch.sort (1/8→7/8 PASS) |
| li_default_a2 失败 | kernel 输出 -1（无效 index）, PA_BSND top-k 没选够 | 9 行有 -1 | ✅ v7.4 sort 前 -inf 填充 (§8.8 步骤 2.5) |

**关键结论**:
1. **score 计算精度没有问题** — 差异 index 的 score 完全匹配 golden
2. **sort 稳定性不是问题** — check_result 两步对比已处理（缺口 3 已确认）
3. **li_default_a2 的 -1 问题根因**: sort 前没有将无效 K 元素 score 设为 -inf, 导致无效位置被错误选入 top-k

**AscendC 处理方式对照**（`service_vector.h:357-368`, 源码验证）:
```cpp
// 1. sort 前先全填 -inf
Duplicate(sortScoreUb, NEG_INF, cuS2LenVecAlign);
PipeBarrier<PIPE_V>();
// 2. 只覆盖有效元素的 score
Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
PipeBarrier<PIPE_V>();
// 3. 无效 index 填 -1（仅当对齐长度 != 有效长度）
LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.ReinterpretCast<int32_t>();
if (cuS2LenVecAlign != cuS2Len) {
    Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
}
PipeBarrier<PIPE_V>();
// 4. 只覆盖有效元素的 index
Adds(sortIndiceUbInt, globalTopkIndice_, (int32_t)cuBaseS2Idx, cuS2Len);
```

**cu_s2_len 动态计算**（考虑 causal mask）:
```cpp
cuRealAcSeq = actS2Size - (actS1Size - cuS1BeginIdxPerAiv);  // attenMask
cuS2Len = min(cuRealAcSeq - cuBaseS2Idx, s2BaseSize_);
```

**v7.4 修复方案**: sort 前对 `score_buf[cu_s2_len:S2_VEC_BLOCK]` 填 -inf + `index_blk_ub[cu_s2_len:S2_VEC_BLOCK]` 填 -1, 详见 §8.8 步骤 2.5.

#### 12.12.2 Phase 2: 单 Kernel 架构确认 ✅

**确认结果**: `lightning_indexer.py` 已是单 kernel 架构, 参考 `sparse_flash_attn_pa_no_cv_pipeline.py` 模式.

| 对比 | lightning_indexer.py | sparse_flash_attn_pa_no_cv_pipeline.py |
|------|---------------------|----------------------------------------|
| 架构 | 单 kernel | 单 kernel |
| Cube/Vector 分离 | `T.Scope("C")/"V")` 显式 | `AUTO_CV_COMBINE` 自动 |
| 同步 | 手动 `set_flag/wait_flag` | 手动 `set_flag/wait_flag` |
| AUTO_SYNC | False | False（注释掉） |
| MEMORY_PLANNING | True | True |

**架构选择**: 保持当前单 kernel + `T.Scope("C")/"V")` + 手动同步方式. v7.3 文档中的"双 Kernel"描述修正为"单 Kernel + CV 分离"（§2.1 已澄清: "Kernel 1/Kernel 2" 为逻辑 CV 计算划分, 非物理双 kernel）.

#### 12.12.3 Phase 3: 边界 Case 分析 ✅

| 边界 Case | 参数 | 问题 | 严重性 | v7.4 修复方案 |
|----------|------|------|--------|--------------|
| **block_size=16** | B=20,S1=3,S2=512,bs=16,PA | BLOCKS_PER_TILE=16 + TOP_K>S2 | 🔴 超时 | 减小 BLOCK_N (§5.2 边界 Case 1) + -inf 填充 (§8.8) |
| **sparse_count=1** | TOP_K=1 | TOP_K_ALIGNED=64, UB=74.9KB | ✅ 无问题 | — |
| **sparse_count=8192 (Over2K)** | TOP_K=8192 | UB=586KB 严重超限 | 🔴 超限 | Over2K 缩小 S1_BLOCK + virTopK=sparse_count (§5.2 边界 Case 3) |
| **S2=128** | S2 < S2_VEC_BLOCK | TOP_K=2048 > S2=128 | 🟡 | -inf 填充 (§8.8 步骤 2.5) |
| **S1=1** | 最小 S1 | 无特殊问题 | ✅ | — |

**block_size=16 超时根因**:
1. `BLOCKS_PER_TILE = BLOCK_N/block_size = 256/16 = 16`, 每个 tile 需 16 次 PA gather
2. TOP_K=2048 > S2=512, 需从 512 个元素选 2048 个
3. AscendC: `S2_BASE_SIZE=512` 固定, PA gather 在 Vector 侧逐 block 处理

**Over2K UB 超限根因**（AscendC `kernel.h:182-186`, 源码验证）:
```cpp
constInfo.isSparseCountOver2K = (sparseCount <= BASE_TOPK) ? false : true;
constInfo.s1BaseSize = isOver2K ? SPARSE_COUNT_8K / sparseCount * 2 : 8;  // 缩小 S1_BLOCK
// virTopK = isOver2K ? sparseCount : 2048  (隐含在 s1BaseSize 调整中)
```

#### 12.12.4 v7.4 相对 v7.3 的关键调整

| # | 章节 | v7.3 现状 | v7.4 调整 | 来源 |
|---|------|----------|----------|------|
| 1 | §8.8 sort 伪代码 | sort 前未填 -inf | 新增步骤 2.5: sort 前 -inf 填充 + index 填 -1, 对齐 AscendC service_vector.h:357-368 | Phase 1 |
| 2 | §8.8 修正点 | 无 v7.4 条目 | 新增 v7.4 修正点 12-14（-inf 填充 / cu_s2_len 动态计算 / TOP_K>S2 处理） | Phase 1+3 |
| 3 | §5.2 tiling | 无边界 Case 处理 | 新增边界 Case 1/3/4 tiling（block_size<64 减小 BLOCK_N / Over2K 缩小 S1_BLOCK / TOP_K>S2） | Phase 3 |
| 4 | §2.1 / §11 | "双 Kernel" 描述 | 澄清为"单 Kernel + CV 分离"（Kernel 1/2 为逻辑划分） | Phase 2 |
| 5 | §10.1 风险表 | block_size=16 🔴 需排查 | 更新为 ✅ v7.4 已处理 + 新增 4 个 v7.4 风险行 | Phase 1+3 |
| 6 | §10.2 Q27 | 仅排查方向（无具体修复） | 补充 v7.4 具体修复方案（BLOCK_N 缩小 + topk_a_ub 初始化 -inf + cu_s2_len 计算） | Phase 3 |
| 7 | §9 验证方案 | 无 torch.sort bug 说明 | 新增测试代码 torch.sort bug 修正说明 + 边界 Case 测试计划 | Phase 1+3 |

#### 12.12.5 为何不会再犯同一错误

- **sort 前 -inf 填充缺失**: v7.4 在 §8.8 步骤 2.5 显式对齐 AscendC `service_vector.h:357-368` 的四步填充（Duplicate(-inf)→Adds(有效)→Duplicate(-1)→Adds(有效index)）, 并在 §10.1 新增风险项追踪 `cu_s2_len` 动态计算正确性. 不再依赖 mask 单独处理无效位置.
- **测试代码 torch.sort 误用**: v7.4 在 §9 明确标注 check_result 的 `cur_cpu[-1]` 依赖 score 降序, 测试代码不能预先 sort; 同时澄清 `check_result` 内部 `np.sort` 是 index 集合对比所需, 非 bug.
- **block_size<64 超时**: v7.4 在 §5.2 给出具体的 `BLOCK_N=min(BLOCK_N,128)` 修复, 不再停留于"排查方向". 在 §10.2 Q27 给出可执行伪代码.
- **Over2K UB 超限**: v7.4 在 §5.2 对齐 AscendC `kernel.h:183-185` 的 `s1BaseSize = SPARSE_COUNT_8K/sparseCount*2` 公式, 不再沿用固定 S1_BLOCK=8.
- **单 Kernel 架构误解**: v7.4 在 §2.1/§11 明确"单 Kernel + CV 分离", "Kernel 1/2" 为逻辑划分, 避免后续 Stage 2 误建双 kernel.
- **增量验证**: Phase 1-3 每项发现都有源码对照（service_vector.h / kernel.h）和可执行修复方案, 避免基于推测的修改.
