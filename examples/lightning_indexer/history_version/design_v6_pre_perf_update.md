# lightning_indexer 算子设计文档

> **设计版本 v6（基于 v5 架构级优化 + 5 项高效 API 优化）**
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
num_stages = 2  # T.Pipelined 流水深度 (v6: UB=34KB 有余量, Stage 2 尝试 3-4)
# qk_workspace: [NUM_CORES, num_stages, BLOCK_M, BLOCK_N] calc_dtype
# 每 core 独立管理自己的 workspace slot, L2 cache 友好
```

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

> **⚠️ v5 Stage 2 验证项**：
> 1. `T.gemm_v0(Q_L1[m_l0*M_L0:(m_l0+1)*M_L0, :], ...)` 的 2D L1 行切片是否被 T.gemm_v0 支持（examples 中仅有 3D 第一维切片先例 `chunk_gated_delta_rule.py:132`，2D 行切片需验证）
> 2. **fallback**：若 2D 行切片不可用，改为分配 `Q_L1 = T.alloc_shared((M_L0, D), input_dtype)`（L1 只存 M_L0 行），外层循环多次 GM→L1 加载。代价：GM→L1 加载次数增加 BLOCK_M/M_L0 倍，但 L1 占用降至 32KB+32KB=64KB
> 3. `T.copy(C_L0, qk_workspace[..., m_slice, :], enable_relu=True)` 的 L0C→GM 部分写入是否可用（v4 用完整写入，部分写入需验证）

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
- **block_size=768 对齐**: 768 % 16 = 0 ✓（T.gemm_v0 仅要求 16 倍数，无 2 的幂次约束）
- **对齐约束**: UB/L1 32B 对齐: BLOCK_N×sizeof(fp16)=256B ✓, G×D×sizeof(fp16)=4096B ✓
- **L0C 容量** (v5 修正): C_L0=[M_L0=128, N_L0=128]×4=64KB < 128KB ✓ **所有场景安全**（v4 的 [BLOCK_M,BLOCK_N] 全尺寸超限已修正）
- **UB 容量 (Kernel 1)** (v6 修正): row_expand_mul + reduce_sum merge 后峰值 ~34KB < 192KB ✓ **所有 N1 场景安全**（v5 的 68KB 已降为 34KB）
- **UB 容量 (Kernel 2 单次 topk 路径)**: 峰值 ~80KB (MAX_S2=8192) / ~160KB (MAX_S2=16384) < 192KB ✓
- **UB 容量 (分段 topk 路径)**: 峰值 ~96KB < 192KB ✓（与 S2 无关，仅与 SEGMENT_SIZE=8192 有关）

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

> **⚠️ v6 num_stages 建议（优化 #1+#2 后更新）**：
> - **v6 UB=34KB**（v5: 68KB），buffer 在 pipeline body 外不被 ring-buffer
> - num_stages 不受 UB×num_stages 限制，仅受 GM workspace ring-buffer slots 和 pipeline 有效性限制
> - **v6 建议**: num_stages=2（保守基线）→ Stage 2 实测后尝试 **3-4**
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

### 8.5 Cube/Vector 核计算流程（v6: Fixed Core + T.Pipelined + TND 直消 + groupInner=16 + L0 inner split + row_expand_mul + reduce_sum merge）

```python
# v6: 编译器自动将以下代码分割为 AIC (GEMM) 和 AIV (其余) 两部分
# v6: Fixed Core + T.Pipelined + TND 直消 + groupInner=16 + L0 inner split
#     + row_expand_mul_experiment (融合 Brcb+Mul) + reduce_sum(clear=False) (融合 reduce+累加)

NUM_CORES = int(torch.npu.get_device_properties("npu").cube_core_num)  # v6: 动态核数 (A2=24, A3=20)
num_stages = 2  # T.Pipelined 流水深度 (v6: UB=34KB 有余量, Stage 2 尝试 3-4)
groupInner = 16  # v5: Vector 侧 G 维内层分块 (AscendC arch22)
M_L0 = 128       # v5: Cube 侧 L0 M 维块大小 (AscendC arch22)
N_L0 = 128       # v5: Cube 侧 L0 N 维块大小 (AscendC arch22)

launch_core_num = T.min(total_tasks, NUM_CORES)  # v5 缺陷 #5: 小任务不空转

with T.Kernel(launch_core_num, threads=2, is_npu=True) as (cid, vid):
    # --- UB/L1/L0C buffer 在 pipeline body 外分配 (避免 ring-buffer, Q19) ---
    Q_L1 = T.alloc_shared((BLOCK_M, D), input_dtype)       # L1, [512,128]=128KB
    K_L1 = T.alloc_shared((BLOCK_N, D), input_dtype)       # L1, [512,128]=128KB
    C_L0 = T.alloc_fragment((M_L0, N_L0), calc_dtype)      # v5: L0C, [128,128]=64KB (非 [BLOCK_M,BLOCK_N])

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

                    # v4: TND 直接消费 — strided DMA (Q18)
                    if layout_query == "TND":
                        q_offset = t_start_q + s1_tile * s1BaseSize
                        T.copy(Query[q_offset : q_offset + s1BaseSize, :, :], Q_L1)
                    else:  # BSND
                        T.copy(Query[b, s1_tile*s1BaseSize:(s1_tile+1)*s1BaseSize, :, :], Q_L1)

                    # Key 加载 (3 种 layout)
                    if layout_key == "TND":
                        k_offset = t_start_k + s2_block * BLOCK_N
                        T.copy(Key[k_offset : k_offset + BLOCK_N, 0, :], K_L1)
                    elif layout_key == "PA_BSND":
                        k_block_id = block_table[b, s2_block]
                        T.copy(Key[k_block_id, :, 0, :], K_L1)
                    else:  # BSND
                        T.copy(Key[b, s2_start:s2_start + BLOCK_N, 0, :], K_L1)

                    # v5: L0 inner split GEMM (Q21 详述)
                    # Q_L1=[BLOCK_M,D], K_L1=[BLOCK_N,D] 在 L1
                    # C_L0=[M_L0,N_L0]=[128,128] 在 L0C
                    # T.gemm_v0 的 M,N 从 C_L0 shape 推导 (ascend.py:413)
                    for m_l0 in T.serial(T.ceildiv(BLOCK_M, M_L0)):
                        for n_l0 in T.serial(T.ceildiv(BLOCK_N, N_L0)):
                            # T.gemm_v0 + BufferRegion 切片 (Q21)
                            T.gemm_v0(
                                Q_L1[m_l0 * M_L0:(m_l0 + 1) * M_L0, :],  # L1 切片 [M_L0, D]
                                K_L1[n_l0 * N_L0:(n_l0 + 1) * N_L0, :],  # L1 切片 [N_L0, D]
                                C_L0,                                      # [M_L0, N_L0]
                                transpose_B=True,
                                init=(n_l0 == 0)                           # N 维首次 init, 后续累加
                            )
                        # C_L0 完成一整行 [M_L0, BLOCK_N], copy 到 GM workspace (enable_relu)
                        T.copy(C_L0,
                               qk_workspace[cid, s2_block % num_stages,
                                            m_l0 * M_L0:(m_l0 + 1) * M_L0, :],
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
                            T.tile.row_expand_mul_experiment(qk_ub, qk_ub, weight_ub)  # [16,512] = [16,512] * [16]

                            # v6: reduce_sum(clear=False) 融合 reduce+累加 (优化 #2)
                            # 替代 v5: T.reduce_sum(qk_ub, scores_partial, dim=0) + T.tile.add(scores_accum, scores_accum, scores_partial)
                            # clear=False: new_out = old_out + reduced_result (merge 语义)
                            T.reduce_sum(qk_ub, scores_accum, dim=0, clear=False)  # [16,512]→[512], 累加

                            # v6: Vector 管线轻量同步 (优化 #4)
                            # 替代部分 T.barrier_all() — 仅等 Vector 管线, 不等 Cube/MTE2/MTE3
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
| **核数** | 硬编码 24 | 硬编码 24 | **动态 cube_core_num** | v6 适配 A2/A3/950 (优化 #3) |
| **Vector 内同步** | barrier_all | barrier_all (AUTO_SYNC) | **pipe_barrier("v")** (Vector 管线) | v6 轻量同步 (优化 #4) |
| 核间流水 | T.Pipelined(num_stages=2) | 同 (Stage 2 尝试 3-4) | 同 (v6: UB=34KB, Stage 2 尝试 3-5) | 保留 |
| Core 分配 | T.Kernel(NUM_CORES=24) | T.Kernel(min(total_tasks,24)) | T.Kernel(min(total_tasks,动态核数)) | v5 缺陷 #5 + v6 动态核数 |
| TND 处理 | kernel 直接 strided DMA | 同 (act_seq 标量读取标注 Stage 2 验证) | 同 | 保留 + 风险标注 |
| workspace | [NUM_CORES, num_stages, BLOCK_M, BLOCK_N] | 同 (BLOCK_M,BLOCK_N 仍是全尺寸, GM 无容量限制) | 同 | 保留 |

> **⚠️ v6 Stage 2 验证清单**（新增 API 用法）：
> 1. `T.gemm_v0(Q_L1[m_l0*M_L0:(m_l0+1)*M_L0, :], ...)` — 2D L1 行切片（examples 中仅有 3D 第一维切片先例，2D 行切片需验证）
> 2. `T.copy(C_L0, qk_workspace[..., m_slice, :], enable_relu=True)` — L0C→GM 部分写入 + enable_relu
> 3. **v6 新增**: `T.tile.row_expand_mul_experiment(qk_ub, qk_ub, weight_ub)` — dst=src0 原地乘, src1 1D 行广播
> 4. **v6 新增**: `T.reduce_sum(qk_ub, scores_accum, dim=0, clear=False)` — merge 语义, 直接累加到 scores_accum
> 5. **v6 新增**: `T.pipe_barrier("v")` — Vector 管线轻量同步
> 6. **v6 新增**: `torch.npu.get_device_properties("npu").cube_core_num` — host 侧动态核数获取
> 7. `T.tile.fill(scores_accum, 0)` — UB buffer 填充
> 8. `actual_seq_q[b - 1]` — GM tensor 标量读取（缺陷 #6）
> 9. fallback：若 2D 行切片不可用，Q_L1 改为 [M_L0, D]，外层循环多次 GM→L1 加载

### 8.6 Vector 核计算流程（Kernel 2, 纯 Vector）

见 §3.3 Kernel 2 伪代码（v2 保留，v3 仅更新 mask 策略见 §10.2 Q3）。

### 8.7 注意事项

- **AUTO_CV_COMBINE 的 GEMM 识别**：编译器通过 `T.gemm_v0` 调用识别 AIC 代码段。确保 GEMM 的输入是 `alloc_shared`（L1）或 `alloc_fragment`（L0C），输出是 `alloc_fragment`（L0C）
- **L0C → UB 中转**：AscendC 硬件限制 UB 与 L1/L0C 不能直通，必须经 GM workspace。AUTO_CV_COMBINE 自动处理此中转
- **ReLU 融合**：v2 用 `T.copy(C_L0, qk_workspace, enable_relu=True)`。v3 Developer 模式下，若 `enable_relu` 在 L0C→UB 路径不可用，改用 `T.copy(C_L0, qk_ub)` 后 `T.tile.max(qk_ub, qk_ub, 0)` 或 `T.cast` + 比较
- **workspace_idx**: [6] (qk_workspace 在 main 函数签名中的位置)
- **fallback**：若 AUTO_CV_COMBINE 编译失败，回退到 v2 Expert 模式（§3.3 Kernel 1 Expert 代码保留）

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

**int32 indices 特殊对比逻辑**（来自 `result_compare_method.py`）：
- 第一步：排序后集合对比（`set(npu_row) == set(cpu_row)`）
- 第二步（仅集合不同时）：差异元素的 value 相对误差 < thres=0.0001
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
| **L0 inner split API** (v5 新增) | ⚠️ 需 Stage 2 验证 | T.gemm_v0 + 2D L1 行切片无 examples 先例（仅 3D 第一维切片）, fallback: Q_L1 改为 [M_L0, D] |
| **groupInner broadcast** (v5 新增, v6 降级) | ✅ API 已验证 (fallback) | T.tile.broadcast(dst=[16,512], src=[16], axis=1) 参数确认 (ascend_tile.py:2031). **v6: 降级为 fallback, 主路径用 row_expand_mul_experiment** |
| **T.tile.add/fill** (v5 新增, v6 add 降级) | ✅ API 已验证 | T.tile.add (ascend_tile.py:856) — **v6: 降级为 fallback, reduce_sum(clear=False) 替代 add**; T.tile.fill (ascend_tile.py:221) 仍用 |
| **row_expand_mul_experiment** (v6 新增) | ✅ API 已验证 | ascend_tile.py:2353-2372, 使用验证 xattention.py:901-913. ⚠️ Stage 2 验证 dst=src0 原地乘 |
| **reduce_sum(clear=False)** (v6 新增) | ✅ API 已验证 | api-compute.md:119-121 merge 语义. ⚠️ Stage 2 验证 scores_accum 初始化 |
| **pipe_barrier("v")** (v6 新增) | ✅ API 已验证 | xattention.py:898, HISA:320. ⚠️ Stage 2 验证不导致死锁 |
| **动态核数 cube_core_num** (v6 新增) | ✅ API 已验证 | xattention.py:24. ⚠️ Stage 2 验证 A2/A3 跨芯片 |

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

**Q4: TopK 稳定性**
- T.tile.topk 硬件排序**不保证**与 PyTorch stable=True 完全一致（相同 value 时索引顺序可能不同）
- **精度对比方法兼容**：`result_compare_method.py` 使用集合 + value 相对误差, 容忍稳定性差异
- 无需额外处理, 在 DESIGN.md 中明确说明即可

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

- **⚠️ Stage 2 验证项**：
  1. `T.gemm_v0(Q_L1[m_l0*M_L0:(m_l0+1)*M_L0, :], ...)` 的 2D L1 行切片是否被支持
     - examples 中仅有 3D 第一维切片先例（`chunk_gated_delta_rule.py:132` `w_chunk_l1[pid, :, :]`）
     - 2D 行切片 `[m_start:m_end, :]` 需实际验证
  2. **fallback 方案**：若 2D 行切片不可用，改为 `Q_L1 = T.alloc_shared((M_L0, D), input_dtype)`（L1 只存 M_L0 行），外层循环多次 GM→L1 加载
     - 代价：GM→L1 加载次数增加 BLOCK_M/M_L0=4 倍
     - 优势：L1 占用降至 32KB+32KB=64KB（更安全）
     - 此方案与 AscendC 的 L1 级 M_BASIC_BLOCK=256 → L0 级 M_L0=128 三级分块一致
  3. `T.copy(C_L0, qk_workspace[..., m_slice, :], enable_relu=True)` — L0C→GM 部分写入 + enable_relu 需验证

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
| **T.gemm_v0 2D 行切片不支持** (v5 风险) | `T.gemm_v0(Q_L1[m:m+128, :], ...)` 编译失败 | 编译报错 | fallback: Q_L1 改为 [M_L0, D], 外层循环多次 GM→L1 加载 (§8.5 Q21) |
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
├── lightning_indexer.py          # 纯 kernel (双 Kernel: score + topk, @tilelang.jit)
├── test_lightning_indexer.py     # from lightning_indexer import ... + golden + L0 测试 + main
├── proto.yaml                    # 算子接口规格 (dtype/attr)
├── DESIGN.md                     # 本设计文档
├── example_lightning_indexer.py  # 旧版参考实现 (保留, 不作为交付物)
├── example_lightning_indexer_dynamic_shape.py  # 旧版动态 shape 参考
└── history_version/              # 历史备份
    ├── example_lightning_indexer_original.py.bak
    └── example_lightning_indexer_dynamic_shape_original.py.bak
```

### 11.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `DESIGN.md` | ✅ 已完成 | 设计文档 (本文件, v5) |
| `proto.yaml` | ✅ 已完成 | 算子接口规格, 覆盖门禁用 |
| `lightning_indexer.py` | ⬜ 待实现 | 双 Kernel: lightning_indexer_score (Developer + threads=2 + T.Pipelined + groupInner=16 + L0 inner split) + lightning_indexer_topk (Developer) + wrapper (v5: 无 TND 转置) |
| `test_lightning_indexer.py` | ⬜ 待实现 | import kernel + Golden (GeneralizedLI) + L0 用例 + main |

### 11.3 命名规范

- 目录名: `lightning_indexer` (snake_case)
- kernel 文件: `lightning_indexer.py`
- 测试文件: `test_lightning_indexer.py` (顶部 `from lightning_indexer import lightning_indexer`)

### 11.4 实现顺序

1. ✅ 设计文档 (DESIGN.md) + proto.yaml + L0 门槛测试计划 (本文件 §9.2)
2. ⬜ kernel 实现 (`lightning_indexer.py`, 双 Kernel @tilelang.jit + wrapper)
3. ⬜ 测试文件 (`test_lightning_indexer.py`): import kernel + Golden (GeneralizedLI) + L0 用例 + main
4. ⬜ L0 门槛测试通过 (精度收敛, 使用 check_result_li 集合+value 误差对比)
5. ⬜ 扩展分层套件 (L1 功能 / L2 异常 / Boundary 特殊值, 由 tilelang-op-test-design 场景 B 生成)
6. ⬜ 全量套件运行 (L0/L1 须通过; L2/Boundary 失败仅记录不阻塞)

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

### 12.1 4 项架构级优化 + 5 项 v6 API 优化的预期收益

| 优化 | 影响场景 | 预期收益 | 收益来源 | 验证方法 |
|------|---------|---------|---------|---------|
| 1: T.Pipelined 核间流水 | 所有 Kernel 1 场景 | 15-25% | Cube/Vector 重叠隐藏延迟, cross_interval=2 减少同步 | msprof: Cube/Vector pipe utilization 重叠度 |
| 2: Fixed Core 任务分配 | 大 B×S1×S2 场景 | 5-10% | workspace 固定 3MB (v3 6MB+), L2 cache 命中率提升; **v5: min(total_tasks,NUM_CORES) 小任务不空转** | msprof: L2 cache hit rate, workspace 访问延迟 |
| 3: kernel 直接消费 TND | TND 场景 (24% 用例) | 15-30% (TND 场景) | 消除 host wrapper Transpose/Copy kernel | msprof: main_kernel 外无 Transpose kernel |
| 4: broadcast + 整 tile mul | 所有 Kernel 1 场景 | 10-15% | **v5: groupInner=16 分块后, s1×outerG 次循环 (如 32 次) vs v3 的 BLOCK_M=512 次逐行 mul** | msprof: aiv_vec_ratio 提升 (v3 基准 65%) |
| **5: row_expand_mul_experiment** (v6) | 所有 Kernel 1 场景 | **5-10%** | **v6: 融合 Brcb+Mul, 移除 broadcast+mul 两步, 移除 weight_2d(32KB UB)** | msprof: aiv_vec_ratio 提升, UB 占用降低 |
| **6: reduce_sum(clear=False)** (v6) | 所有 Kernel 1 场景 | **3-5%** | **v6: 融合 reduce+累加, 移除 add 步骤, 移除 scores_partial(2KB UB)** | msprof: aiv_vec_ratio 提升 |
| **7: 动态核数 cube_core_num** (v6) | A3/950 等非 A2 芯片 | **适配性** | **v6: 适配 A3(20核)/950, 不再硬编码 24** | 跨芯片测试 |
| **8: pipe_barrier("v")** (v6) | 所有 Kernel 1 场景 | **2-3%** | **v6: Vector 管线轻量同步, 不等 Cube/MTE2/MTE3** | msprof: 同步开销降低 |
| **合计预期** | — | **35-55%** (v5: 30-50% + v6: 10-18%) | — | 对比 v3 实测 vs v6 实测 |

> **⚠️ v6 num_stages 建议（优化 #2 后更新）**：
> - **v6 UB=34KB**（v5: 68KB），buffer 在 pipeline body 外不被 ring-buffer
> - num_stages 不受 UB×num_stages 限制，仅受 GM workspace ring-buffer slots 和 pipeline 有效性限制
> - **v6 建议**: num_stages=2（保守基线）→ Stage 2 实测后尝试 **3-4**
> - num_stages=3 参考 `matmul_add_pipeline.py:46`（已验证可运行）
> - 参考 `flash_attn_optimize.md` 的 "KNOWN BROKEN" 教训：UB buffer 不在 pipeline body 内分配（v6 已确保）

### 12.2 各 msprof 基准场景的预期表现

| Case | AscendC(us) | 80%目标(us) | v3 预估(us) | v4 预期(us) | 达标? | 主要优化 |
|------|------------|------------|------------|------------|-------|---------|
| BSND_BSND (B=16,S1=5,S2=3072,N1=64) | 73.2 | 91.5 | ~110 | ~80-90 | ✅ | 优化1+2+4 (TND 不适用) |
| BSND_PA_BSND (B=2,S1=1,S2=2048,N1=8) | 21.8 | 27.2 | ~35 | ~30-35 | ⚠️ | launch overhead 主导, 优化空间小 |
| TND_TND (B=8,S1=5,S2=3072,N1=24) | 40.3 | 50.4 | ~70 | ~45-55 | ✅ | 优化1+2+3+4 (TND 直消 15-30%) |
| TND_PA_BSND (B=20,S1=3,S2=512,N1=64) | 42.3 | 52.9 | ~65 | ~50-60 | ✅ | 优化1+2+3+4 |

> **BSND_PA_BSND 场景风险**：B=2, S1=1, N1=8 数据量极小，Cube 利用率仅 1.1%，launch overhead 主导。v4 优化对此场景收益有限，可能无法达到 80% 目标。需 Stage 2 实测后评估是否需要 decode 窄块（优化 7）。

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
