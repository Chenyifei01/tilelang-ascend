# 缺口验证结果汇总

> 验证时间: 2026-07-21 | 设备: Ascend910B3 | CANN 9.0

## 缺口验证清单

| # | 缺口 | 优先级 | 状态 | 关键发现 |
|---|------|--------|------|---------|
| 2 | 实际数据规模下的内存验证 | P0 | ✅ 完成 | 方案 C（VID_S1=1）可行，UB=184.4KB |
| 4 | 增量实现策略 | P0 | ✅ 完成 | 4 阶段增量路线 |
| 1 | Expert 模式组合验证 | P1 | ✅ 完成 | T.mma+独立 buffer 可行；pipe_barrier 需精细同步 |
| 3 | sort 稳定性问题的验证方案 | P1 | ✅ 完成 | 两条路径都不稳定，差异完全相同 |
| 6 | AscendC 关键优化的代码级对照 | P2 | ✅ 完成 | 6 个遗漏点，最关键是 SparseTopK 截断 |

---

## 缺口 2: 实际数据规模下的内存验证 ✅

### 关键发现
- v7.1 sort 缓存优化的原始方案 UB 超限（216.4KB > 192KB）
- **方案 C（VID_S1=1 + buffer 复用）可行**：UB=184.4KB ✅
- VID_S1=1 不影响性能（AscendC 也是 innerS1Idx 串行处理 S1 行）
- sort 缓存优化只在 S1≤4 时启用（46% 用例），S1>4 走标准路径不受影响

### 4 种优化方案对比
| 方案 | UB 占用 | 可行 | 说明 |
|------|---------|------|------|
| A（复用 sort_tmp+merge_output） | 200.4KB | ❌ | 仍差 8.4KB |
| B（2 块缓存+复用） | 192.4KB | ❌ | 仍差 0.4KB |
| **C（VID_S1=1+4 块缓存+复用）** | **184.4KB** | **✅** | 省 16KB topk_a_ub |
| D（S2_VEC=256+4 块缓存+复用） | 192.4KB | ❌ | 仍差 0.4KB |

### 数据范围支撑（Excel 37 个用例）
- S1 ≤ 4: 17 个 (46%) → 缓存路径，VID_S1=1 多 1 轮循环，影响 <0.5%
- S1 > 4: 20 个 (54%) → 标准路径，不受影响

---

## 缺口 4: 增量实现策略 ✅

### 4 阶段增量路线
1. **Phase 1**: 只修正 API 误用（独立 buffer + sort 1D 平铺），不改变性能策略 → 验证精度不退化
2. **Phase 2**: 加 pipe_barrier("V") 优化 → 对比性能
3. **Phase 3**: 加 num_stages=3 → 对比性能
4. **Phase 4**: 加 sort 缓存优化（方案 C） → 对比性能

每个 Phase 都有明确的验证标准和回退点。

---

## 缺口 1: Expert 模式组合验证 ✅

### 关键发现
1. **Expert T.mma + 独立 L1 buffer: ✅ PASS**（diff=0.0156）
   - 4 个独立 (128,128) L1 buffer 可行
   - T.copy(L1, L0, transpose=True) + T.mma 组合正确
   - L1 占用 128KB < 512KB ✅

2. **pipe_barrier("V") + T.Scope("V"): 需要精细同步**
   - `as (cid, vid)` 不是 `as (cid, _)` — T.Scope("V") 内 codegen 引用 vid
   - L0C 不能直接到 UB，必须经 GM 中转：L0C(float32) → GM(cast) → UB → GM
   - Cube→Vector 同步需要 `set_flag/wait_flag`，不是简单 barrier_all
   - **lightning_indexer.py 已有完整实现**（line 633-826），可参考

3. **Cube→Vector 数据路径确认**:
   - 正确: L0C → GM workspace → UB → Vector 处理 → GM output
   - 错误: L0C → UB（直接跨级，硬件不支持）
   - 参考: `sparse_flash_attn_pa_no_cv_pipeline.py` 的模式

### API 使用注意事项
| 场景 | 正确用法 | 错误用法 |
|------|---------|---------|
| T.Scope("V") 内 | `as (cid, vid)` | `as (cid, _)` |
| L0C → UB | L0C → GM → UB | L0C → UB（直接跨级） |
| Cube→Vector 同步 | set_flag/wait_flag | barrier_all（不够） |

---

## 缺口 3: sort 稳定性问题的验证方案 ✅

### 关键发现（已修正）
- ~~sort 稳定性是问题~~ → **sort 稳定性差异是正常行为，AscendC 也有**
- **check_result 已有正确的对比逻辑**：先比 index 集合，集合不同时比 value 误差
- **不需要在 sort 后做稳定化处理**
- 真正的精度问题在 **value 误差**（score 计算精度），不是 sort

### check_result 对比逻辑（result_compare_method.py）
1. **第一步**: 排序 index 后比集合（`np.sort` + 集合相等）→ 集合相同直接通过
2. **第二步**: 集合不同时，比较差异 index 的 value 与 golden 最小 topk value 的相对误差
   - `value_bm = topk_value[b, n2, s1, cpu_topk[-1]]` — golden 最小 topk value
   - `npu_re = abs(npu_value - value_bm) / value_bm` — NPU 差异 index 的相对误差
   - 阈值 `thres = 0.0001`，误差 ≤ 阈值则通过

### 结论修正
| 之前的错误结论 | 正确理解 |
|--------------|---------|
| sort 稳定性是问题 | 正常行为，AscendC 也有相同行为 |
| 需要在 sort 后做稳定化处理 | 不需要，check_result 已处理 |
| 是 sort API 的问题 | 真正问题在 score 计算精度 |

### 精度问题正确方向
精度失败的根因是 **score 值有微小差异**，导致 kernel 选了不同的 top-k index：
- 差异 index 的 score 不完全相同（否则 value 误差 = 0，会通过）
- 需要提高 score 计算精度：GEMM float32 累加、Weight mul cast 路径、ReLU 精度
- **不是 sort 问题，不需要稳定化处理**

---

## 缺口 6: AscendC 关键优化的代码级对照 ✅

### 6 个遗漏点

| # | 遗漏点 | 优先级 | 风险 |
|---|--------|--------|------|
| 1 | SparseTopK 显式截断 | 🔴 高 | merge_sort 输出可能超出 buffer |
| 2 | SortedBasicBlock_ 与 globalTopkUb_ 内存复用 | 🟡 中 | 多耗 16KB UB |
| 3 | MergeSort 3072 阈值自适应 | 🟡 中 | Over2K 场景性能 |
| 4 | ifExhaustedSuspension = false | 🟡 中 | 不等长队列 merge 性能 |
| 5 | Vector 侧 MTE2∥V double buffer | 🟡 中 | Kernel 1 Vector 利用率 |
| 6 | ProcessLD 跨核归约 | 🟢 低 | 架构差异，非遗漏 |

### 最关键遗漏: SparseTopK 截断
- AscendC 用 `DataCopy(dst, tmp, topk * 2)` 显式截断到 BASE_TOPK=2048
- v7.1 用 `merge_sort(global_topk, global_topk, merge_output)` 可能不截断
- **风险**: merge_sort 输出 = 输入大小之和，可能超出 global_topk_ub 的分配

### 建议
1. 验证 `T.tile.merge_sort` 当 dst buffer 小于输出时是否自动截断
2. 若不截断，用 `T.copy(merge_output[0:BASE_TOPK*2], global_topk)` 显式截断
3. 考虑内存复用（合并 topk_a_ub + sorted_cache）
4. 评估 MTE2∥V double buffer 对 Kernel 1 的收益

---

## 对 v7.1 设计的修正建议汇总

| 来源 | 修正项 | 章节 |
|------|--------|------|
| 缺口 1 | Cube→Vector 必须 L0C→GM→UB，不能直接 L0C→UB | §8.5 |
| 缺口 1 | T.Scope("V") 内用 `as (cid, vid)` | §8.5 |
| 缺口 2 | sort 缓存用方案 C（VID_S1=1）或方案 E（buffer alias） | §6, §8.8 |
| 缺口 3 | sort 稳定性需后处理（非 sort 路径问题） | §8.8 |
| 缺口 6 | 补充 SparseTopK 显式截断逻辑 | §8.8 |
| 缺口 6 | 考虑内存复用（topk_a_ub + sorted_cache） | §6 |
