# TileLang vs AscendC arch22 实现对比分析

> 生成时间: 2026-07-21
> 对比对象:
> - TileLang: `examples/lightning_indexer/lightning_indexer.py` (基于 li_0720.py 整改后)
> - AscendC: `/home/ops-transformer/attention/lightning_indexer/op_kernel/arch22/`

---

## 一、精度问题

### P1-1. mask 处理方式不同（可能导致 sort 稳定性差异）

**AscendC** (`service_vector.h:356-368`):
```cpp
// sort 前把无效位置 score=-inf, index=-1
Duplicate(sortScoreUb, NEG_INF, cuS2LenVecAlign);
Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
if (cuS2LenVecAlign != cuS2Len) {
    Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);  // ← index 填 -1
}
Adds(sortIndiceUbInt, globalTopkIndice_, cuBaseS2Idx, cuS2Len);
// sort(score, index) — 无效位置 index=-1
```

**TileLang** (`lightning_indexer.py:772-823`):
```python
# mask 把无效 score 设为 -inf，但 index 仍是原始位置
T.tile.compare(mask_blk_ub, index_blk_ub, limit, "LT")
T.tile.select(reduce_g_ub, mask_blk_ub, reduce_g_ub, -inf, ...)
# sort(score, index) — 无效位置 index=0,1,2,...（原始位置）
T.tile.sort(cache_tmp_ub, reduce_g_ub, _BLOCK_N_VEC)
# sort 后才加 s2 偏移
T.tile.axpy(cache_tmp_ub, stride2_blk_ub, s2_start)
```

**差异影响**:
- 当两个 -inf score 排序时，AscendC 的 index=-1，TileLang 的 index=原始位置
- sort 硬件指令对相同 score 的排序可能依赖 index 值做 tie-breaker
- 导致 topk indices 顺序不同 → check_result 的集合匹配失败

**严重度**: 🔴 高（影响 mode=3 和大 S2 场景）

---

### P1-2. index 加偏移时机不同

**AscendC**: sort 前加偏移 `Adds(sortIndiceUbInt, globalTopkIndice_, cuBaseS2Idx, cuS2Len)`
**TileLang**: sort 后加偏移 `T.tile.axpy(cache_tmp_ub, stride2_blk_ub, s2_start)`

**差异影响**:
- AscendC sort 的 (score, real_index) 对
- TileLang sort 的 (score, local_index) 对，sort 后 local_index + s2_start
- 如果 sort 稳定，结果应一致；但 Sort32 硬件指令的稳定性未确认
- 当 score 接近相同时（bf16 精度边界），可能导致不同的排序顺序

**严重度**: 🟡 中（影响边界精度）

---

### P1-3. fp32→bf16→fp32 rounding 时机

**AscendC**: sort 前不做 rounding，直接用 float32 score 排序

**TileLang** (`lightning_indexer.py:178-180`):
```python
# sort 前做 fp32→bf16→fp32 rounding
T.tile.cast(score_bf16_ub, score_accum_ub, "CAST_ROUND", MAX_S2)
T.tile.cast(score_accum_ub, score_bf16_ub, "CAST_ROUND", MAX_S2)
```

**差异影响**:
- golden 做 bf16 rounding（`to_be_sort_ele = reduce_sum.clone().to(torch.bfloat16)`）
- AscendC 不做 rounding，但 check_result 仍通过（允许 5% 差异）
- TileLang 做 rounding 匹配 golden，但可能引入新的精度问题
- 如果 AscendC 的 check_result 允许 5% 差异，说明 AscendC 也有少量不匹配
- TileLang 做 rounding 后理论上应更接近 golden，但实测反而更差

**严重度**: 🟡 中（需要验证 rounding 是否正确）

---

### P1-4. groupInner 维度不同

**AscendC** (`service_vector.h:198`):
```cpp
groupInner_ = 16;  // 硬编码 16
```

**TileLang** (`lightning_indexer.py:235-242`):
```python
if G % 8 == 0:
    VECTOR_BASEG = 8   # ← 比 AscendC 小
```

**差异影响**:
- AscendC: G 维内层 16 个一组，reduce 树形 16→4→2→1
- TileLang: G 维内层 8 个一组，reduce 树形 8→4→2→1
- reduce 的计算顺序不同，浮点累加误差不同
- 但两者最终都是 float32 累加，差异应该很小

**严重度**: 🟢 低（理论上有微小精度差异）

---

### P1-5. reduce 树形算法不同

**AscendC** (`vector.h:130-153`):
```cpp
// DoReduce: 二分法树形 reduce
uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
// 先处理非 2 的幂次余数
// 然后二分: 16→8→4→2→1
```

**TileLang** (`lightning_indexer.py:756-770`):
```python
# 手动树形 reduce
if VECTOR_BASEG == 8:
    T.tile.add(reduce_tmp_ub[0:4], reduce_tmp_ub[0:4], reduce_tmp_ub[4:8])
    T.tile.add(reduce_tmp_ub[0:2], reduce_tmp_ub[0:2], reduce_tmp_ub[2:4])
# 然后 add(reduce_g_ub, [0,:], [1,:])
```

**差异影响**:
- AscendC 用二分法（找到最近的 2 的幂次，先处理余数）
- TileLang 用硬编码的树形（8→4→2→1）
- 对于 groupInner=16 (AscendC) vs VECTOR_BASEG=8 (TileLang)，树形结构不同
- 浮点累加顺序不同 → 微小精度差异

**严重度**: 🟢 低

---

### P1-6. sparse_mode=3 mask 逻辑验证

**AscendC** (`service_vector.h:300-313`):
```cpp
cuRealAcSeq = info.actS2Size;
if (constInfo_.attenMaskFlag) {
    cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
}
// 循环中: cuRealAcSeq += 1
// cuS2Len = min(cuRealAcSeq - cuBaseS2Idx, s2BaseSize_)
```

**TileLang** (`lightning_indexer.py:750-756`):
```python
s2_valid = _k_len_b
causal_limit = _k_len_b - _q_len_b + s1_idx + 1
_is_causal = sparse_mode == 3
s2_valid = T.if_then_else(_is_causal & (causal_limit > 0), causal_limit, s2_valid)
```

**差异分析**:
- AscendC: `cuRealAcSeq = actS2 - (actS1 - cuS1BeginIdx) + innerS1Idx + 1`
  - 展开: `cuRealAcSeq = actS2 - actS1 + cuS1BeginIdx + innerS1Idx + 1`
  - 其中 `cuS1BeginIdx + innerS1Idx = s1_idx`（全局 S1 索引）
  - 所以 `cuRealAcSeq = actS2 - actS1 + s1_idx + 1`
- TileLang: `causal_limit = actK - actQ + s1_idx + 1`

**结论**: 逻辑一致 ✓

但 AscendC 用 `cuS2Len = min(cuRealAcSeq - cuBaseS2Idx, s2BaseSize)` 限制有效长度，TileLang 用 mask 设 -inf。效果应一致，但实现路径不同。

**严重度**: 🟢 低（逻辑一致）

---

## 二、性能问题

### PER-1. 未用 row_expand_mul_experiment（v6 P0 优化未应用）

**AscendC** (`vector.h:93-108`):
```cpp
// Brcb + Mul 两步（与 TileLang v5 方案相同）
Brcb(tmpBuff, weightsUb, ...);  // broadcast
for (int i = 0; i < groupInner; i++) {
    Mul(reduceCacheBuf[i*s2], mmOutUb[i*s2], tmpBuff[i*8], ...);
}
```

**TileLang v6 设计**: `T.tile.row_expand_mul_experiment`（融合 Brcb+Mul 为单条指令）

**当前 TileLang** (`lightning_indexer.py:783-791`):
```python
# 用 broadcast + mul 两步（未用 row_expand_mul_experiment）
T.tile.broadcast(weight_2d_ub, weight_ub)
T.tile.mul(reduce_tmp_ub, mm_res_ub[_gpp, :, :], weight_2d_ub)
```

**差异影响**:
- 当前 TileLang 与 AscendC 用相同的两步方案
- v6 设计的 row_expand_mul_experiment 可以减少 1 条指令 + 32KB UB
- 但 row_expand_mul_experiment 需要 PTO codegen 支持（attempt 1 验证不可用）

**严重度**: 🟡 中（Stage 3 优化点）

---

### PER-2. S2 分块大小不同

**AscendC**:
- `S2_BASIC_BLOCK = 256` (L1 层)
- `S2_BASIC_BLOCK_L0 = 128` (L0 层)
- `s2BaseSize_` 从 tiling 来（通常 512）

**TileLang**:
- `S2_VEC_BLOCK = min(512, ...)` (Vector 层)
- `BLOCK_N = 128/256/512` (Cube 层，自适应)

**差异影响**:
- TileLang 的 S2_VEC_BLOCK=512 比 AscendC 的 s2BaseSize=512 一致
- 但 Cube 侧 BLOCK_N 自适应（PA 时可能到 512），AscendC 固定 256
- 更大的 BLOCK_N 减少 S2 循环次数，但增加 UB 压力

**严重度**: 🟢 低（设计合理）

---

### PER-3. sort 缓存优化缺失

**AscendC** (`service_vector.h:372-407`):
```cpp
if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
    SortAll + MergeSort
} else {
    // actS1Size <= 4: 用 SortedBasicBlock_ 缓存 4 块
    // 缓存满 4 块后精排，减少 merge_sort 次数
    Sort<float, true>(SortedBasicBlock_[...], ...);
    if (globalTopkUbCacheIdx == 3 || isS2End) {
        MrgBasicBlock(...);  // 4-way merge
    }
}
```

**TileLang**: 无缓存优化，每次 S2 块都 sort + merge_sort

**差异影响**:
- 当 S1 <= 4 时，AscendC 缓存 4 块再 merge，减少 75% 的 merge_sort 调用
- TileLang 每次 merge_sort，性能较差
- 对于 S1=1 的推理场景（最常见），AscendC 的缓存优化效果显著

**严重度**: 🔴 高（影响推理性能）

---

### PER-4. sparse_count > 2048 特殊处理缺失

**AscendC** (`service_vector.h:321-323, 371, 408-410`):
```cpp
if (constInfo_.isSparseCountOver2K) {
    // 用 tmpUb_ 作为额外 UB
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}
// ...
if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
    SortAll + MergeSort  // 用不同的 sort 路径
}
```

**TileLang**: 不支持 sparse_count > 2048

**差异影响**:
- AscendC 支持 sparse_count 到 8192（用分段输出）
- TileLang 仅支持 [1, 2048]
- 官方文档也只说 [1, 2048]，所以不影响合规性

**严重度**: 🟢 低（不影响当前需求）

---

### PER-5. LD (Lightning Decode) 机制差异

**AscendC** (`service_vector.h:530-704`):
- 专门的 `ProcessLD()` 阶段
- 用 `vec1ResGm` 和 `vec1ParamGm` 存储中间结果
- 跨核 merge 时用 4-way MrgSort，每满 4 个 list 聚合一次
- 支持 `ifExhaustedSuspension`（排序用尽时暂停）

**TileLang** (`lightning_indexer.py:892-990`):
- Phase 2 cross-core merge
- 用 `TopK_Workspace` 存储中间结果
- 跨核 merge 用 `T.tile.merge_sort`
- 每次只 merge 1 个 list，没有 4-way 批量聚合

**差异影响**:
- AscendC 的 4-way 批量 merge 效率更高
- TileLang 的逐个 merge 效率较低
- 但对于 80% 性能目标，可能影响不大

**严重度**: 🟡 中（Stage 3 优化点）

---

### PER-6. Cube 侧 tiling 差异

**AscendC**:
- `M_BASIC_BLOCK = 256` (L1 层 S1*G)
- `M_BASIC_BLOCK_L0 = 128` (L0 层)
- `S2_BASIC_BLOCK = 256` (L1 层 S2)
- `S2_BASIC_BLOCK_L0 = 128` (L0 层)

**TileLang**:
- `M_L1 = S1_BLOCK * G` (S1_BLOCK=8, G=64 → M_L1=512)
- `BLOCK_M_L0 = 128`
- `BLOCK_N = 128/256/512` (自适应)
- `_N_SPLIT = ceil(BLOCK_N/128)` (L0 内层 N 分割)

**差异影响**:
- AscendC 的 M_BASIC_BLOCK=256 vs TileLang 的 M_L1=S1_BLOCK*G=512
- TileLang 的 M_L1 更大，减少 S1 外层循环
- 但 L0 层都是 128，一致
- N 维: AscendC 固定 256，TileLang 自适应 128/256/512

**严重度**: 🟢 低（设计合理）

---

### PER-7. 同步开销

**AscendC**:
- Cube 侧: 细粒度 set_flag/wait_flag（KEY/QUERY/L0 各自独立）
- Vector 侧: pingpong 双缓冲 set_flag/wait_flag
- Cube→Vector: cross-core event

**TileLang**:
- Cube 侧: 也用细粒度 set_flag/wait_flag
- Vector 侧: 也用 pingpong + set_flag/wait_flag
- Cube→Vector: set_cross_flag/wait_cross_flag

**差异影响**: 同步机制类似，开销应该接近

**严重度**: 🟢 低

---

## 三、架构差异

### ARCH-1. Cube/Vector 协作模式

| 维度 | AscendC | TileLang |
|------|---------|---------|
| Kernel 类型 | KERNEL_TYPE_MIX_AIC_1_2 (AIC:AIV=1:2) | T.Kernel(core_num, is_npu=True) |
| Cube 职责 | QK GEMM + ReLU + Fixp → GM | 相同 |
| Vector 职责 | Weight cast + mul + reduce + sort + merge + output | 相同 |
| 同步 | cross-core event (FIA_SYNC_MODE2) | set_cross_flag/wait_cross_flag |
| GM workspace | mm1ResGm (pingpong 2 slots) | QK_Workspace (pp_slots=2) |

**结论**: 架构一致 ✓

---

### ARCH-2. 内存层级

| 层级 | AscendC | TileLang |
|------|---------|---------|
| Q L1 | 2 bufs × 256×128 | 2 bufs × M_L1_padded×128 |
| K L1 | 3 bufs × 256×128 | 3 bufs × BLOCK_N×128 |
| L0A/L0B | 2 bufs × 128×128 | 2 bufs × 128×128 |
| L0C | 2 bufs × 128×128 | 2 bufs × 128×L0B_N |
| UB | tmpBuf_ (68KB) + sortOutBuf_ (64KB) + ... | w_raw + mm_res + weight + ... (34KB) |

**结论**: 内存层级一致，buffer 数量相同 ✓

---

### ARCH-3. PA block_table 寻址

**AscendC** (`service_cube.h:249-253`):
```cpp
// 逐 token 查 block_table
uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / kCacheBlockSize;
uint64_t keyGmOffset = blkTableGm_.GetValue(bIdx * maxBlockNumPerBatch + s2BlkId) * ...;
```

**TileLang** (`lightning_indexer.py:575-582`):
```python
# 按 block 粒度查 block_table
_block_table_idx = s2_blk * _BLOCKS_PER_TILE + s2_sub * _BLOCKS_PER_TILE + sub
_safe_block_table_idx = T.min(_block_table_idx, max_block_num - 1)
T.copy(Key[BlockTable[b_idx, _safe_block_table_idx], ...], ...)
```

**差异影响**:
- AscendC 逐 token 查（更灵活，支持任意 block_size）
- TileLang 按 block 粒度查（要求 BLOCK_N 是 block_size 的整数倍）
- TileLang 用 min 限制索引，可能访问无效 block 的数据

**严重度**: 🟡 中（边界正确性）

---

## 四、问题优先级汇总

### 精度问题（按严重度排序）

| # | 问题 | 严重度 | 影响用例 | 修复方案 |
|---|------|--------|---------|---------|
| P1-1 | mask 处理方式不同（index=-1 vs 原始位置） | 🔴 高 | mode=3, 大 S2 | sort 前把无效 index 设为 -1 |
| P1-3 | fp32→bf16→fp32 rounding | 🟡 中 | 所有用例 | 验证 rounding 是否应移除（匹配 AscendC） |
| P1-2 | index 加偏移时机不同 | 🟡 中 | 边界精度 | 改为 sort 前加偏移 |
| P1-4 | groupInner 16 vs 8 | 🟢 低 | 微小精度 | 改为 VECTOR_BASEG=16 |
| P1-5 | reduce 树形算法不同 | 🟢 低 | 微小精度 | 匹配 AscendC 的二分法 |
| P1-6 | sparse_mode=3 mask 逻辑 | 🟢 低 | 无 | 逻辑一致，无需修改 |

### 性能问题（按严重度排序）

| # | 问题 | 严重度 | 影响 | 修复方案 |
|---|------|--------|------|---------|
| PER-3 | sort 缓存优化缺失 | 🔴 高 | S1<=4 推理性能 | 实现 SortedBasicBlock 缓存 |
| PER-1 | 未用 row_expand_mul_experiment | 🟡 中 | Vector 指令数 | 等 PTO codegen 支持后启用 |
| PER-5 | LD 4-way merge 缺失 | 🟡 中 | 跨核 merge | 实现 4-way 批量 merge |
| PER-4 | sparse_count > 2048 | 🟢 低 | 功能完整性 | 非当前需求 |
| PER-2 | S2 分块大小 | 🟢 低 | 已优化 | 无需修改 |
| PER-6 | Cube tiling | 🟢 低 | 已优化 | 无需修改 |
| PER-7 | 同步开销 | 🟢 低 | 已优化 | 无需修改 |

---

## 五、修复建议（按优先级）

### 第一优先级：修复 P1-1（mask 处理）

**问题**: sort 前无效位置的 index 不同（-1 vs 原始位置）

**修复方案**: 在 sort 前把无效位置的 index 设为 -1

```python
# 当前:
T.tile.compare(mask_blk_ub, index_blk_ub, limit, "LT")
T.tile.select(reduce_g_ub, mask_blk_ub, reduce_g_ub, -inf, ...)
T.tile.sort(cache_tmp_ub, reduce_g_ub, _BLOCK_N_VEC)

# 修复后（参考 AscendC）:
T.tile.compare(mask_blk_ub, index_blk_ub, limit, "LT")
T.tile.select(reduce_g_ub, mask_blk_ub, reduce_g_ub, -inf, ...)
T.tile.select(index_blk_ub, mask_blk_ub, index_blk_ub, -1, ...)  # ← 新增：无效 index 设为 -1
T.tile.sort(cache_tmp_ub, reduce_g_ub, _BLOCK_N_VEC)
```

### 第二优先级：验证 P1-3（bf16 rounding）

**问题**: li_0720.py 做了 fp32→bf16→fp32 rounding，但 AscendC 不做

**验证方案**: 
1. 先移除 rounding，跑测试看精度变化
2. 如果移除后更差，说明 rounding 是对的
3. 如果移除后更好，说明 rounding 引入了问题

### 第三优先级：修复 P1-2（index 加偏移时机）

**问题**: sort 后加偏移 vs sort 前加偏移

**修复方案**: 改为 sort 前加偏移（匹配 AscendC）

### 第四优先级：实现 PER-3（sort 缓存优化）

**问题**: S1<=4 时每次 merge_sort，AscendC 缓存 4 块再 merge

**修复方案**: 实现 SortedBasicBlock 缓存机制

---

## 六、总结

当前 TileLang 实现与 AscendC arch22 在**架构层面基本一致**（Cube/Vector 协作、内存层级、同步机制），但在 **sort 前的数据准备**（P1-1 mask 处理、P1-2 index 偏移时机）和 **sort 优化**（PER-3 缓存）方面存在差距。

精度问题的核心是 **P1-1（sort 前无效 index 处理）**，这直接影响 sort 的稳定性，是 mode=3 和大 S2 场景精度失败的主要原因。

性能问题的核心是 **PER-3（sort 缓存优化）**，但对于 80% 性能目标可能不是必须的。
