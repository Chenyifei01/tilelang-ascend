# Phase 1-3 验证结果

> 验证时间: 2026-07-21

## Phase 1: 精度根因定位 ✅

### 重大发现
**精度问题的主要根因是测试代码 bug**，不是 score 计算精度问题。

| 问题 | 根因 | 影响 | 修复 |
|------|------|------|------|
| 7/8 用例精度失败 | `torch.sort` 破坏 check_result 期望的 score 降序 | value_bm 取错值 | ✅ 去掉 torch.sort |
| li_default_a2 失败 | kernel 输出 -1，PA_BSND top-k 没选够 | 9 行有 -1 | ⚠️ Stage 2 修复 |

### 修复效果
- 修复前: 1/8 PASS
- 修复后: **7/8 PASS**（仅 li_default_a2 失败）

### 关键结论
1. **score 计算精度没有问题** — 差异 index 的 score 完全匹配 golden
2. **sort 稳定性不是问题** — check_result 两步对比已处理
3. **check_result 逻辑**：先比 index 集合，集合不同时比 value 误差（thres=0.0001）
4. **li_default_a2 的 -1 问题**：PA_BSND 的某些 batch top-k 没选够 2048 个有效 index

---

## Phase 2: 单 Kernel 架构确认 ✅

### 确认结果
**lightning_indexer.py 已是单 kernel 架构**，参考 sparse_flash_attn_pa_no_cv_pipeline.py 模式。

| 对比 | lightning_indexer.py | sparse_flash_attn |
|------|---------------------|-------------------|
| 架构 | 单 kernel | 单 kernel |
| Cube/Vector 分离 | T.Scope("C")/"V") 显式 | AUTO_CV_COMBINE 自动 |
| 同步 | 手动 set_flag/wait_flag | 手动 set_flag/wait_flag |
| AUTO_SYNC | False | False (注释掉) |
| MEMORY_PLANNING | True | True |

### 架构选择
保持当前单 kernel + T.Scope("C")/"V") + 手动同步方式。
v7.3 设计文档中的"双 Kernel"描述需修正为"单 Kernel + CV 分离"。

---

## Phase 3: 边界 Case 分析 ✅

### 边界 Case 分析结果

| 边界 Case | 参数 | 问题 | 严重性 | 修复方案 |
|----------|------|------|--------|---------|
| **block_size=16** | B=20,S1=3,S2=512,bs=16,PA | BLOCKS_PER_TILE=16 + TOP_K > S2 | 🔴 超时 | 减小 BLOCK_N + 处理 TOP_K > S2 |
| **sparse_count=1** | TOP_K=1 | TOP_K_ALIGNED=64, UB=74.9KB | ✅ 无问题 | — |
| **sparse_count=8192** | TOP_K=8192 | UB=586KB 严重超限 | 🔴 超限 | Over2K 标准路径 + 特殊 UB 规划 |
| **S2=128** | S2 < S2_VEC_BLOCK | TOP_K=2048 > S2=128 | 🟡 | 填充 -1（代码已有逻辑） |
| **S1=1** | 最小 S1 | 无特殊问题 | ✅ | — |

### block_size=16 超时根因
1. **BLOCKS_PER_TILE=16**：BLOCK_N=256, block_size=16 → 每个 tile 需 16 次 PA gather
2. **TOP_K=2048 > S2=512**：需要从 512 个元素中选 2048 个，不可能选满
3. 两个问题叠加导致超时

### 修复方案

#### block_size=16 修复
```python
# 当 block_size < 64 时，减小 BLOCK_N 到 128
if is_pa and block_size < 64:
    BLOCK_N = min(BLOCK_N, 128)  # BLOCKS_PER_TILE = 128/16 = 8
```

#### TOP_K > S2 修复
```python
# 当 TOP_K > 有效 S2 时，输出 S2 个有效 index + (TOP_K - S2) 个 -1
# 代码已有 T.tile.select(..., -1.0, "VSEL_TENSOR_SCALAR_MODE") 处理
# 但需要确认 topk_a_ub 初始化为 -inf 确保未选位置输出 -1
```

#### sparse_count=8192 (Over2K) 修复
```python
# isSparseCountOver2K 时走标准路径（不缓存）
# virTopK = sparse_count (最大 8192)
# S1_BLOCK 缩小到 2
# UB 需要特殊规划：topk_a_ub 用 virTopK 而非 BASE_TOPK
```

### UB 超限问题
当前估算 UB=215.4KB > 192KB，但实际代码用 `T.annotate_address` 和 `_UB_LIMIT=196352` 控制。这说明：
1. 我的估算可能不准确（未考虑地址重叠）
2. 实际代码可能通过内存复用控制 UB
3. 需要在 Stage 2 验证实际 UB 占用

---

## 总结

| Phase | 状态 | 关键发现 |
|-------|------|---------|
| Phase 1: 精度根因 | ✅ | 测试代码 bug（torch.sort）+ li_default_a2 PA_BSND -1 bug |
| Phase 2: 架构确认 | ✅ | 单 kernel + T.Scope + 手动同步 |
| Phase 3: 边界 Case | ✅ | block_size=16 超时 + Over2K UB 超限 + TOP_K > S2 处理 |

### 对 Stage 2 的影响
1. **精度修复优先**：去掉 torch.sort（已修复）+ 修复 PA_BSND -1 bug
2. **block_size=16 修复**：减小 BLOCK_N + 确认 TOP_K > S2 处理
3. **Over2K 处理**：走标准路径，特殊 UB 规划
4. **单 kernel 架构**：保持当前架构，不走双 kernel
