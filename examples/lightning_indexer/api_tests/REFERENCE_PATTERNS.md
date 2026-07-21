# 参考算子设计模式总结

> 5 个参考算子深度分析，提取对 lightning_indexer 有价值的设计模式

## 参考算子清单

| # | 算子 | 核心参考价值 |
|---|------|-------------|
| 1 | `sparse_flash_attn_pa_no_cv_pipeline.py` | Cube→Vector workspace 中转、MTE2∥V pingpong、PA gather |
| 2 | `xattention_paged.py` | PA block_table、brcb+row_expand_mul、3-slot PRE_LAUNCH |
| 3 | `xattention.py` | row_expand_mul 完整用法、reduce_sum 累积、flag 初始化 |
| 4 | `HISA/paged_block_sparse_mqa_attn_expert.py` | 4×L0C、MTE2∥V overlap、tail-fill、wave pipeline |
| 5 | `fa_opt/flash_attn_bhsd_*.py` | num_stages、cross_interval、T.Pipelined 不可用教训 |

---

## 关键设计模式

### 1. Cube→Vector 数据传递（5 个算子一致）

```
Cube: L0C → set_flag("m","fix") → wait_flag → T.copy(L0C, GM workspace)
Vector: T.copy(GM workspace, UB) → set_flag("mte2","v") → wait_flag → V 计算
```

**lightning_indexer 现状**: ✅ 已采用（QK_Workspace + SYNC_C1V1/SYNC_V1C1）

### 2. 4×L0C 消除 MMA/copy 争用（HISA）

```python
# 4 个独立 L0C，每个 MMA 独占一个，消除 FIX↔M flag 争用
l0c_0 = T.alloc_L0C([M, N], accum_dtype)
l0c_1 = T.alloc_L0C([M, N], accum_dtype)
l0c_2 = T.alloc_L0C([M, N], accum_dtype)
l0c_3 = T.alloc_L0C([M, N], accum_dtype)
```

**lightning_indexer 现状**: 用 2×L0C，MMA 和 L0C→GM copy 可能争用
**改进**: 若 L0C 容量允许（4×128×128×4=256KB < 512KB），升级到 4×L0C

### 3. MTE2∥V overlap（HISA 核心）

```python
# late DMA 入队后不立即等，先做 early output，让 MTE2 与 V 并行
T.copy(ws_buf[late_ws_slot], s_ub_l)  # late DMA 入队
T.set_flag("MTE2", "V", SIG_S_L)

# early output（此时 late DMA 在 MTE2 队列并行执行）
T.copy(logits_e, Logits[token, ...])

# late compute（等 late DMA 完成）
T.wait_flag("MTE2", "V", SIG_S_L)
T.tile.relu(s_ub_l, s_ub_l)
```

**lightning_indexer 现状**: sort 期间不 enqueue 下一个 S2 块的 DMA
**改进**: sort 期间 enqueue 下一个 S2 块的 GM→UB DMA

### 4. brcb_experiment + row_expand_mul_experiment（xattention）

```python
# 三步模式: scalar → brcb → lane broadcast → row_expand_mul per chunk
T.tile.brcb_experiment(shared_m_broadcast_buf, m_i_prev, REPEAT, 1, 8)
T.pipe_barrier("v")
for o_chunk in T.serial(D // VECTOR_ELEMS):
    T.tile.row_expand_mul_experiment(O_ub[...], O_ub[...], shared_m_broadcast_buf)
T.pipe_barrier("v")
```

**lightning_indexer 现状**: 用 broadcast + mul，pipe_barrier 更多
**改进**: 改用 brcb_experiment + row_expand_mul_experiment

### 5. cross_interval 减少跨核同步（fa_opt）

```python
# 每 cross_interval 个块才发一次跨核信号
if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
    T.set_cross_flag("FIX", SEM_WS1_C2V)
```

**lightning_indexer 现状**: 每 S2 块都跨核同步
**改进**: cross_interval=2，每 2 个 S2 块同步一次

### 6. num_stages 批处理（fa_opt）

```python
num_stages=14  # 一批处理 14 个 KV 块
for k in T.serial(num_outer):
    for i in T.serial(batch_iters):
        # 处理第 k*num_stages + i 个块
```

**lightning_indexer 现状**: 每 S2 块都同步
**改进**: num_stages=4~8，批处理多个 S2 块后一次性同步

### 7. T.Pipelined 不可用教训（fa_opt KNOWN BROKEN）

```
T.Pipelined 让编译器对所有 UB scratch buffer 做 ring-buffering
UB 密集型 kernel 不能用 T.Pipelined，必须用 T.serial + 手动 flag
```

**lightning_indexer 现状**: ✅ 已用 expert 手动 flag 方式

### 8. make_zn_layout / make_nz_layout（fa_opt）

```python
T.annotate_layout({
    q_l1: make_zn_layout(q_l1),    # Q: ZN
    k_l1: make_nz_layout(k_l1),    # K: NZ（转置友好）
    p_l1: make_zn_layout(p_l1),
    v_l1: make_zn_layout(v_l1),
})
```

**lightning_indexer 现状**: 未用 annotate_layout
**改进**: K L1 加 make_nz_layout，Q L1 加 make_zn_layout

### 9. tail-fill 替代 mask（HISA）

```python
# 标量循环填 -inf，替代 compare + select
for i in T.serial(kv):
    if T.cast(i, index_dtype) >= e_limit:
        logits_e[0, i] = -T.infinity(accum_dtype)
```

**lightning_indexer 现状**: 用 compare + select mask
**改进**: 考虑 tail-fill 减少指令数

### 10. reduce_sum(dim=0) 硬件归约（HISA）

```python
T.reduce_sum(s_ub_e, logits_e, dim=0, clear=True)  # 沿 H 维度归约
```

**lightning_indexer 现状**: 用 tile.add 手动树形归约
**改进**: 改用 reduce_sum(dim=0, clear=True)

### 11. DMA 重排（HISA Wave 0）

```
Wave 0: K[0]+Q 优先 DMA → K[1..3] 后续 overlap
```

**lightning_indexer 现状**: K 块顺序加载
**改进**: 首个 K 块 + Q 优先 DMA

### 12. 3-slot PRE_LAUNCH 流水（xattention_paged）

```python
PRE_LAUNCH = 2
TASKQUE_SLOTS = PRE_LAUNCH + 1  # 3
# Cube 提前 2 步生产，Vector 滞后 2 步消费
```

**lightning_indexer 现状**: pp_slots=2
**改进**: 可考虑 PRE_LAUNCH=2 的 3-slot 流水

---

## 对 lightning_indexer 的改进建议（按优先级）

### P0 — 直接性能影响（针对 BSND_BSND 慢 64%）
1. **4×L0C 消除争用**（HISA 行 156-159）
2. **make_nz_layout for K L1**（fa_opt 行 105-112）
3. **DMA 重排: K[0]+Q 优先**（HISA Wave 0）
4. **4-way merge_sort**（AscendC SortedBasicBlock）

### P1 — 性能提升
1. **cross_interval=2 减少同步**（fa_opt）
2. **MTE2∥V overlap**（HISA 行 357-370）
3. **brcb+row_expand_mul 替代 broadcast+mul**（xattention）

### P2 — 代码质量/微优化
1. **reduce_sum(dim=0) 替代手动树形 add**（HISA）
2. **tail-fill 替代 mask**（HISA）
3. **signal ID 命名规范**（HISA）
