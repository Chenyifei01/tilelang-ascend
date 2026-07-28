# TileLang-Ascend UB 内存管理实战：从分配、别名到优化约束

> **TileLang 版本**：`0.1.4+13a1817749f785b9746f942182ba7d9b7f608c46`
> **硬件平台**：Ascend 910B3, CANN 9.0.0
> **参考实现**：AC arch22 LightningIndexer
> **基于算子**：LightningIndexer（sparse attention index selector）

---

## 第 1 章 引言——为什么 UB 是算子优化的关键

Ascend NPU 的内存层级是两条并列通路：Cube 侧 `GM ↔ L1(Cube 缓存) ↔ L0A/L0B ↔ L0C`，Vector 侧 `GM ↔ UB(Vector 缓冲)`——L1 与 UB 都直连 GM，彼此并不串联（L0C 经 FixPipe 回写 GM/L1，UB 与 L1 之间需经 GM 中转）。其中 UB（Unified Buffer）是 Vector pipe 的唯一工作区，所有 V（Vector 计算）、MTE2（GM→UB 搬运）、MTE3（UB→GM 输出）指令的数据都必须经过 UB。UB 的物理大小为 192KB（框架可用常量 196352 字节，已预留 stack headroom），是算子开发中最紧张的资源。

UB 管理对性能有决定性影响。以 LightningIndexer 算子为例，**早期版本**实测到：TileLang 实现的 Vector scalar pipe 等待时间（v_sc_wait）高达 179us，而 AC arch22 参考实现仅 3us——这 176us 的差距曾占总性能 gap 的 70%。根因分析发现，v_sc_wait 的长短直接由 UB buffer 的同步策略决定：buffer 布局决定了需要多少次 flag 同步、能否 overlap MTE2 与 V 计算、能否消除 barrier_all。经同步策略重构（merge loop 改用 V→MTE2 flag、Cube 侧引入 `unit_flag=0b11`，见第 7 章），当前版本 v_sc_wait 已降至 ~23us 级，剩余 Vector gap 主要来自 VB=8 vs 16 的 MTE2 issue 翻倍与 MTE2→V 无 overlap（见 7.4 节与附录 B）。

本文以 LightningIndexer 的优化历程为线索，系统总结 TileLang-Ascend 的 UB 内存管理机制，包括：分配 API、内存规划器的自动别名原理、buffer 生命周期分析方法、手动地址管理、TIR scoping 约束、同步机制交互、以及 UB 作为优化约束的分析方法论。读者应具备基本的 Ascend NPU 知识和 TileLang 开发经验。

---

## 第 2 章 UB 基础——分配 API 与限制

### 2.1 UB 大小

TileLang 框架在内存规划器中定义了 UB 的硬限制：

```python
# 框架常量（伪码）
ASCEND_SHARED_MEM_SIZE = 196352  # UB 可用上限（物理 192KB=196608B，预留 stack 后 196352B），shared.ub scope
ASCEND_SHARED_DYN_MEM_SIZE = 524032  # L1 可用上限（物理 512KB，预留后 524032B），shared.l1 scope
```

910B3 的 UB 物理大小为 192KB（196608 字节），框架预留 stack headroom 后可用常量为 196352 字节（约 191.75KB）。所有 `T.alloc_ub` 分配的 buffer 都在 `shared.ub` scope 内，共享这一空间。

### 2.2 T.alloc_ub API

```python
# TileLang — 声明式：shape × dtype → 字节数
mm_res = T.alloc_ub((2, 8, 512), "float")    # 2×8×512×4 = 32KB
output_ub = T.alloc_ub(2048, "int32")          # 2048×4 = 8KB
output_val = T.alloc_ub(2048, "float16")       # 2048×2 = 4KB
```

`T.alloc_ub` 接受 shape 和 dtype 两个参数，buffer 的字节数 = prod(shape) × sizeof(dtype)。dtype 的选择直接影响 UB 占用：float32 每元素 4 字节，float16/int16 每元素 2 字节。在 UB 紧张时，用 half 代替 float 可以减半占用，但需要额外的 cast 指令。

### 2.3 AC arch22 对比

```cpp
// AC arch22 — 命令式：直接指定字节数
pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
// AC 还使用 TQue 做队列管理
TQue<QuePosition::VECOUT, 1> outQueue_;
pipe->InitBuffer(outQueue_, 1, outNeedBufSize);
```

AC 的 `InitBuffer` 直接传入字节数，而 TileLang 的 `alloc_ub` 传入 shape 和 dtype。AC 还有 `TQue` 机制——一种带硬件队列管理的 buffer，能自动处理 V↔MTE3 同步（通过 EnQue/DeQue），TileLang 目前没有等价物。这个差异对同步开销有重大影响，第 7 章会详细分析。

---

## 第 3 章 内存规划器——自动别名机制深度分析

TileLang 的 UB buffer 地址不是由开发者手动分配的，而是由框架的内存规划 Pass（`tl.AscendMemoryPlanning`）自动分配。理解这个 Pass 的工作原理，是做好 UB 管理的前提。

### 3.1 两种分配模式与默认行为

内存规划器有两种模式，由 PassConfig `tl.ascend_memory_planning` 控制：

```python
# 内存规划器入口逻辑（伪码）
def plan_memory(prim_func):
    liveness_analysis(prim_func)          # 生命周期分析
    for scope, buffers in group_by_scope(buffers):
        if memory_auto_plan:              # 默认 False！
            plan_with_aliasing(scope, buffers)    # 有别名
        else:
            plan_linear(scope, buffers)           # 无别名
```

**关键发现**：`memory_auto_plan` 默认为 `False`。这意味着默认情况下，UB buffer **不做别名**，按声明顺序线性分配地址。需要开发者手动通过 PassConfig 开启自动别名：

```python
# 在 kernel 定义时开启自动别名
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,  # 开启别名
})
def kernel_func():
    ...
```

如果不开启，两个生命周期完全不重叠的 buffer（如 G-reduce 阶段的 mm_res 和 output 阶段的 output_ub）也会被分配到不同地址，造成 UB 浪费。

### 3.2 LivenessAnalysis——生命周期分析

无论哪种模式，规划器首先执行生命周期分析，为每个 buffer 计算 `[start, end]` 活跃区间：

```python
# 生命周期分析（伪码）
def liveness_analysis(stmt_sequence):
    seen = set()
    # 反向遍历：找 KILL 点（每个 buffer 的最后一次使用）
    for stmt in reversed(stmt_sequence):
        for buf in stmt.touched_buffers:
            if buf not in seen:
                seen.add(buf)
                kill_points[stmt].append(buf)  # 在此 stmt 处被 kill
    # 正向遍历：找 GEN 点（每个 buffer 的第一次使用）
    for stmt in stmt_sequence:
        for buf in stmt.touched_buffers:
            if first_use[buf] == stmt.index:
                gen_points[stmt].append(buf)   # 在此 stmt 处被 gen
```

分析结果为每个 buffer 生成一个 `LiveInterval`，包含 `start`（GEN 点序号）、`end`（KILL 点序号）和 `size`（字节数）。两个 interval `[s1, e1]` 和 `[s2, e2]` 如果满足 `e1 < s2 || e2 < s1`，则生命周期不重叠，可以共享地址。

### 3.3 ExtendKillIndex——循环携带变量的特殊处理

简单的 "最后一次使用" 规则在循环场景下不够准确。如果一个 buffer 在循环内被更新，且在循环外被使用，其 KILL 点需要扩展到循环结束：

```python
# 循环携带变量处理（伪码）
def extend_kill_index(buffer, gen_idx, kill_idx):
    for loop in containing_loops(kill_idx):
        if buffer is defined_inside(loop) and used_outside(loop):
            kill_idx = loop.end_index  # 扩展到循环结束
    return kill_idx
```

**实战影响**：LightningIndexer 的 `topk_a_ub` 在 s2_block 循环内通过 `merge_sort` 累积更新，在循环外（BSN end output）被读取。ExtendKillIndex 会把它的活跃区间扩展到覆盖整个循环体（G-reduce + sort + output），导致它看起来在 G-reduce 阶段也 "活跃"——即使 G-reduce 阶段并不读写 topk_a_ub。

这直接导致 `topk_a_ub`（32KB）无法与 `mm_res_ub`（32KB）别名，因为两者的活跃区间在 G-reduce 阶段重叠。VG=16/S1_BLOCK=8 时二者分别升至 64KB 且必须共存，是 G-reduce UB 紧张的直接原因之一（第 8 章详述：不可缓解的根本原因是 tilelang 无 buffer 别名 API）。

### 3.4 LinearScanAllocator——别名算法

当 `memory_auto_plan = True` 时，规划器使用线性扫描分配算法（Linear Scan Register Allocation）为 buffer 分配地址：

```python
# 线性扫描分配（伪码）
def linear_scan_allocate(intervals, memory_limit):
    # 按 start 排序
    intervals.sort(key=lambda x: x.start)
    
    active = []        # 当前活跃的 interval（按 end 排序的优先队列）
    free_blocks = []   # 已释放的可复用地址块
    next_offset = 0    # 下一个新分配的起始地址
    
    for interval in intervals:
        # 1. 释放已过期的 interval，回收地址到 free_blocks
        while active and active[0].end < interval.start:
            expired = active.pop()
            free_blocks.append(AddressBlock(expired.offset, expired.size))
        
        merge_adjacent_blocks(free_blocks)
        
        # 2. 优先在 free_blocks 中找可复用块（first-fit）
        reused_offset = find_reusable_block(interval.size, free_blocks)
        if reused_offset is not None:
            interval.offset = reused_offset
            interval.is_reused = True
        else:
            # 3. 分配新地址
            interval.offset = align_up(next_offset, 32)
            next_offset = interval.offset + interval.size
        
        active.append(interval)
    
    return intervals
```

算法的核心理念是：**当一个 buffer 的生命周期结束后（`end < next.start`），它占用的地址可以被后续 buffer 复用。** 这与寄存器分配中的线性扫描算法一致，时间复杂度为 O(n log n)。

`find_reusable_block` 使用 first-fit 策略——在 free_blocks 中找到第一个能放下的块。这可能导致碎片，但对 UB 这种中小型分配通常足够。

### 3.5 check_overflow——静默溢出问题

在默认的线性分配模式下，有一个令人意外的行为：

```python
# 线性分配（伪码）
def plan_linear(scope, buffers):
    check_overflow = False  # ← 硬编码关闭！
    offset = 0
    for buffer in buffers:
        if offset + buffer.size > memory_limit and check_overflow:
            fatal_error("Out of memory")  # 永远不执行
        address_map[buffer] = offset
        offset = align_up(offset + buffer.size, 32)
```

`check_overflow` 被硬编码为 `False`，意味着 **UB 超限不会在编译期报错**。地址会继续递增，超出 192KB 物理限制。

**实战影响**：
- 当前 tnd_tnd 配置（VG=8/S1_BLOCK=4）buffer 名义总量约 184KB，未超 192KB，不会触发溢出
- 但若切到 VG=16/S1_BLOCK=8 目标配置，名义总量升至约 280KB，超 192KB 达 88KB——此时因 `check_overflow` 关闭，代码仍正常编译，运行时才出现数据损坏、精度下降、aicore crash（507015 错误）
- 实际是否溢出取决于 buffer 生命周期是否真的重叠：开启别名后实际峰值（G-reduce ~236KB）低于名义总量，但仍超限
- 由于默认不开别名且 `check_overflow` 关闭，开发者很难在编译期发现 UB 溢出

**建议**：开发期间手动计算 UB 预算表（第 4 章方法论），不要依赖编译器检查。

### 3.6 32B 对齐

每个 buffer 的起始地址会被 32 字节对齐（`align_up(offset, 32)`）。这是因为 Ascend 的 DataCopyPad 指令要求 32 字节对齐。

对小 buffer 的影响：一个 16 字节的 buffer（如 8 个 half 类型的 weights）实际占用 32 字节，因为下一个 buffer 必须从 32 字节边界开始。LightningIndexer 中 `w_raw_ub` 的 `_w_raw_slot` 取 16（而非 8），让每个 slot = 16 个 half = 32B 恰好对齐，避免 slot 间 padding 浪费；整体 shape `(2, 16)` float16 = 64B。

### 3.7 T.annotate_address——手动地址 pin

当自动规划无法满足需求时，可以通过 `T.annotate_address` 手动指定 buffer 地址：

```python
# 手动地址 pin（伪码）
def annotate_address(address_map):
    # address_map: {buffer_object: physical_address}
    return block_attribute({"address_map": address_map})
```

使用方式：

```python
# 将 save_buf 固定在地址 0x30000
with T.annotate_address({save_buf: 0x30000}):
    T.tile.sort(sort_buf, input_buf, ...)
    T.copy(sort_buf, save_buf)
```

规划器对 `pre_alloc_buffer`（通过 annotate_address 指定的 buffer）跳过自动分配，直接使用指定地址。但如果指定地址与其他活跃 buffer 冲突，规划器会报 FATAL 错误。

典型使用场景见第 5 章。

---

## 第 4 章 Buffer 生命周期分析实战

理解了规划器的工作原理后，开发者需要学会自己分析 buffer 的生命周期，以预测 UB 使用量和别名机会。

### 4.1 三类 buffer 生命周期模式

在 LightningIndexer 中，我们识别出三类典型的生命周期模式：

| 模式 | 例子 | 生命周期特征 | 可别名性 |
|------|------|------------|---------|
| **全程持久** | topk_a_ub | G-reduce + sort + output + Phase 2 全程活跃 | ❌ 几乎不可别名 |
| **阶段性** | mm_res_ub, output_ub | 分别在 G-reduce / output 阶段活跃 | ✅ 可互相别名 |
| **瞬态** | cache_tmp_ub | 仅在 sort 阶段短暂活跃 | ✅ 可别名 |

**全程持久型** buffer 是 UB 优化的最大障碍。它们在整个算子执行期间都需要保留数据，不能与其他 buffer 共享地址。LightningIndexer 的 `topk_a_ub` 就是典型——它在 s2_block 循环中累积 merge 结果，每个 s2_block 的 sort/merge 都要读写它，直到 BSN end 才输出。

### 4.2 关键约束：topk_a_ub 与 mm_res_ub 必须共存

以下是 LightningIndexer 的简化代码流，展示为什么 topk_a_ub 和 mm_res_ub 必须共存：

```python
for s2_block in range(num_s2_blocks):
    # ===== G-reduce 阶段：使用 mm_res_ub =====
    T.copy(QK_Workspace[...], mm_res_ub[gpp, :, :])      # MTE2 加载
    T.tile.mul(reduce_tmp_ub, mm_res_ub[gpp, :, :], weight_2d_ub)  # V 计算
    
    # ===== sort/merge 阶段：读写 topk_a_ub =====
    T.tile.merge_sort(merged_ub, topk_a_ub[s1_local, :], cache_tmp_ub)  # 读 topk_a_ub
    T.copy(merged_ub[0:_TA2], topk_a_ub[s1_local, :])                  # 写 topk_a_ub（更新累积结果）

# ===== BSN end output 阶段：读 topk_a_ub =====
T.copy(topk_a_ub[si, :], p2_acc_ub)  # 读最终累积结果
```

`topk_a_ub` 在每个 s2_block 的 sort/merge 中被读写（累积结果），在 G-reduce 期间必须保留前一个 s2_block 的结果。即使 G-reduce 阶段不直接读写 topk_a_ub，ExtendKillIndex 会把它的活跃区间扩展到覆盖整个循环——因为循环外（BSN end output）还要读它。

**结论**：即使开启自动别名，topk_a_ub（32KB）和 mm_res_ub（32KB）在 G-reduce 阶段也必须同时占用 UB，共 64KB。加上其他必须共存的 buffer，G-reduce 阶段总 UB 使用约 140KB，距 192KB 上限尚有约 52KB 余量。

### 4.3 UB 预算表方法论

要做好 UB 管理，需要建立系统化的预算分析流程：

**步骤 1：列出所有 buffer**

| Buffer | shape | dtype | 字节数 |
|--------|-------|-------|--------|
| topk_a_ub | (2, 4096) | float | 32KB |
| mm_res_ub | (2, 8, 512) | float | 32KB |
| merged_ub | (8192,) | float | 32KB |
| ... | ... | ... | ... |

**步骤 2：分析生命周期，标注共存阶段**

| Buffer | 字节数 | G-reduce 阶段 | sort 阶段 | output 阶段 |
|--------|--------|-------------|----------|------------|
| topk_a_ub | 32KB | ✅ 存活 | ✅ 存活 | ✅ 存活 |
| mm_res_ub | 32KB | ✅ 存活 | ❌ | ❌ |
| output_ub | 8KB | ❌ | ❌ | ✅ 存活 |
| ... | ... | ... | ... | ... |

**步骤 3：计算每个阶段的最大同时使用**

| 阶段 | 必须共存的 buffer | 总计 |
|------|-----------------|------|
| G-reduce | topk_a(32) + mm_res(32) + weight_2d(16) + reduce_tmp(16) + merged(32) + other(12) | **140KB ✅** |
| output | topk_a(32) + p2_acc(16) + output(8) + ... | **76KB ✅** |

**步骤 4：识别瓶颈**

G-reduce 阶段约 140KB，距 192KB 上限有约 52KB 余量。VG=16/S1_BLOCK=8 目标配置会让多个 buffer 翻倍（详见第 8 章），届时才会逼近并超出上限。

这个方法论在第 8 章的 VG=16 不可行性分析中会完整应用。

---

## 第 5 章 手动地址管理实战

当自动规划无法正确处理 buffer 生命周期时（通常是框架内部 temp buffer 的生命周期对规划器不透明），需要用 `T.annotate_address` 手动 pin 地址。

### 5.1 场景：T.tile.sort 内部 temp 与 save buffer 地址重叠

在 LightningIndexer 的 2-slot cache 优化中，我们遇到如下问题：

```python
# 开发者声明的 buffer
save_buf = T.alloc_ub((4096,), "float")

# T.tile.sort 内部会自动分配一个 temp buffer
T.tile.sort(sort_buf, input_buf, ...)
# 规划器看不到 sort 内部 temp 的精确生命周期
# 可能将 temp 和 save_buf 分配到重叠地址

T.copy(sort_buf, save_buf)  # save_buf 可能已被 sort 的 temp 覆盖
```

**症状**：507015 aicore crash（非法内存访问）。

**根因**：`T.tile.sort` 内部分配的 temp buffer 的生命周期对规划器不透明。规划器的 LivenessAnalysis 基于 TIR 层的 buffer touch 追踪，但 sort 的 temp 是在 lowering 阶段生成的，规划器无法准确追踪其活跃区间。结果，temp 和 save_buf 被分配到重叠地址，sort 的 temp 写入损坏了 save_buf 的数据。

### 5.2 解决方案

```python
# 用 T.annotate_address 固定 save_buf 的地址
save_buf = T.alloc_ub((4096,), "float")

with T.annotate_address({save_buf: 0x30000}):
    T.tile.sort(sort_buf, input_buf, ...)
    T.copy(sort_buf, save_buf)
    # save_buf 固定在 0x30000，sort 的 temp 会被分配到其他地址
```

规划器看到 save_buf 已有固定地址（0x30000），会将 sort 的 temp 分配到不冲突的位置。

### 5.3 使用原则

1. **优先依赖自动规划**——只在确认自动规划导致冲突时才用 annotate_address
2. **pin 后必须手动验证**——确保指定地址不与该时刻其他活跃 buffer 冲突
3. **地址需 32B 对齐**——如 0x30000 是 32B 对齐的
4. **记录 pin 原因**——代码注释说明为什么需要手动 pin，便于后续维护

### 5.4 当前主要用途：L1/L0C 地址规划

除上述 sort temp 场景外，`T.annotate_address` 在当前实现中的主要用途是 **L1/L0C 地址规划**——codegen 要求 L1/L0C buffer 必须有显式地址，规划器不会自动分配：

```python
T.annotate_address({
    q_l1: _q_l1_addr,        # L1 buffer 固定地址
    acc_l0c: _acc_l0c_addr,  # L0C buffer 固定地址
    w_raw_ub: _ub_w_raw,     # 顺带 pin UB buffer 到高地址区
    mm_res_ub: _ub_mm_res,   # 让 sort temp 有低地址空间
})
```

这种 pin 同时服务两个目的：(1) 满足 L1/L0C 的显式地址要求；(2) 把 UB buffer 钉到固定地址，为框架内部 temp（如 sort temp）预留不冲突的地址空间。注意 `annotate_address` 是**独占分配**——同一地址不能被两个 buffer 共用（见第 8 章别名缺失），它解决的是"给 buffer 指定地址"，不是"让两 buffer 别名"。

---

## 第 6 章 TIR Scoping 与 UB 可见性

### 6.1 问题

TileLang 的 `@T.prim_func` 中，Python 的 `if/else` 会被翻译为 TIR 的 `IfThenElse` 节点。在 `if/else` 内部 `T.alloc_ub` 分配的 buffer，在 `if/else` 外部不可见：

```python
@T.prim_func
def kernel(...):
    if condition:
        buf = T.alloc_ub((1024,), "float")  # 在 IfThenElse scope 内分配
        T.copy(buf, output)
    # buf 在这里不可见！后续代码无法引用 buf
    T.copy(buf, another_output)  # 编译错误：buf 未定义
```

这是因为 `T.alloc_ub` 生成 `Allocate` IR 节点，嵌套在 `IfThenElse` 的 then-branch 内。Allocate 的作用域限定在其子语句中，外部无法引用。

### 6.2 变通方案

**方案 1：在 if/else 外分配，用索引选择 slot**

```python
buf = T.alloc_ub((2, 1024), "float")  # 2 slots，在 if/else 外分配
idx = T.if_then_else(condition, 0, 1)  # 三元表达式选择 slot
T.copy(buf[idx, :], output)
```

**方案 2：用 Python 编译期条件（非常量条件不适用）**

```python
# 如果 condition 是编译期常量
if COMPILE_TIME_FLAG:
    buf = T.alloc_ub((1024,), "float")
    ...
```

### 6.3 对 ping-pong 的影响

ping-pong（双缓冲）模式通常用 `g_id % 2` 选择 slot：

```python
mm_res = T.alloc_ub((2, 8, 512), "float")  # 2 slots，循环外分配

for g_id in range(num_g_groups):
    gpp = g_id % 2  # slot 索引
    # 用 gpp 索引选择当前 slot
    T.copy(QK_Workspace[...], mm_res[gpp, :, :])
    T.tile.mul(reduce_tmp, mm_res[gpp, :, :], weight_2d)
```

不能写成在 if/else 内分配两个独立 buffer，因为它们在 if/else 外不可见。正确做法是用一个 `(2, ...)` 的 buffer 加索引选择。

---

## 第 7 章 UB 与同步机制的交互

UB buffer 的读写需要同步保护，防止 WAR（Write-After-Read）和 RAW（Read-After-Write）冲突。TileLang 提供两类同步原语：`set_flag`/`wait_flag`（跨 pipe 的 flag 锁）和 `pipe_barrier`（核内 pipe 屏障，可指定单 pipe 或 `PIPE_ALL`；`barrier_all` 即 `pipe_barrier(PIPE_ALL)` 的特例），它们与 UB buffer 的交互方式直接影响性能。

### 7.1 UB buffer 的 WAR 保护

**问题场景**（LightningIndexer merge loop）：

```
迭代 N:   V 读 topk_a_ub（merge_sort 输入）
迭代 N+1: MTE2 写 topk_a_ub（T.copy from TopK_Workspace）
```

如果没有同步，MTE2 可能在 V 还没读完 topk_a_ub 时就覆写，导致 merge_sort 读到半新半旧的数据——表现为精度错误或 aicore crash。

**早期方案**：曾用 `T.barrier_all()` 确保 V 完成后才允许 MTE2 写：

```python
for m in range(num_merge):
    T.barrier_all()  # 确保 V 完成上一迭代的读取
    T.copy(TopK_Workspace[...], topk_a_ub)  # MTE2 写
    T.tile.merge_sort(merged, p2_acc, topk_a_ub)  # V 读
```

**代价**：`barrier_all` 是核内 `pipe_barrier(PIPE_ALL)`，阻塞所有 pipe（V、MTE2、MTE3、Scalar），每次约 3us。在 LightningIndexer 中，merge loop 有 ~5 次迭代 × 3us = ~15us 的 barrier 开销。

**当前方案**：已改用 V→MTE2 flag 做跨迭代 RAW 保护，遵循 7.3 节的 flag 方向性约束——`set_flag` 放在 if-block 外无条件执行（避免死锁），`wait_flag` 放在 if-block 内由 V 评估条件（首迭代可跳过）。这样既消除了 ~15us 的 barrier_all 开销，又规避了 flag 方向性死锁。

### 7.2 set_flag/wait_flag 的方向性

TileLang 的 flag 同步有方向性，不同方向的安全性不同：

| flag 方向 | set 执行者 | wait 执行者 | 能在 if-block 内？ | 原因 |
|-----------|-----------|------------|-------------------|------|
| MTE2→V | MTE2 (DMA) | V (可评估条件) | ✅ 安全 | V 能评估 if 条件 |
| V→MTE2 | V (可评估) | MTE2 (DMA) | ❌ 死锁 | MTE2 不能评估复杂条件 |
| V→MTE3 | V | MTE3 (DMA) | ❌ 死锁 | MTE3 不能评估复杂条件 |
| MTE3→V | MTE3 | V | ✅ 安全 | V 能评估条件 |

**原因**：MTE2 和 MTE3 是 DMA 引擎，没有标量计算能力。当 `wait_flag("V", "MTE2", N)` 被 MTE2 执行时，MTE2 **无条件执行** wait——它不会评估周围的 if 条件。如果 V 在 if-block 内条件性地跳过了 `set_flag`，MTE2 的 wait 就永远等不到 → 死锁。

V pipe 有标量计算能力，可以评估 if 条件，所以 V 执行的 wait_flag 是安全的。

### 7.3 死锁案例

在 LightningIndexer 的 Phase 2 Load Dispatch 优化中，我们尝试用 V→MTE2 flag 替代 barrier_all：

```python
# ❌ 死锁写法
if cross_core_condition:  # 涉及 while 循环、GM 读取的复杂条件
    T.set_flag("V", "MTE2", 7)     # V 条件性 set（只在 cross_core 时 set）

# MTE2 无条件执行 wait_flag（DMA 引擎不评估 if 条件）
T.wait_flag("V", "MTE2", 7)  # 当 cross_core_condition=False 时永远等待 → 死锁
```

```python
# ✅ 正确写法：set_flag 在 if 外（无条件执行）
T.set_flag("V", "MTE2", 7)  # 总是 set，不管条件如何
if cross_core_condition:
    T.copy(TopK_Workspace[...], topk_a_ub)
T.wait_flag("V", "MTE2", 7)  # MTE2 总是能等到 V 的 set
```

**经验法则**：V→MTE2 和 V→MTE3 方向的 `set_flag` 必须放在 if-block 外（无条件执行），对应的 `wait_flag` 可以放在 if-block 内（V 可以评估条件）。

### 7.4 AC SetWaitFlag vs TL set_flag + wait_flag

这是 TileLang 和 AC 在同步机制上的差异（**早期版本**观察），曾被认为是 176us v_sc_wait 差距的根因——但本节末尾"当前状态"表明 AC 同样使用 flag，差距本质是 UB 容量。

```python
# TileLang：两条独立指令，wait_flag 阻塞 scalar pipe
T.set_flag("V", "MTE3", 8)   # scalar 发出 → V pipe 执行 set
T.wait_flag("V", "MTE3", 8)  # scalar 发出后阻塞 → 等 MTE3 响应
# scalar pipe 在 wait 期间不能发出后续指令 → v_sc_wait 累积
```

```cpp
// AC arch22：组合指令，scalar 不阻塞
SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
// set + wait 原子完成，硬件自动同步
// scalar pipe 发出后立即继续，不等 MTE3 响应
```

**性能影响实测**（LightningIndexer tnd_tnd, block 0, vector0）：

| 指标 | AC arch22 | TileLang | Gap |
|------|-----------|----------|-----|
| v_sc_wait | 3.0us | 179.2us | +176.2us |
| v_sc_vec_stall | 114.1us | 17.0us | -97.1us |
| v_scalar | 44.3us | 101.9us | +57.6us |

AC 的 scalar pipe 几乎不等（3us），V pipe stall 较多（114us）——因为 SetWaitFlag 不阻塞 scalar，scalar 快速发出所有指令，V pipe 在指令队列里排队等待数据。

TileLang 的 scalar pipe 大量等待（179us），V pipe 几乎不 stall（17us）——因为 wait_flag 阻塞 scalar，scalar 发一条 wait 就卡住，V pipe 没有指令可执行。

两者总等待时间（sc_wait + vec_stall）AC=117us，TL=196us，差 79us——这就是 v_sc_wait 差距对 aiv_time 的净影响。

**我们尝试过通过减少 wait_flag 数量来降低 v_sc_wait**（batch MTE3 输出、延迟 wait_flag 到 V compute 之后），但实测 v_sc_wait 基本不变。根因是：wait_flag 的阻塞时间由 MTE3/MTE2 的执行时间决定，不是由 wait_flag 指令数决定。即使减少 wait_flag 次数，每次 wait 仍需等 MTE3 完成拷贝。

**当前状态**：上述 179us 数据来自早期版本。经同步策略重构（merge loop 改用 V→MTE2 flag、Phase 1→2 用 `barrier_all`+`sync_all`、Cube 侧 `unit_flag=0b11`）后，当前版本 msprof 实测（tnd_tnd, 20 核均值）Vector 侧 gap 已重构为：

| 指标 | AC arch22 | TileLang（当前） | Gap |
|------|-----------|----------|-----|
| aiv_scalar_wait | 4.54us | 23.21us | +18.67us |
| aiv_scalar_mte2_stall | 0 | 18.62us | +18.62us |
| aiv_vec | 46.72us | 54.06us | +7.34us |
| aiv_time | 68.32us | 84.69us | +16.37us |

剩余 Vector gap 的根因不再是 scalar 阻塞，而是 **VB=8 vs 16 的维度差异**：VB=8 使 MTE2 issue 次数翻倍且与 V 计算无 overlap，导致 mte2_stall 18.62us + C1V1 wait。这指向第 8 章的 UB 别名约束——只有装下 VB=16 才能 overlap MTE2 与 V。注意 AC 同样使用 flag 同步机制，差距的本质是 UB 容量/别名能力，而非 flag API 形式。

### 7.5 跨核同步与 UB 私有性

UB 是核内私有资源，不能跨核访问。跨核数据交换必须通过 GM workspace：

```python
# LightningIndexer 的 TopK_Workspace 模式
# Phase 1：各核将结果写到 GM workspace
T.copy(topk_a_ub, TopK_Workspace[cid, ...])  # UB → GM

T.barrier_all()  # 核内 pipe_barrier(PIPE_ALL)：确保本核 MTE3（UB→GM）写完
T.sync_all()     # 核间同步：等所有核到达，GM 写入对 Phase 2 可见

# Phase 2：各核从 GM workspace 读取其他核的结果
T.copy(TopK_Workspace[other_cid, ...], p2_acc_ub)  # GM → UB
```

AC 用 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 做细粒度跨核同步（只等特定核的特定 flag）。TileLang 的跨核同步靠 `sync_all`（等所有核，粗粒度）；`barrier_all` 是核内 `pipe_barrier(PIPE_ALL)`，只同步本核各 pipe，不等其他核。跨核场景下二者常配合使用——先 `barrier_all` 让本核 GM 写入落盘，再 `sync_all` 等所有核。粗粒度同步更简单但开销更大：核内 `barrier_all` 每次约 3us，跨核 `sync_all` 开销更高，而 AC 的 `CrossCoreWaitFlag` 只需 ~0.5us。

在 LightningIndexer 中，Phase 1 结束时的 `barrier_all + sync_all` 是必要的：`barrier_all`（核内）确保本核 Phase 1 的 GM 写入完成，`sync_all`（核间）确保所有核的写入对 Phase 2 可见。Phase 2 内部的 per-row `barrier_all` 则是冗余的（第 8 章分析），但受限于 flag 方向性死锁问题，无法用 flag sync 替代。

### 7.6 kernel-driven mma unit_flag（Cube 侧配套）

前面几节聚焦 Vector 侧 UB 的同步开销。Cube 侧的 mma→fixpipe 流水同样存在软件 flag 开销，TileLang 通过 `unit_flag` 机制将其卸载到硬件。

**问题**：传统写法中，mma 写 L0C、FixPipe 读 L0C 转换并搬出，需软件 `set_flag`/`wait_flag`（M_FIX/FIX_M 方向）协调 L0C 的 2-slot ping-pong，scalar pipe 发射这些 flag 会产生 `aic_scalar_mte1_stall`。

**方案**：mma 与 L0C→GM 搬运改用硬件 `unit_flag=0b11`：

```python
T.mma(a_l0[side, :, :], b_l0[side, :, :], acc_l0c[side, :, :], init=True, unit_flag=0b11)
T.copy(acc_l0c[side, :, :], GmOut[...], unit_flag=0b11)
```

`unit_flag=0b11` 让硬件自动管理 mma→fixpipe 的 L0C 2-slot ping-pong，无需软件 M_FIX/FIX_M flag（`cmatrixSource=false` 固定、`init=true` 不走累加路径，安全）。L0A/L0B 的 m↔mte1 flag（软件 ping-pong）仍保留，acc_l0c 保持 (2, BLOCK_M_L0, _L0B_N) 2-slot + side 交替。AC arch22 同样使用 `unitFlag=0b11`（service_cube.h）。

**实测**（tnd_tnd, 20 核均值）：Task Duration 94.58→92.84us（**-1.74us**）；cube0 aic_time -1.76、aic_scalar_mte1_stall -1.10、aic_scalar_time -1.82（省 flag 所致）；aic_fixpipe_time +6.26（硬件流水下 fixpipe active 统计含等 mma 释放 L0C，但被 overlap 抵消，net aic_time 仍降）；vector0 aiv_time -2.10 无回归。

**边界**：unit_flag 是 Cube 侧优化，无法闭合 Vector 侧 VB=16 的 UB 阻塞 gap（见第 8 章）——剩余 gap 纯属 UB 别名缺失。

---

## 第 8 章 UB 作为优化约束——VG=16 不可行性分析

UB 不只是"够用就行"的资源约束，它直接决定了哪些算法优化可行、哪些不可行。本章以 VG（Vector G-group chunk size）从 8 提升到 16 的优化尝试为例，展示 UB 预算分析如何判断优化可行性。

### 8.1 VG=16 的收益与代价

AC arch22 使用 `groupInner = 16` + `s1BaseSize = 8`，TileLang 当前使用 `VECTOR_BASEG = 8` + `S1_BLOCK = 4`。对齐 AC 需同时把 VG 提到 16、S1_BLOCK 提到 8（二者共同决定 GEMM 的 M 维与 g-reduce 的 batch）。对于 G=24 的 tnd_tnd 样例：

| 维度 | VG=8/S1_BLOCK=4（当前） | VG=16/S1_BLOCK=8（目标） | delta |
|------|------------|-------------|-------|
| G 迭代次数 | 3（ceil(24/8)） | 2（ceil(24/16)） | -33% |
| vec_time | ~194us | ~167us（对齐 AC） | -27us |
| topk_a_ub | 32KB（2×4096×4，VID_S1=2） | 64KB（4×4096×4，VID_S1=4） | +32KB |
| mm_res_ub | 32KB（2×8×512×4） | 64KB（2×16×512×4） | +32KB |
| weight_2d_ub | 16KB（8×512×4） | 32KB（16×512×4） | +16KB |
| reduce_tmp_ub | 16KB（8×512×4） | 32KB（16×512×4） | +16KB |

VG=16/S1_BLOCK=8 能减少 33% 的 G 迭代次数，预计节省 27us vec_time，且 GEMM M 维从 256 翻到 512 提升 Cube 利用率。但四个 G-reduce 相关 buffer 共增加 96KB。

### 8.2 UB 预算分析

G-reduce 阶段是 UB 使用量的峰值阶段。以下 buffer 必须同时存活（详见第 4 章生命周期分析）：

| Buffer | VG=8/S1_BLOCK=4（当前） | VG=16/S1_BLOCK=8（目标） | 说明 |
|--------|-------------|--------------|------|
| topk_a_ub | 32KB | 64KB | 全程持久，不可缩减 |
| mm_res_ub | 32KB（2-slot） | 64KB（2-slot） | G-reduce 必须 |
| weight_2d_ub | 16KB | 32KB | G-reduce 必须 |
| reduce_tmp_ub | 16KB | 32KB | G-reduce 必须 |
| merged_ub | 32KB | 32KB | sort 紧跟 G-reduce |
| cache_tmp_ub | 4KB | 4KB | sort 必须 |
| 其他小 buffer | 8KB | 8KB | reduce_g, stride2, index_blk 等 |
| **G-reduce 总计** | **140KB** | **236KB** | **+96KB** |
| **UB 限制** | **192KB** | **192KB** | — |
| **余量/超出** | 余 52KB | 超 44KB | ❌ |

VG=8/S1_BLOCK=4 当前配置 G-reduce 仅 140KB，距上限有 52KB 余量——UB 并不紧张。但切到 VG=16/S1_BLOCK=8 对齐 AC 时，四个 G-reduce buffer 翻倍，G-reduce 升至 236KB，超 192KB 达 44KB。

### 8.3 尝试的解锁方案

我们尝试了多种方案来为 VG=16/S1_BLOCK=8 腾出 UB 空间：

| 方案 | 节省 | VG=16 剩余 | 可行？ | 原因 |
|------|------|-----------|--------|------|
| 去掉 weight_2d_ub（用 row_expand_mul 替代） | -32KB | 204KB | ❌ | 仍超 12KB；AscendC 路径必需 tmp，不省 |
| 去掉 merged_ub（in-place merge_sort） | -32KB | 204KB | ❌ | 硬件不支持 dst=src0 |
| 同时去掉 weight_2d + merged | -64KB | 172KB | ✅ | 但 in-place merge 不可行 |
| 减小 topk_a_ub（VID_S1: 4→2） | -32KB | 204KB | ❌ | 翻倍 output 迭代，+60us 开销 |
| 减小 S2_VEC_BLOCK（512→256） | -64KB | 172KB | ❌ | 翻倍 s2_blocks，sort 开销翻倍 |
| 手动 annotate 别名（mm_res↔merged_ub） | -32KB | 204KB | ❌ | 框架拒绝：LinearScan 禁止两 buffer 同地址 |

**结论**：在不修改框架的前提下，VG=16/S1_BLOCK=8 不可行。根本原因不是"差几 KB"的预算问题，而是 **tilelang 缺少 buffer 别名 API**——`T.alloc_ub` 无 free/scope 级 liveness，`LinearScanAllocator` 禁止两个 buffer 共用同一地址（即使生命周期错开），手动 `annotate_address` 同地址会被拒（`memory allocate conflict`）。

### 8.4 UB 优化的通用方法论

从 VG=16 的分析中，可以提炼出 UB 优化的通用方法论：

1. **列出所有 buffer**：shape × dtype → 字节数
2. **分析生命周期**：每个 buffer 的 GEN/KILL 点，标注共存阶段
3. **识别必须共存的 buffer 集合**：特别是全程持久的 buffer
4. **计算每个阶段的最大同时使用**：取所有阶段的最大值
5. **尝试优化**：按以下优先级
   - 自动别名（开启 `tl.ascend_memory_planning`，规划器对生命周期错开的 buffer 自动复用地址）
   - 减 size（减小 shape、用更小的 dtype）
   - 减 slot（2-slot ping-pong 替代 3-slot）
   - 换 GM（将大 buffer 放到 GM workspace，用 MTE2 按需加载）
   - same-buffer 重构（多个 buffer 合并为一个 `alloc_ub`，分阶段用不同 region——可绕过别名 API 缺失，但重构风险高）

关键原则：**全程持久型 buffer 是 UB 优化的最大障碍**。如果能将全程持久 buffer 改为阶段性（如分块处理、GM 中转），能释放大量 UB 空间。

**与 AscendC 的根本差距**：AC arch22 能装下 VB=16/S1_BLOCK=8，靠的是**手动 buffer 别名**——`SortedBasicBlock_` 别名 `globalTopkUb_`、`tmpBuf_` 68KB 双职责（搬运 cube 结果时作 db 双缓冲，mrgsort 时复用为 merge temp），显式时间错开而不依赖规划器。tilelang 无 alias/overlay API（`view` 只是只读视图），`alloc_ub` 无 free/scope 级 liveness，无法复刻这种手动复用——这是 VG=16 不可行、以及与 AC 剩余性能 gap 的根本约束（详见附录 B）。

---

## 第 9 章 常见陷阱与解决方案

以下是 LightningIndexer 优化过程中遇到的 UB 相关陷阱汇总，按出现频率排序。

### 陷阱 1：内存规划器地址重叠

**症状**：507015 aicore crash（非法内存访问）

**根因**：`T.tile.sort` 等 API 内部分配的 temp buffer 生命周期对规划器不透明，规划器将其与用户 buffer 分配到重叠地址。

**解决**：用 `T.annotate_address` 固定用户 buffer 地址，迫使规划器将 temp 分配到其他位置。

### 陷阱 2：UB 静默溢出

**症状**：精度下降、偶发 crash、难以复现的数据损坏

**根因**：`check_overflow = false`（硬编码关闭），UB 超限不报编译错误。buffer 地址超出 192KB 物理范围，访问到其他核的 UB 或未定义内存。

**解决**：开发期间手动计算 UB 预算表（第 4 章方法论），不依赖编译器检查。对每个阶段（G-reduce、sort、output）分别计算最大同时使用。

### 陷阱 3：32B 对齐

**症状**：数据损坏，特定 shape 下出现

**根因**：DataCopyPad 指令要求 32B 对齐。如果 buffer 大小不是 32B 的倍数，后续 buffer 的起始地址虽然会自动 32B 对齐，但当前 buffer 的有效数据可能不足 32B，导致 DataCopyPad 读到 stale 数据。

**解决**：将小 buffer pad 到 32B 的倍数。例如 `w_raw_ub` 从 `(8,)` half = 16B pad 到 `(16,)` half = 32B。

### 陷阱 4：T.copy mixed dtype

**症状**：编译错误 "no matching function for copy_gm_to_ub"

**根因**：ascendc backend 的 `copy_gm_to_ub` 不带 dtype 模板参数，不支持 fp16→float 的混合 dtype 拷贝。只有 PTO backend 的 `copy_gm_to_ub_dynamic<half_t, float>` 支持。

**解决**：分两步——先 `T.copy` 保持原 dtype，再用 `T.tile.cast` 转换 dtype。

### 陷阱 5：TIR scope 可见性

**症状**：编译错误 "undefined variable"

**根因**：在 Python if/else 内 `T.alloc_ub` 分配的 buffer，在 if/else 外不可见（TIR Allocate 嵌套在 IfThenElse scope 内）。

**解决**：在 if/else 外分配 buffer，用索引选择 slot。详见第 6 章。

### 陷阱 6：V→MTE2 flag 死锁

**症状**：kernel 挂起（无输出、无 crash、不退出）

**根因**：V→MTE2 方向的 `set_flag` 在 if-block 内条件执行，但 MTE2 的 `wait_flag` 无条件执行（DMA 引擎不评估 if 条件）。当条件为 false 时，V 不 set，MTE2 永远 wait。

**解决**：V→MTE2 和 V→MTE3 方向的 `set_flag` 必须放在 if-block 外（无条件执行）。详见第 7 章。

### 陷阱 7：跨迭代 WAR

**症状**：aicore crash（VEC illegal configuration）或精度错误

**根因**：循环迭代 N+1 的 MTE2 写 UB buffer，与迭代 N 的 V 读同一 buffer 产生 WAR 冲突。

**解决**（按优先级）：
1. **V→MTE2 flag**（当前方案，见 7.1）：`set_flag` 放 if 外、`wait_flag` 放 if 内，开销远低于 barrier_all
2. **ping-pong buffer**（2-slot 交替）：彻底消除 WAR，但需额外 UB
3. **`barrier_all`**（早期方案，已弃用）：`pipe_barrier(PIPE_ALL)` 阻塞所有 pipe，每次约 3us，仅作兜底

---

## 第 10 章 UB 调试方法论

前述陷阱给出了"是什么、为什么"，本章给出"怎么定位"。UB 问题往往表现为运行时 crash 或精度错误，编译期无报错，需要系统化的定位手段。

### 10.1 症状→定位方法 速查表

| 症状 | 可能原因 | 首选定位方法 |
|------|---------|------------|
| 507015 aicore crash | UB 地址重叠 / UB 溢出 / 越界 | UB 预算表核对（10.5）+ `dump_tensor` 邻近 buffer |
| kernel 挂起（不退出、无 crash） | flag 死锁（V→MTE2 set 在 if 内） | 检查 set_flag/wait_flag 位置（7.3）+ 二分注释 |
| 精度错误（NaN / 数值偏差） | 跨迭代 WAR / mixed dtype / 未初始化 | `dump_tensor` 分段对比 golden（10.2） |
| 编译错误 "undefined variable" | TIR scope 可见性 | if/else 内 alloc_ub（第 6 章） |
| 编译错误 "memory allocate conflict" | annotate_address 同地址 | 别名不可行（第 8 章） |

### 10.2 T.dump_tensor：NPU vs golden 逐步对比

`T.dump_tensor`（`ascend.py:499`）把 UB buffer 内容输出到 GM 供 host 读取对比，是定位精度问题的核心手段。在可疑计算前后插入 dump，逐段与 CPU golden 比对：

```python
T.tile.mul(reduce_tmp_ub, mm_res_ub[_gpp, :, :], weight_2d_ub)
T.dump_tensor(reduce_tmp_ub, 0, VECTOR_BASEG * S2_VEC_BLOCK)  # dump mul 后
T.tile.add(reduce_tmp_ub[0:4, :], reduce_tmp_ub[0:4, :], reduce_tmp_ub[4:8, :])
T.dump_tensor(reduce_tmp_ub, 1, VECTOR_BASEG * S2_VEC_BLOCK)  # dump reduce 后
```

方法：从最小用例开始，分 CopyIn → Compute → CopyOut 三段 dump，定位首个偏差段。注意 dump 会改变时序，可能掩盖竞态——WAR 类问题需结合 msprof（10.4）。定位后应移除 dump 代码（生产 kernel 不带调试代码）。

### 10.3 T.printf：标量诊断

`T.printf`（`ascend.py:451`）打印标量值（循环变量、地址、长度），用于核对 tiling 计算与边界：

```python
T.printf("s2_blk=%d, _gpp=%d, vid=%d\n", s2_blk, _gpp, s1_local)
```

适用于：边界越界（`s1_local` 超出 `VID_S1`）、slot 索引错误、BSN 切分异常。注意 printf 走 scalar pipe，过多会干扰时序，定位后移除。

### 10.4 msprof：pipe 瓶颈定位

msprof 采集的 `PipeUtilization.csv` 是性能/同步问题的主依据（测量方法见附录 D）：

- `aic_scalar_mte1_stall` / `aiv_scalar_mte2_stall` 高 → MTE 与上游 pipe 无 overlap，检查 flag 同步是否过粗
- `aiv_scalar_wait` 高 → wait_flag 阻塞 scalar，检查能否合并/前移 wait
- `aic_fixpipe_time` 异常高 → fixpipe 等待 mma 释放 L0C，检查 L0C 2-slot 是否冲突
- 某 pipe time 近零 → 该 pipe 未被利用，检查数据通路是否断开

### 10.5 UB 预算表核对

对 507015 / 疑似溢出，第一步是手算 UB 预算表（第 4 章方法论）：列出所有 buffer 的 shape×dtype→字节数，按 G-reduce / sort / output 分阶段求和，与 192KB 上限比对。注意名义总量与峰值（别名后）的差别——即便名义超限，若生命周期错开且开了 `tl.ascend_memory_planning`，实际峰值可能不超；反之同名 buffer 在循环级 liveness 重叠则不能别名（见 3.3）。

---

## 第 11 章 总结与最佳实践 Checklist

### 11.1 UB 管理 Checklist

**分配阶段**：
- [ ] 每个 `alloc_ub` 算清字节数，建预算表（附录 A 格式）
- [ ] 区分全程持久 / 阶段性 / 瞬态 buffer，标注共存阶段
- [ ] 小 buffer pad 到 32B 倍数，避免对齐浪费
- [ ] ping-pong 用 `(2, ...)` 单 buffer + 索引，不要在 if/else 内分配两个

**别名阶段**：
- [ ] 默认开 `tl.ascend_memory_planning`（默认 False）
- [ ] 不要依赖编译器溢出检查（`check_overflow` 硬编码 False）
- [ ] `annotate_address` 是独占分配，不能用于两 buffer 别名
- [ ] L1/L0C buffer 必须用 `annotate_address` 显式 pin 地址

**同步阶段**：
- [ ] 跨迭代 WAR 优先用 V→MTE2 flag（set 外 / wait 内），`barrier_all` 仅兜底
- [ ] V→MTE2 / V→MTE3 的 `set_flag` 必须在 if-block 外
- [ ] 跨核同步用 `sync_all`，`barrier_all` 是核内、不等其他核
- [ ] Cube 侧 mma→fixpipe 优先用 `unit_flag=0b11` 卸载软件 flag

**调试阶段**：
- [ ] 精度问题：`dump_tensor` 分段对比 golden
- [ ] crash/溢出：UB 预算表核对 + 邻近 buffer dump
- [ ] 挂起：检查 flag set/wait 位置
- [ ] 性能：msprof `PipeUtilization` 看 stall/wait

### 11.2 核心结论

1. **UB 是 Vector 侧最紧张资源**（192KB），但当前 VG=8/S1_BLOCK=4 配置 G-reduce 仅 140KB、余量充足——UB 紧张是 VG=16/S1_BLOCK=8 目标配置的问题。
2. **tilelang 无 buffer 别名 API 是根本约束**——`LinearScanAllocator` 禁止同地址，`annotate_address` 独占。这是 VG=16 不可行、与 AC 剩余 gap 的本质，需框架层面 alias/overlay API 才能突破。
3. **同步开销可通过 flag + unit_flag 压低**——v_sc_wait 从早期 179us 降至当前 ~23us，剩余 gap 纯 VB 维度 UB 阻塞。
4. **check_overflow 关闭 + 默认不开别名**是两大隐患——开发期必须手算预算表，不能依赖编译器。

### 11.3 适用边界

本文基于 Ascend 910B3（UB 192KB）。不同芯片 UB/L1 容量不同（A2/A3/A5 差异），跨芯片移植时需用硬件规格接口重新核算 UB 上限，本文的方法论（预算表、生命周期分析、同步约束）通用，但具体数值不可直接套用。

---

## 附录 A：LightningIndexer UB 预算完整表

以下为 tnd_tnd 样例（B=8, S1=64, S2=3072, G=24, TOP_K=2048）的完整 UB 预算。该配置下 S1_BLOCK=4→VID_S1=2、VECTOR_BASEG=8、S2_VEC_BLOCK=512、USE_SORT_CACHE=False：

| Buffer | shape | dtype | 字节数 | 生命周期 | G-reduce 存活？ |
|--------|-------|-------|--------|---------|---------------|
| topk_a_ub | (2, 4096) | float | 32KB | 全程 | ✅ |
| mm_res_ub | (2, 8, 512) | float | 32KB | G-reduce（2-slot ping-pong） | ✅ |
| merged_ub | (8192,) | float | 32KB | sort + Phase 2 | ✅ |
| weight_2d_ub | (8, 512) | float | 16KB | G-reduce | ✅ |
| reduce_tmp_ub | (8, 512) | float | 16KB | G-reduce | ✅ |
| p2_acc_ub | (4096,) | float | 16KB | output + Phase 2 | ❌ |
| cache_tmp_ub | (1, 1024) | float | 4KB | sort（sort 输出） | ✅ |
| stride2_blk_ub | (1024,) | float | 4KB | sort | ✅ |
| output_ub | (2048,) | int32 | 8KB | output | ❌ |
| topk_index_ub | (2048,) | float | 8KB | output | ❌ |
| score_topk_ub | (2048,) | float | 8KB | output | ❌ |
| output_val_ub | (2048,) | float16 | 4KB | output | ❌ |
| reduce_g_ub | (512,) | float | 2KB | G-reduce + sort | ✅ |
| index_blk_ub | (512,) | float | 2KB | sort | ✅ |
| w_raw_ub | (2, 16) | float16 | 64B | G-reduce | ✅ |
| weight_ub | (8,) | float | 32B | G-reduce | ✅ |
| mask_blk_ub | (64,) | uint8 | 64B | sort | ✅ |
| mask_topk_ub | (256,) | uint8 | 256B | output | ❌ |
| reduce_sum_tmp_ub | (1,) | float | 4B | G-reduce（dummy¹） | ✅ |
| cache_buf | (1, 1) | float | 4B | sort（dummy²） | ✅ |
| prev_bsn_ub | (1,) | int32 | 4B | 全程 | ✅ |

> ¹ tnd_tnd 下 `_NEED_VEC_PAD=False`，`reduce_sum_tmp_ub` 退化为 `(1,)` dummy；PA 小 batch 场景为 `(512,)`。
> ² `USE_SORT_CACHE=False`（S1>4），`cache_buf` 退化为 `(1,1)` dummy；S1≤4 时为 `(VID_S1*4, _KP2)`。

**G-reduce 阶段总计**（存活项）：32+32+32+16+16+4+4+2+2+0.06+0.03+0.06+0.004+0.004+0.004 ≈ **140KB**（距 192KB 上限尚有约 52KB 余量）

**output 阶段总计**：topk_a(32)+p2_acc(16)+output(8)+topk_index(8)+score_topk(8)+output_val(4)+mask_topk(0.25)+prev_bsn(0.004) ≈ **76KB**（余量充足）

---

## 附录 B：AC arch22 UB 管理对比

| 维度 | AC arch22 | TileLang | 影响 |
|------|-----------|----------|------|
| 分配 API | `TPipe::InitBuffer` + `TBuf`（字节数） | `T.alloc_ub`（shape + dtype） | TileLang 更声明式，但需自己算字节数 |
| 队列管理 | `TQue`（EnQue/DeQue 自动 V↔MTE3 sync） | 无（手动 set_flag/wait_flag） | AC 的 TQue 不阻塞 scalar，TL 的 wait_flag 阻塞 |
| 同步指令 | `SetWaitFlag`（组合 set+wait） | `set_flag` + `wait_flag`（分离） | 早期版本曾归因于 176us v_sc_wait 差距；实测 AC 也用 flag，差距本质是 UB 容量（见 7.4） |
| 跨核同步 | `CrossCoreSetFlag/WaitFlag`（细粒度） | `sync_all`（粗粒度，等所有核） | TL 更简单但开销更大；`barrier_all` 为核内 pipe_barrier |
| 内存别名 | `ReinterpretCast` 手动 + TPipe 管理（`SortedBasicBlock_` 别名 `globalTopkUb_`、`tmpBuf_` 68KB 双职责） | 规划器自动复用生命周期错开的地址；但**无 alias/overlay API**，`annotate_address` 同地址会被 `LinearScanAllocator` 拒绝 | TL 无法复刻 AC 的手动别名，VB=16 装不下的根因 |
| 溢出检查 | N/A（TPipe 手动管理，开发者自负） | `check_overflow=false`（静默） | TL 可能静默溢出 |
| in-place 操作 | MrgSort4 支持 dst 与 src 部分重叠 | merge_sort 不支持 dst=src0 | TL 需要 32KB 额外 merged_ub |
| Buffer 复用 | ReinterpretCast 同地址不同 dtype | 无等价物（需 alloc_ub 两个 buffer） | TL 浪费 UB |

**核心差异总结**：早期版本中，AC 的 `TQue` + `SetWaitFlag` 体系让 scalar pipe 几乎不阻塞（v_sc_wait=3us），而 TileLang 的 `set_flag + wait_flag` 让 scalar pipe 大量阻塞（v_sc_wait=179us）。但经同步重构后当前 v_sc_wait 已降至 ~23us 级，且 AC 同样使用 flag 机制——**剩余差距的根本原因是 UB 别名能力缺失**（AC 靠手动别名装下 VB=16/S1_BLOCK=8，TL 装不下），而非 flag API 形式。弥合需框架层面提供 buffer alias/overlay API。

---

## 附录 C：优化历程 UB 发现时间线

| 阶段 | UB 相关发现 | 影响 |
|------|-----------|------|
| v10_a5 2-slot cache | T.tile.sort 内部 temp 与 save buffer 地址重叠 → 507015 crash | 发现需要 T.annotate_address |
| v11_c1 3-slot ping-pong | mm_res_ub 从 2-slot 增到 3-slot，UB 增加 16KB | 后回退 2-slot（3-slot 未带来净收益） |
| v11_c2 Load Dispatch | V→MTE2 flag 在 if-block 内死锁 | 发现 flag 方向性约束 |
| v11_c2_s3 barrier_all 删除 | merge loop barrier_all 不能删（WAR on topk_a_ub） | 发现跨迭代 WAR 问题 |
| v11_p0b batch MTE3 | 减少 wait_flag 数量但 v_sc_wait 不变 | 发现 wait_flag 指令数非瓶颈，MTE3 执行时间才是 |
| VG=16/S1_BLOCK=8 分析 | topk_a + mm_res 必须共存，目标配置 G-reduce 超 44KB | 确认 UB 别名缺失是算法优化的硬约束 |
| 当前（unit_flag + 2-slot） | Cube 侧 unit_flag=0b11 省 flag，mm_res 回 2-slot；v_sc_wait 179→23us | 剩余 gap 纯 VB=16 UB 阻塞（见 7.6/8） |

---

## 附录 D：性能测量方法

文中性能数据（v_sc_wait、aiv_time、Task Duration 等）均来自 msprof 采集。为保证可复现，记录测量方法。

### D.1 msprof 采集命令

```bash
msprof op --kernel-name="main" --output=<output_dir> python <perf_script>.py
```

- `--kernel-name="main"`：指定采集的 kernel 名（TileLang 生成入口）
- 默认生成 `OpBasicInfo.csv`、`PipeUtilization.csv` 等，无需额外 `--application` / `--aic-metrics` 参数
- 多核场景取 20 核均值；单核指标取 cube0 / vector0 / vector1 对应行

### D.2 关键 CSV 列

| CSV | 行 | 关键列 | 含义 |
|-----|---|--------|------|
| OpBasicInfo | — | Task Duration(us) | 算子总耗时 |
| PipeUtilization | cube0 | aic_time | Cube 核总时间 |
| PipeUtilization | cube0 | aic_fixpipe_time | fixpipe 时间（unit_flag 下含等 mma 释放 L0C） |
| PipeUtilization | cube0 | aic_scalar_mte1_stall_time | scalar 等 mte1 的 stall |
| PipeUtilization | vector0/1 | aiv_time | Vector 核总时间 |
| PipeUtilization | vector0/1 | aiv_scalar_wait_id13_time | C1V1 跨核 wait（eventId 13） |
| PipeUtilization | vector0 | aiv_scalar_mte2_stall_time | scalar 等 mte2 的 stall（VB 维度瓶颈指标） |

### D.3 对比基线

AC arch22 基线用同 shape 同 tiling 跑 msprof，取相同 CSV 列对比。注意 AC 与 TileLang 的 tiling 对齐（BLOCK_M_L0 / BLOCK_K / BLOCK_N、S1_BLOCK、VB）后，差距才反映框架机制差异而非 tiling 差异。

### D.4 数据可复现性

- 性能数据受环境（CANN 版本、驱动、核频、热状态）影响，绝对值会浮动，相对 gap（AC vs TL）更稳定
- 本文数据基于 CANN 9.0.0 / 910B3，跨环境复现以相对 gap 为准

---

*文档完*
