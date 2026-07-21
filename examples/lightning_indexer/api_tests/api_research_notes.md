# API 调研笔记（基于 explore agent 调研结果）

## 1. T.tile.merge_sort

### 函数签名
```python
def merge_sort(
    dst: Buffer | BufferRegion,
    src0: Buffer | BufferRegion,
    src1: Buffer | BufferRegion,
    src2: Buffer | BufferRegion | None = None,
    src3: Buffer | BufferRegion | None = None,
):
```

### 关键结论
- **没有 `way` 参数**：路数由非 None 源 buffer 个数决定（2-4）
- **4-way 调用**：`merge_sort(dst, s0, s1, s2, s3)`
- **支持 BufferRegion 切片**：作为 src 可行（`retrieve_ptr` 处理偏移）
- **num_ways 是编译期常量**：不能用运行时变量选择路数
- **动态路数替代方案**：if-elif 分支显式调用 2/3/4-way

### 现有示例
- `examples/sort/example_merge_sort.py` — 2/3/4-way 完整测试
- `examples/lightning_indexer/lightning_indexer.py:785` — 2-way with 2D slices

---

## 2. T.tile.sort

### 函数签名
```python
def sort(dst: Buffer, src: Buffer, actual_num: PrimExpr):
```

### 关键结论
- **不支持 BufferRegion 切片**：签名仅 `Buffer`，直接调用 `dst.access_ptr("w")`
- **支持多维完整 Buffer**：测试中用 2D `(block_M, ub_N)`
- **dst 必须是 src 的 2 倍大小**（interleaved value-index 对，降序）

### v7 设计文档的错误
- `T.tile.sort(sorted_cache_ub[cache_idx], ...)` 不可行
- 替代方案：1D 平铺 buffer + T.copy 搬运，或 4 个独立 buffer + if 分支

---

## 3. T.pipe_barrier

### 函数签名
```python
def pipe_barrier(pipe: _pipe):  # _pipe = Literal["fix", "mte1", "mte2", "mte3", "m", "v", "s"]
```

### 关键结论
- **核内同步**：仅同步指定管线
- **`pipe_barrier("v")`**：仅等 Vector 管线，不等 Cube/MTE
- **`barrier_all()` = `pipe_barrier("ALL")`**：等所有管线
- **A5 限制**：pipe_barrier("v") 在 A5 不支持（当前 Ascend910B3 不受影响）
- **替代 barrier_all 条件**：仅 Vector 管线内部同步时可行；跨管线依赖不可替代

### 现有示例
- `examples/xattention/xattention.py:890,898,914,920` — Vector 内部用 `pipe_barrier("v")`
- `examples/HISA/paged_block_sparse_mqa_attn_expert.py:320` — `T.copy` 后插 `pipe_barrier("v")`

---

## 4. T.Pipelined

### 函数签名
```python
def Pipelined(start, stop=None, num_stages=0, order=None, stage=None, sync=None, group=None, cross_interval=1):
```

### 关键结论
- **num_stages 校验**：`CHECK(num_stages >= 1)`，**无上界限制**
- **num_stages=3 有已验证示例**：`examples/pipeline/matmul_add_pipeline.py:46`
- **num_stages=8 UB 超限**：`flash_attn_bhsd_auto_pipeline_h16_d128.py` KNOWN BROKEN
- **关键前提**：UB buffer 必须在 `T.Pipelined` body **外部**分配，否则 ring-buffer 放大 UB

### 现有示例
| num_stages | 文件 | 状态 |
|------------|------|------|
| 2 | `examples/pipeline/matmul_add_pipeline.py:57` | ✅ |
| 3 | `examples/pipeline/matmul_add_pipeline.py:46` | ✅ |
| 3 | `examples/pipeline/gemm_v0_pipeline.py:49` | ✅ |
| 8 | `examples/flash_attention/fa_opt/...` | ⚠️ BROKEN |

---

## 5. threads=2 + AUTO_CV_COMBINE

### 配置方式
```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
# threads=2 在 T.Kernel 中指定
with T.Kernel(blocks, threads=2, is_npu=True) as (cid,):
```

### 关键结论
- **threads=2**：AIC:AIV=1:2 模式，Cube+Vector 双核
- **AUTO_CV_COMBINE**：CombineCV Pass 将 kernel body 分裂为 cube_code + vec_code
- **5 个 developer_mode 示例验证可运行**
- **可与 T.Pipelined 组合**

### 现有示例
- `examples/developer_mode/matmul_add_developer.py`
- `examples/developer_mode/gelu_mul_developer.py`
- `examples/developer_mode/flash_attn_bshd_developer.py`
- `examples/developer_mode/sparse_flash_attn_developer_vid_reduce.py`

---

## 6. 3 层 tiling + L1 行切片

### T.gemm_v0 签名
```python
def gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False, kL0Size=128, n_actual=None):
```

### 关键结论
- **支持 BufferRegion 切片**：A/B/C 都支持（`_retrieve_ptr` 处理偏移）
- **只支持行切片**（前导维度切片，内存连续）；**禁止列切片**
- **kL0Size 只切分 K 维度**，不切分 M/N
- **M/N 来自 buffer shape**：L0 层切分 M/N 需通过 L1 buffer 行切片实现

### 4 层嵌套循环可行性
- **lightning_indexer 已有 4 层 T.serial 嵌套**（n2→g→m→n），编译通过
- **chunk_gated_delta_rule 证明 gemm_v0 接受 L1 切片**
- **未验证的组合**：M/N 在 L0 层切分（256→128）+ 行切片

### L1/L0 buffer 分配
- **Developer 模式**：`alloc_shared`（→L1）+ `alloc_fragment`（→L0C）
- **Expert 模式**：`alloc_L1` / `alloc_L0A` / `alloc_L0B` / `alloc_L0C`
- **多 buffer 方式**：3D buffer + slot 维度，或独立命名 buffer
- **HISA**：4×K L1 + 4×L0C + 2×L0B ping-pong，用 T.mma

### 硬件容量限制
- L1 总量 ~512KB/核
- L0A: 64KB, L0B: 64KB, L0C: 512KB/核
