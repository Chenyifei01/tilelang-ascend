# API 验证结果汇总

> 验证时间: 2026-07-20 | 设备: Ascend910B3 | CANN 9.0

## 验证清单

| # | 文件 | 验证目标 | 结果 | 关键发现 |
|---|------|---------|------|---------|
| 1 | `test_merge_sort_4way.py` | `T.tile.merge_sort` 4-way merge | ✅ PASS | 4-way 通过位置参数调用，切片作为 src 可行 |
| 2 | `test_sort_cache_1d.py` | sort 缓存优化 1D 平铺方案 | ✅ PASS | 1D 平铺 + T.copy 搬运 + 切片 src 可行 |
| 3 | `test_3layer_tiling.py` | 3 层 tiling + L1 行切片 | ⚠️ 部分通过 | **transpose_B=True + L1 行切片不可行** |
| 4 | `test_pipeline_cv_combine.py` | num_stages=3-4 + threads=2 + CV_COMBINE | ✅ PASS | num_stages=3-4 可行，buffer 须在外部分配 |
| 5 | `test_pipe_barrier_v.py` | `T.pipe_barrier("v")` 替代 `barrier_all` | ✅ PASS | pipe_barrier("v") 精度与 barrier_all 一致 |

## 详细结果

### 验证 1: 4-way merge_sort ✅

```
基础 4-way merge: ✅ PASS (Values match golden)
切片 src 4-way:   ✅ PASS (Values match golden)
```

**关键结论**:
- `T.tile.merge_sort(dst, s0, s1, s2, s3)` 通过位置参数指定 4-way
- **不存在 `way` 参数**（v7 设计文档中的 `way=cache_idx` 是错误的）
- `merge_sort` 支持 BufferRegion 切片作为 src
- 1D 平铺 buffer 的切片作为 src 可行

### 验证 2: sort 缓存优化 1D 平铺方案 ✅

```
4-way cache 方案:   ✅ PASS (Values match golden)
2-way standard:     ✅ PASS (Values match golden)
```

**关键结论**:
- v7 sort 缓存优化的 1D 平铺方案可行
- 流程: sort → T.copy 到 1D cache 偏移 → 4-way merge_sort 用切片 src
- `T.tile.sort` **不支持 BufferRegion 切片**作为 dst，需要先 sort 到临时 buffer 再 T.copy

### 验证 3: 3 层 tiling ⚠️ 关键发现

```
3 层 tiling (L1 切片):        ❌ FAIL (diff=66.5156)
4 层 serial 嵌套 (L1 切片):   ❌ FAIL (diff=66.5156)
独立 buffer (无切片):          ✅ PASS (diff=0.0156)
单层 tiling 基线:              ✅ PASS (diff=0.0156)
```

**⚠️ 关键发现**: **`T.gemm_v0(transpose_B=True)` 不支持 L1 行切片**！

- L1 行切片 + transpose_B=True → 精度错误（diff=66.5）
- 独立 buffer（无切片）+ transpose_B=True → 精度正确（diff=0.0156）
- 单层 tiling + transpose_B=True → 精度正确

**对 v7 设计的影响**:
- v7 设计的 "L1 行切片传 gemm_v0(transpose_B=True)" **不可行**
- **替代方案**: 为每个子 tile 分配独立的 (128, 128) L1 buffer
  - L1 占用: 4×(128×128)×2 bytes = 128KB(A) + 128KB(B) = 256KB < 512KB ✓
- **或**: 用 Expert 模式（T.mma + T.copy(transpose=True)），如 lightning_indexer 现有实现

### 验证 4: num_stages + threads=2 + CV_COMBINE ✅

```
num_stages=3: ✅ PASS (diff=0.0312)
num_stages=4: ✅ PASS (diff=0.0312)
num_stages=2: ✅ PASS (diff=0.0312)
```

**关键结论**:
- num_stages=3 和 4 都编译通过且精度正确
- threads=2 + AUTO_CV_COMBINE 组合可行
- **关键**: buffer 必须在 `T.Pipelined` body **外部**分配
- threads=2 时 Kernel 用 `as (cid)` 解包（1 个值），不是 `as (cid, _)`

### 验证 5: pipe_barrier("v") ✅

```
pipe_barrier('v'):   ✅ PASS (diff=0.0312)
barrier_all:         ✅ PASS (diff=0.0312)
no explicit sync:    ✅ PASS (diff=0.0312)
```

**关键结论**:
- `T.pipe_barrier("v")` 编译通过，精度与 barrier_all 一致
- 可替代 barrier_all 用于 Vector 管线内部同步
- 三种同步方式精度完全相同（diff=0.0312）

## API 使用注意事项

### 1. Kernel 解包规则
| threads | 解包方式 | 示例 |
|---------|---------|------|
| 1（默认） | `as (cid, _)` | `with T.Kernel(blocks, is_npu=True) as (cid, _):` |
| 2 | `as (cid)` | `with T.Kernel(blocks, threads=2, is_npu=True) as (cid):` |

### 2. T.tile.sort vs T.tile.merge_sort
| API | BufferRegion 切片支持 | 用法 |
|-----|---------------------|------|
| `T.tile.sort` | ❌ 不支持（dst 仅 Buffer） | `T.tile.sort(dst_buffer, src_buffer, actual_num)` |
| `T.tile.merge_sort` | ✅ 支持（src 可切片） | `T.tile.merge_sort(dst, s0, s1, s2, s3)` |

### 3. T.gemm_v0 + transpose_B
| 场景 | 可行性 |
|------|--------|
| 完整 buffer + transpose_B=True | ✅ |
| 独立 buffer + transpose_B=True | ✅ |
| **L1 行切片 + transpose_B=True** | ❌ 精度错误 |
| L1 行切片 + transpose_B=False | 未测试 |

### 4. T.Pipelined + buffer 分配
| buffer 位置 | num_stages 影响 | 建议 |
|------------|----------------|------|
| Pipelined body 内部 | ring-buffer 放大 UB | ❌ 避免 |
| Pipelined body 外部 | 不影响 UB | ✅ 推荐 |

## 对 v7 设计的修正建议

1. **§8.5 Cube GEMM tiling**: 放弃 "L1 行切片传 gemm_v0"，改用独立 buffer 方案
   - 为每个 (M_L0, N_L0) 子 tile 分配独立的 L1 buffer
   - L1 占用 256KB < 512KB，可行

2. **§8.8 sort 缓存优化**: 修正 API 用法
   - 删除 `way=cache_idx`（不存在）
   - sort 到临时 buffer → T.copy 到 1D cache 偏移 → merge_sort 用切片 src
   - 动态路数用 if-elif 分支（2/3/4-way）

3. **§6 流水线配置**: 确认 num_stages=3-4 可行
   - buffer 在 Pipelined body 外部分配
   - threads=2 时用 `as (cid)` 解包

4. **§8.5 同步优化**: 确认 pipe_barrier("v") 可行
   - Vector 管线内部同步可用 pipe_barrier("v")
   - 跨管线同步仍需 barrier_all 或 set_flag/wait_flag
