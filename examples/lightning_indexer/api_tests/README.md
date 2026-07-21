# API 验证测试

v7 设计中关键 API 的最小验证用例，在实际应用到 `lightning_indexer.py` 之前确认 API 准确用法。

## 验证结果（2026-07-20）

| # | 文件 | 验证目标 | 结果 | 关键发现 |
|---|------|---------|------|---------|
| 1 | `test_merge_sort_4way.py` | 4-way merge_sort | ✅ PASS | 位置参数调用，切片 src 可行 |
| 2 | `test_sort_cache_1d.py` | sort 缓存 1D 平铺 | ✅ PASS | 1D 平铺 + T.copy + 切片 src |
| 3 | `test_3layer_tiling.py` | 3 层 tiling + L1 切片 | ⚠️ 部分通过 | **transpose_B=True + L1 行切片不可行** |
| 4 | `test_pipeline_cv_combine.py` | num_stages=3-4 + CV_COMBINE | ✅ PASS | buffer 须在外部分配 |
| 5 | `test_pipe_barrier_v.py` | pipe_barrier("v") | ✅ PASS | 精度与 barrier_all 一致 |

## 关键发现

### ⚠️ `T.gemm_v0(transpose_B=True)` 不支持 L1 行切片

- L1 行切片 + transpose_B=True → 精度错误（diff=66.5）
- 独立 buffer（无切片）→ 精度正确（diff=0.0156）
- **v7 设计需改用独立 buffer 方案**

### `T.tile.sort` 不支持 BufferRegion 切片

- 签名仅 `Buffer`，不支持切片作为 dst
- `T.tile.merge_sort` 支持切片作为 src
- **sort 缓存优化需用 1D 平铺 + T.copy 搬运**

### `T.tile.merge_sort` 无 `way` 参数

- 4-way 通过位置参数: `merge_sort(dst, s0, s1, s2, s3)`
- 动态路数用 if-elif 分支

## 运行方式

```bash
cd /home/tilelang-ascend
source set_env.sh
python examples/lightning_indexer/api_tests/test_<name>.py
```

## 文件

- `VERIFICATION_RESULTS.md` — 详细验证结果和对 v7 设计的修正建议
- `api_research_notes.md` — API 调研笔记（源码引用）
- `test_*.py` — 5 个验证脚本
