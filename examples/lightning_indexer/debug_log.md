## Attempt 1 — 2026-07-20T21:20:00Z
- mode: first_impl
- classification: precision_fail
- fail_category: precision (bfloat16 QK precision vs golden float32)
- test_level: l0
- coverage: L0:8 (of 15 planned), L1:0, L2:0, Boundary:0
- boundary_warnings: none
- changes:
  - Created `lightning_indexer.py`: Expert mode single kernel (T.Scope C/V, 4D tensors, T.tile.mul, T.reduce_sum, T.tile.topk). Supports BSND + PA_BSND layouts, sparse_mode 0/3, return_value, BLOCK_N=128.
  - Created `test_lightning_indexer.py`: Import kernel + GeneralizedLI golden + check_result compare. 8 of 15 L0 cases implemented (BSND + PA_BSND).
- error_summary:
  - PA_BSND block_table shape mismatch (fixed: use per-batch max_blocks, not total block_num)
  - L0C→UB direct copy not supported (fixed: use GM workspace intermediate)
  - T.copy GM→UB for weights inside T.Scope("V") initially failed (fixed: use 4D tensor + T.barrier_all pattern matching existing example)
  - T.tile.mul with weight_ub[g_i] works when following existing example's exact pattern (4D weights, barrier_all, per-row mul)
  - BSND cases have 83-99% accuracy due to bfloat16 GEMM precision vs golden's float32 BMM
  - Large cases (B=18, S2=3072) need longer compilation time
  - TND and segment topk not yet implemented
- design_error_reason: none
- rollback: no
- backup_path: n/a
- instrumentation_cleaned: n/a
- next_hint:
  1. Fix BSND precision: either use float32 Q/K for GEMM or adjust comparison threshold for bfloat16 cases
  2. Add TND layout support (strided DMA, prefix sum act_seq reading)
  3. Add segment topk for S2 > 16384 (cases 8, 9, 15)
  4. Add return_value testing (cases 6, 9, 10, 15)
  5. Add remaining L0 cases (TND_TND, TND_PA_BSND, large_s2, large_s1, block_size_768)
  6. Run full L0 suite with extended timeout
  7. Key learning: T.copy inside T.Scope("V") requires T.barrier_all() before/after and 4D tensor format for weights
