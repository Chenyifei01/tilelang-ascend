#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Test lightning_indexer TileLang kernel against official GeneralizedLI golden.

Uses:
  - li_pytest/lightning_indexer_golden.py :: GeneralizedLI (CPU golden)
  - li_pytest/result_compare_method.py :: check_result (official verification)
  - examples/lightning_indexer/lightning_indexer.py :: TileLang kernel (replaces torch_npu.npu_lightning_indexer)

Test cases:
  - li_default_a2 from test_lightning_indexer_paramset.py (BSND+PA_BSND, bf16)
  - 4 layout scenarios from li_0720.py examples
  - Extensible to 37 cases from test_cases.xlsx
"""

import sys
import os
import math
import torch
import torch_npu
import numpy as np

# Add li_pytest to path for official golden + check_result
_LI_PYTEST_DIR = "/home/tilelang-ascend/li_pytest"
if _LI_PYTEST_DIR not in sys.path:
    sys.path.insert(0, _LI_PYTEST_DIR)

import lightning_indexer_golden
import result_compare_method
from lightning_indexer import lightning_indexer as tl_lightning_indexer


def _to_tensor(shape, dtype, datarange, device="npu:0"):
    """Generate random tensor in given dtype and range."""
    t = torch.tensor(np.random.uniform(datarange[0], datarange[1], shape), dtype=dtype)
    return t.to(device)


def li_output_tilelang(params):
    """Run golden + TileLang kernel, return (cpu_result, npu_result, topk_value, sparse_value).

    Mirrors lightning_indexer_golden.li_output_single but replaces
    torch_npu.npu_lightning_indexer with our TileLang wrapper.
    """
    (batch_size, q_seq, k_seq, q_t_size, k_t_size,
     q_head_num, k_head_num, head_dim, block_size, block_num,
     qk_dtype, weight_dtype, actual_seq_dtype, act_seq_q, act_seq_k,
     layout_query, layout_key, sparse_count, sparse_mode,
     query_datarange, key_datarange, weights_datarange, return_value) = params

    # Build GeneralizedLI golden
    test_li = lightning_indexer_golden.GeneralizedLI(
        batch_size, q_seq, k_seq, q_t_size, k_t_size,
        q_head_num, k_head_num, head_dim, block_size, block_num,
        qk_dtype, weight_dtype, actual_seq_dtype,
        act_seq_q, act_seq_k, layout_query, layout_key,
        sparse_count, sparse_mode)

    # actual_seq tensors
    if act_seq_q is None:
        actual_seq_lengths_query = torch.tensor(
            np.random.uniform(q_seq, q_seq, batch_size), dtype=actual_seq_dtype).npu()
    else:
        actual_seq_lengths_query = torch.tensor(act_seq_q, dtype=actual_seq_dtype).npu()

    if act_seq_k is None:
        actual_seq_lengths_key = torch.tensor(
            np.random.uniform(k_seq, k_seq, batch_size), dtype=actual_seq_dtype).npu()
    else:
        actual_seq_lengths_key = torch.tensor(act_seq_k, dtype=actual_seq_dtype).npu()

    # Shape derivation
    if layout_query == "BSND":
        query_shape = [batch_size, q_seq, q_head_num, head_dim]
        weights_shape = [batch_size, q_seq, q_head_num]
    elif layout_query == "TND":
        query_shape = [q_t_size, q_head_num, head_dim]
        weights_shape = [q_t_size, q_head_num]

    if layout_key == "BSND":
        key_shape = [batch_size, k_seq, k_head_num, head_dim]
    elif layout_key == "TND":
        key_shape = [k_t_size, k_head_num, head_dim]
    elif layout_key == "PA_BSND":
        k_max_s2 = math.floor(max(act_seq_k))
        key_shape = [batch_size, k_head_num, k_max_s2, head_dim]

    # Random data
    query = _to_tensor(query_shape, qk_dtype, query_datarange)
    weights = _to_tensor(weights_shape, weight_dtype, weights_datarange)
    if weights_datarange[0] >= 0:
        weights = weights.abs()  # weights should be non-negative
    key = _to_tensor(key_shape, qk_dtype, key_datarange)
    block_table = None
    key_cpu = key

    # PA_BSND: build block_table + scatter key
    if layout_key == "PA_BSND":
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size)
        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in act_seq_k:
            cur_cmp_act_k = math.floor(cur_act_k)
            cur_key_block_num = math.ceil(cur_cmp_act_k / block_size)
            key_block_num_per_batch.append(cur_key_block_num)
            key_block_num_sum += cur_key_block_num
        if block_num < key_block_num_sum:
            raise ValueError(f"key actual block num < needed block num")

        block_id_list = np.arange(block_num)
        block_id_list = np.random.permutation(block_id_list).astype(np.int32)
        cur_block_id = 0
        block_table_np = np.full((batch_size, k_max_block_num_per_batch),
                                  fill_value=-1, dtype=np.int32)
        batch_idx = 0
        for cur_block_id_threshold in key_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                block_table_np[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1

        key_expand = torch.zeros(
            (batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim),
            dtype=qk_dtype)
        key_expand[:, :, :k_max_s2, :] = key_cpu
        key = torch.zeros((block_num, block_size, k_head_num, head_dim), dtype=qk_dtype)
        for i_batch in range(batch_size):
            for i_block, cur_block_id in enumerate(block_table_np[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                for i_n in range(k_head_num):
                    key[cur_block_id, :, i_n, :] = \
                        key_expand[i_batch, i_n, block_start_pos:block_start_pos + block_size, :]
        key = key.npu()
        block_table = torch.from_numpy(block_table_np).to(dtype=torch.int32).npu()

    # CPU golden
    cpu_result, topk_value = test_li.forward(
        query, key_cpu, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table)
    # 注意: 不做 torch.sort — check_result 期望按 score 降序排列的 index
    # cur_cpu[-1] 应该是 score 最小的 index（topk 边界），value_bm = 其 score
    # 如果 sort 会破坏顺序，导致 value_bm 取错值，误报精度失败

    # TileLang kernel (replaces torch_npu.npu_lightning_indexer)
    npu_result, sparse_value = tl_lightning_indexer(
        query, key, weights,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=sparse_count,
        sparse_mode=sparse_mode,
        return_value=return_value if return_value is not None else False,
    )
    torch.npu.synchronize()
    # 同样不做 sort
    return cpu_result, npu_result, topk_value, sparse_value


def run_test(name, params):
    """Run a single test case and print result."""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"  layout: {params[15]}+{params[16]}, dtype: {params[10]}, "
          f"B={params[0]}, S1={params[1]}, S2={params[2]}, "
          f"N1={params[5]}, sparse_count={params[17]}, mode={params[18]}")
    print(f"{'='*70}")

    try:
        cpu_result, npu_result, topk_value, sparse_value = li_output_tilelang(params)
    except Exception as e:
        print(f"  [ERROR] Kernel execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Debug: print golden scores for first s1 row to compare with kernel DBG output
    batch_size = params[0]
    layout_query = params[15]
    if layout_query == "BSND":
        print("  [GOLDEN] topk_value[0, 0, 0, :8] =", topk_value[0, 0, 0, :8].cpu().tolist())

    # Official check_result
    result, fulfill_percent = result_compare_method.check_result(
        cpu_result, npu_result, topk_value, sparse_value, params)

    passed = (result == "Pass")
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n  [{name}] {status}  (fulfill={fulfill_percent:.2f}%)")

    # Precision analysis: compare golden vs kernel topk scores
    if not passed and layout_query == "BSND":
        b, s1, n2 = 0, 0, 0
        g_idx = cpu_result[b, s1, n2].cpu()
        k_idx = npu_result[b, s1, n2].cpu()
        g_scores = topk_value[b, n2, s1].cpu()
        g_valid = g_idx[g_idx >= 0]
        k_valid = k_idx[k_idx >= 0]
        g_topk_scores = g_scores[g_valid]
        k_topk_scores = g_scores[k_valid]
        print(f"  [ANALYSIS] b={b} s1={s1} n2={n2}")
        print(f"  Golden top-5 idx: {g_valid[:5].tolist()}")
        print(f"  Kernel top-5 idx: {k_valid[:5].tolist()}")
        print(f"  Golden top-5 sco: {[f'{v:.4f}' for v in g_topk_scores[:5].tolist()]}")
        print(f"  Kernel top-5 sco: {[f'{v:.4f}' for v in k_topk_scores[:5].tolist()]}")
        g_set = set(g_valid.tolist())
        k_set = set(k_valid.tolist())
        only_g = list(g_set - k_set)[:5]
        only_k = list(k_set - g_set)[:5]
        if only_g or only_k:
            print(f"  Only golden: {only_g}")
            print(f"  Only kernel: {only_k}")
            for idx in only_g + only_k:
                print(f"    idx={idx} score={g_scores[idx].item():.6f}")
    elif layout_query == "TND":
        print("  [GOLDEN] topk_value[0, 0, 0, :8] =", topk_value[0, 0, 0, :8].cpu().tolist())

    return passed


# ============================================================
# Test cases
# ============================================================

def test_li_default_a2():
    """li_default_a2: BSND+PA_BSND, bf16, B=18, S1=3, S2=3072, N1=16, mode=3."""
    params = (
        18,    # batch_size
        3,     # q_seq
        3072,  # k_seq
        1,     # q_t_size
        1,     # k_t_size
        16,    # q_head_num
        1,     # k_head_num
        128,   # head_dim
        128,   # block_size
        469,   # block_num
        torch.bfloat16,  # qk_dtype
        torch.bfloat16,  # weight_dtype
        torch.int32,     # actual_seq_dtype
        [3]*18,          # act_seq_q
        [3016]*17 + [3072],  # act_seq_k
        "BSND",          # layout_query
        "PA_BSND",       # layout_key
        2048,            # sparse_count
        3,               # sparse_mode
        [-1, 1],         # query_datarange
        [-1, 1],         # key_datarange
        [-130, 130],     # weights_datarange
        None,            # return_value
    )
    return run_test("li_default_a2", params)


def test_bsnd_bsnd_bf16():
    """BSND+BSND, bf16, B=2, S1=3, S2=32768, N1=64, mode=3."""
    params = (
        2, 3, 32768, 1, 1,
        64, 1, 128, 128, 1,
        torch.bfloat16, torch.bfloat16, torch.int32,
        [3, 3], [32768, 32768],
        "BSND", "BSND",
        2048, 3,
        [-1, 1], [-1, 1], [-130, 130],
        None,
    )
    return run_test("bsnd_bsnd_bf16", params)


def test_bsnd_bsnd_fp16():
    """BSND+BSND, fp16, B=2, S1=3, S2=32768, N1=64, mode=3."""
    params = (
        2, 3, 32768, 1, 1,
        64, 1, 128, 128, 1,
        torch.float16, torch.float16, torch.int32,
        [3, 3], [32768, 32768],
        "BSND", "BSND",
        2048, 3,
        [-1, 1], [-1, 1], [-130, 130],
        None,
    )
    return run_test("bsnd_bsnd_fp16", params)


def test_bsnd_pa_bf16():
    """BSND+PA_BSND, bf16, B=2, S1=1, N1=8, block_size=128, mode=0."""
    params = (
        2, 1, 2048, 1, 1,
        8, 1, 128, 128, 32,
        torch.bfloat16, torch.bfloat16, torch.int32,
        [1, 1], [1024, 2048],
        "BSND", "PA_BSND",
        2048, 0,
        [-1, 1], [-1, 1], [-130, 130],
        None,
    )
    return run_test("bsnd_pa_bf16", params)


def test_tnd_tnd_fp16():
    """TND+TND, fp16, B=8, S1=64, S2=3072, N1=24, mode=0.
    Note: TND act_seq uses prefix-sum format per official spec.
    """
    params = (
        8, 64, 3072, 512, 24576,
        24, 1, 128, 128, 1,
        torch.float16, torch.float16, torch.int32,
        [64, 128, 192, 256, 320, 384, 448, 512],      # act_seq_q: prefix-sum
        [3072, 6144, 9216, 12288, 15360, 18432, 21504, 24576],  # act_seq_k: prefix-sum
        "TND", "TND",
        2048, 0,
        [-1, 1], [-1, 1], [-130, 130],
        None,
    )
    return run_test("tnd_tnd_fp16", params)


def test_tnd_pa_fp16():
    """TND+PA_BSND, fp16, B=2, N1=8, block_size=128, mode=0.
    Note: TND query act_seq uses prefix-sum; PA_BSND key act_seq uses per-batch length.
    """
    params = (
        2, 2048, 8192, 3072, 1,
        8, 1, 128, 128, 128,
        torch.float16, torch.float16, torch.int32,
        [1024, 3072],    # act_seq_q: prefix-sum (batch0=1024, batch1=1024+2048=3072)
        [4096, 8192],    # act_seq_k: per-batch length (PA_BSND)
        "TND", "PA_BSND",
        2048, 0,
        [-1, 1], [-1, 1], [-130, 130],
        None,
    )
    return run_test("tnd_pa_fp16", params)


def test_bsnd_bsnd_small_weights():
    """BSND+BSND, fp16, mode=3, weights in [0,1] (like li_0720.py examples)."""
    params = (
        2, 3, 32768, 1, 1,
        64, 1, 128, 128, 1,
        torch.float16, torch.float16, torch.int32,
        [3, 3], [32768, 32768],
        "BSND", "BSND",
        2048, 3,
        [-1, 1], [-1, 1], [0, 1],   # weights [0,1] like li_0720.py
        None,
    )
    return run_test("bsnd_bsnd_small_weights", params)


def test_bsnd_bsnd_return_value():
    """BSND+BSND with return_value=True, fp16, mode=0."""
    params = (
        2, 4, 4096, 1, 1,
        16, 1, 128, 128, 1,
        torch.float16, torch.float16, torch.int32,
        [4, 4], [4096, 4096],
        "BSND", "BSND",
        2048, 0,
        [-1, 1], [-1, 1], [-130, 130],
        True,  # return_value
    )
    return run_test("bsnd_bsnd_return_value", params)


# ============================================================
# Main
# ============================================================

ALL_TESTS = {
    "li_default_a2": test_li_default_a2,
    "bsnd_bsnd_bf16": test_bsnd_bsnd_bf16,
    "bsnd_bsnd_fp16": test_bsnd_bsnd_fp16,
    "bsnd_pa_bf16": test_bsnd_pa_bf16,
    "tnd_tnd_fp16": test_tnd_tnd_fp16,
    "tnd_pa_fp16": test_tnd_pa_fp16,
    "bsnd_bsnd_small_weights": test_bsnd_bsnd_small_weights,
    "bsnd_bsnd_return_value": test_bsnd_bsnd_return_value,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test lightning_indexer TileLang kernel")
    parser.add_argument("test_name", nargs="?", default="all",
                        help=f"Test name ({'|'.join(ALL_TESTS.keys())}|all)")
    args = parser.parse_args()

    torch_npu.npu.set_device(0)
    np.random.seed(42)

    if args.test_name == "all":
        tests = list(ALL_TESTS.items())
    else:
        if args.test_name not in ALL_TESTS:
            print(f"Unknown test: {args.test_name}")
            print(f"Available: {', '.join(ALL_TESTS.keys())}")
            sys.exit(1)
        tests = [(args.test_name, ALL_TESTS[args.test_name])]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            ok = fn()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*70}")
    print(f"Summary: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*70}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
