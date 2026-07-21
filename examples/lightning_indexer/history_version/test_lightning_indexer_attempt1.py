"""Test file for lightning_indexer kernel.

Imports kernel from lightning_indexer.py, golden from lightning_indexer_golden.py,
and compare method from result_compare_method.py.

L0 test cases: 15 cases from DESIGN.md §9.2.
Usage: python test_lightning_indexer.py --level L0
"""

import sys
import os
import argparse
import math
import numpy as np
import torch
import torch_npu

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LI_PYTEST_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "li_pytest")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, LI_PYTEST_DIR)

import tilelang
from lightning_indexer import lightning_indexer, NUM_CORES, BLOCK_N
from lightning_indexer_golden import GeneralizedLI
from result_compare_method import check_result


def lightning_indexer_wrapper(
    query, key, weights, actual_seq_q, actual_seq_k, block_table,
    layout_query, layout_key, sparse_count, sparse_mode, return_value,
    qk_dtype, weight_dtype,
):
    """Wrapper that compiles and calls the kernel."""
    # Extract compile-time params
    if layout_query == "BSND":
        B, S1, N1, D = query.shape
    else:
        # TND not fully supported yet
        raise NotImplementedError(f"layout_query={layout_query} not supported")

    N2 = 1
    G = N1 // N2
    D = query.shape[3]  # Use actual D from query, not hardcoded

    if layout_key == "BSND":
        S2 = key.shape[1]
        block_size = BLOCK_N
        max_block_num = 1
    elif layout_key == "PA_BSND":
        block_size = key.shape[1]
        max_block_num = key.shape[0]  # TOTAL physical blocks (key's first dim)
        S2 = max(actual_seq_k.tolist())
        # Pad block_table to (B, max_block_num) — kernel declares
        # block_table shape as (B, max_block_num), but generate_inputs
        # creates it as (B, per_batch_max_blocks). Pad with -1 (unused).
        if block_table is not None and block_table.shape[1] < max_block_num:
            padded = torch.full((B, max_block_num), -1, dtype=torch.int32)
            padded[:, :block_table.shape[1]] = block_table
            block_table = padded
    else:
        raise NotImplementedError(f"layout_key={layout_key} not supported")

    MAX_S2 = S2
    input_dtype_str = "float16" if qk_dtype == torch.float16 else "bfloat16"
    calc_dtype_str = "float"

    # Weights enter as input_dtype (fp16/bf16); kernel casts to float32 internally
    # via T.tile.cast (attempt 2 cast fix, matching AscendC arch22 + li_0720.py).
    weights_input = weights.reshape(B, S1, N2, G).npu()

    # Handle actual_seq
    if actual_seq_q is None:
        actual_seq_q = torch.tensor([S1] * B, dtype=torch.int32)
    if actual_seq_k is None:
        actual_seq_k = torch.tensor([S2] * B, dtype=torch.int32)
    if block_table is None:
        block_table = torch.zeros((B, 1), dtype=torch.int32)

    # Compile kernel
    kernel = lightning_indexer(
        B, S1, S2, N1, D, G, sparse_count, MAX_S2,
        layout_query, layout_key, block_size, max_block_num,
        sparse_mode, return_value,
        input_dtype=input_dtype_str, calc_dtype=calc_dtype_str,
    )

    # Call kernel
    output, output_value = kernel(
        query.npu(), key.npu(), weights_input,
        actual_seq_q.npu(), actual_seq_k.npu(), block_table.npu(),
    )
    torch.npu.synchronize()

    if return_value:
        # output_value is already input_dtype (kernel casts via T.tile.cast)
        return output, output_value
    return output, None


def build_test_params(case_name):
    """Build test parameters for a given L0 case."""
    cases = {
        "l0_li_default_a2_bf16": {
            "batch_size": 18, "q_seq": 3, "k_seq": 3072, "q_t_size": 1, "k_t_size": 1,
            "q_head_num": 16, "k_head_num": 1, "head_dim": 128, "block_size": 128,
            "block_num": 469, "qk_dtype": torch.bfloat16, "weight_dtype": torch.bfloat16,
            "actual_seq_dtype": torch.int32,
            "act_seq_q": [3]*18, "act_seq_k": [3016]*17+[3072],
            "layout_query": "BSND", "layout_key": "PA_BSND",
            "sparse_count": 2048, "sparse_mode": 3,
            "query_datarange": [-1, 1], "key_datarange": [-1, 1],
            "weights_datarange": [-130, 130], "return_value": False,
        },
        "l0_li_default_a2_fp16": {
            "batch_size": 18, "q_seq": 3, "k_seq": 3072, "q_t_size": 1, "k_t_size": 1,
            "q_head_num": 16, "k_head_num": 1, "head_dim": 128, "block_size": 128,
            "block_num": 469, "qk_dtype": torch.float16, "weight_dtype": torch.float16,
            "actual_seq_dtype": torch.int32,
            "act_seq_q": [3]*18, "act_seq_k": [3016]*17+[3072],
            "layout_query": "BSND", "layout_key": "PA_BSND",
            "sparse_count": 2048, "sparse_mode": 3,
            "query_datarange": [-1, 1], "key_datarange": [-1, 1],
            "weights_datarange": [-130, 130], "return_value": False,
        },
        "l0_li_small_bf16": {
            "batch_size": 1, "q_seq": 1, "k_seq": 256, "q_t_size": 1, "k_t_size": 1,
            "q_head_num": 8, "k_head_num": 1, "head_dim": 128, "block_size": 128,
            "block_num": 2, "qk_dtype": torch.bfloat16, "weight_dtype": torch.bfloat16,
            "actual_seq_dtype": torch.int32,
            "act_seq_q": [1], "act_seq_k": [256],
            "layout_query": "BSND", "layout_key": "PA_BSND",
            "sparse_count": 128, "sparse_mode": 3,
            "query_datarange": [-1, 1], "key_datarange": [-1, 1],
            "weights_datarange": [-130, 130], "return_value": False,
        },
        "l0_li_small_fp16_nomask": {
            "batch_size": 1, "q_seq": 1, "k_seq": 256, "q_t_size": 1, "k_t_size": 1,
            "q_head_num": 8, "k_head_num": 1, "head_dim": 128, "block_size": 128,
            "block_num": 2, "qk_dtype": torch.float16, "weight_dtype": torch.float16,
            "actual_seq_dtype": torch.int32,
            "act_seq_q": [1], "act_seq_k": [256],
            "layout_query": "BSND", "layout_key": "PA_BSND",
            "sparse_count": 128, "sparse_mode": 0,
            "query_datarange": [-1, 1], "key_datarange": [-1, 1],
            "weights_datarange": [-130, 130], "return_value": False,
        },
        "l0_li_bsnd_bsnd": {
            "batch_size": 2, "q_seq": 4, "k_seq": 512, "q_t_size": 1, "k_t_size": 1,
            "q_head_num": 16, "k_head_num": 1, "head_dim": 128, "block_size": 128,
            "block_num": 4, "qk_dtype": torch.bfloat16, "weight_dtype": torch.bfloat16,
            "actual_seq_dtype": torch.int32,
            "act_seq_q": [4, 4], "act_seq_k": [512, 512],
            "layout_query": "BSND", "layout_key": "BSND",
            "sparse_count": 256, "sparse_mode": 3,
            "query_datarange": [-1, 1], "key_datarange": [-1, 1],
            "weights_datarange": [-130, 130], "return_value": False,
        },
        "l0_li_large_b_48": {
            "batch_size": 48, "q_seq": 4, "k_seq": 3072, "q_t_size": 1, "k_t_size": 1,
            "q_head_num": 24, "k_head_num": 1, "head_dim": 128, "block_size": 128,
            "block_num": 96, "qk_dtype": torch.float16, "weight_dtype": torch.float16,
            "actual_seq_dtype": torch.int32,
            "act_seq_q": [4]*48, "act_seq_k": [3072]*48,
            "layout_query": "BSND", "layout_key": "BSND",
            "sparse_count": 1774, "sparse_mode": 3,
            "query_datarange": [-1, 1], "key_datarange": [-1, 1],
            "weights_datarange": [-130, 130], "return_value": False,
        },
        "l0_li_n1_64": {
            "batch_size": 16, "q_seq": 5, "k_seq": 3072, "q_t_size": 1, "k_t_size": 1,
            "q_head_num": 64, "k_head_num": 1, "head_dim": 128, "block_size": 128,
            "block_num": 384, "qk_dtype": torch.float16, "weight_dtype": torch.float16,
            "actual_seq_dtype": torch.int32,
            "act_seq_q": [5]*16, "act_seq_k": [3072]*16,
            "layout_query": "BSND", "layout_key": "BSND",
            "sparse_count": 2048, "sparse_mode": 3,
            "query_datarange": [-1, 1], "key_datarange": [-1, 1],
            "weights_datarange": [-130, 130], "return_value": False,
        },
        "l0_li_sparse_count_153": {
            "batch_size": 2, "q_seq": 1, "k_seq": 2048, "q_t_size": 1, "k_t_size": 1,
            "q_head_num": 8, "k_head_num": 1, "head_dim": 128, "block_size": 128,
            "block_num": 32, "qk_dtype": torch.float16, "weight_dtype": torch.float16,
            "actual_seq_dtype": torch.int32,
            "act_seq_q": [1, 1], "act_seq_k": [2048, 2048],
            "layout_query": "BSND", "layout_key": "PA_BSND",
            "sparse_count": 153, "sparse_mode": 3,
            "query_datarange": [-1, 1], "key_datarange": [-1, 1],
            "weights_datarange": [-130, 130], "return_value": False,
        },
    }
    return cases.get(case_name)


def generate_inputs(params):
    """Generate random input tensors from params. Returns (query, key_npu, key_cpu_bnsd, weights, act_q, act_k, block_table)."""
    B = params["batch_size"]
    S1 = params["q_seq"]
    S2 = params["k_seq"]
    N1 = params["q_head_num"]
    N2 = params["k_head_num"]
    D = params["head_dim"]
    block_size = params["block_size"]
    block_num = params["block_num"]
    qk_dtype = params["qk_dtype"]
    weight_dtype = params["weight_dtype"]
    act_seq_dtype = params["actual_seq_dtype"]
    layout_query = params["layout_query"]
    layout_key = params["layout_key"]
    act_seq_q = params["act_seq_q"]
    act_seq_k = params["act_seq_k"]

    # Shapes
    query_shape = [B, S1, N1, D]
    weights_shape = [B, S1, N1]

    if layout_key == "PA_BSND":
        k_max_s2 = math.floor(max(act_seq_k))
        key_shape = [B, N2, k_max_s2, D]  # BNSD format for golden
    else:
        key_shape = [B, S2, N2, D]

    # Random data
    np.random.seed(42)
    query = torch.tensor(np.random.uniform(*params["query_datarange"], query_shape)).to(qk_dtype).npu()
    weights = torch.tensor(np.random.uniform(*params["weights_datarange"], weights_shape)).to(weight_dtype).npu()
    key_bnsd = torch.tensor(np.random.uniform(*params["key_datarange"], key_shape)).to(qk_dtype)

    actual_seq_q = torch.tensor(act_seq_q, dtype=act_seq_dtype).npu()
    actual_seq_k = torch.tensor(act_seq_k, dtype=act_seq_dtype).npu()

    block_table = None
    key_npu = key_bnsd.npu()

    if layout_key == "PA_BSND":
        k_max_s2 = math.floor(max(act_seq_k))
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size)
        block_table = torch.zeros((B, k_max_block_num_per_batch), dtype=torch.int32)
        cur_block_id = 0
        for b_idx in range(B):
            cur_act_k = math.floor(act_seq_k[b_idx])
            cur_key_block_num = math.ceil(cur_act_k / block_size)
            for i_block in range(cur_key_block_num):
                block_table[b_idx, i_block] = cur_block_id
                cur_block_id += 1

        # Reorganize BNSD key into PA_BSND [block_num, block_size, N2, D]
        key_new = torch.zeros((block_num, block_size, N2, D), dtype=qk_dtype)
        cur_block_id = 0
        for b_idx in range(B):
            cur_act_k = math.floor(act_seq_k[b_idx])
            cur_key_block_num = math.ceil(cur_act_k / block_size)
            for i_block in range(cur_key_block_num):
                start = i_block * block_size
                end = min(start + block_size, cur_act_k)
                key_new[block_table[b_idx, i_block].item(), :end-start, :, :] = \
                    key_bnsd[b_idx, :, start:end, :].permute(1, 0, 2)  # [N2, block_size, D] -> [block_size, N2, D]
        key_npu = key_new.npu()
        block_table = block_table.npu()

    # key_bnsd stays on CPU for golden, key_npu is the kernel input
    return query, key_npu, key_bnsd, weights, actual_seq_q, actual_seq_k, block_table


def compute_golden(params, query, key, weights, actual_seq_q, actual_seq_k, block_table):
    """Compute golden using GeneralizedLI."""
    B = params["batch_size"]
    S1 = params["q_seq"]
    S2 = params["k_seq"]
    N1 = params["q_head_num"]
    N2 = params["k_head_num"]
    D = params["head_dim"]
    block_size = params["block_size"]
    block_num = params["block_num"]
    qk_dtype = params["qk_dtype"]
    weight_dtype = params["weight_dtype"]
    act_seq_dtype = params["actual_seq_dtype"]
    layout_query = params["layout_query"]
    layout_key = params["layout_key"]
    sparse_count = params["sparse_count"]
    sparse_mode = params["sparse_mode"]

    test_li = GeneralizedLI(
        B, S1, S2, 1, 1, N1, N2, D, block_size, block_num,
        qk_dtype, weight_dtype, act_seq_dtype,
        params["act_seq_q"], params["act_seq_k"],
        layout_query, layout_key, sparse_count, sparse_mode,
    )

    cpu_result, topk_value = test_li.forward(
        query, key.cpu(), weights, actual_seq_q, actual_seq_k, block_table,
    )
    cpu_result, _ = torch.sort(cpu_result)
    return cpu_result, topk_value


def run_l0_case(case_name):
    """Run a single L0 test case."""
    params = build_test_params(case_name)
    if params is None:
        print(f"  [SKIP] {case_name}: not implemented yet")
        return None

    print(f"  [RUN] {case_name}: B={params['batch_size']}, S1={params['q_seq']}, "
          f"S2={params['k_seq']}, N1={params['q_head_num']}, "
          f"layout={params['layout_query']}_{params['layout_key']}, "
          f"dtype={params['qk_dtype']}, sparse_count={params['sparse_count']}, "
          f"sparse_mode={params['sparse_mode']}")

    try:
        # Generate inputs
        query, key_npu, key_bnsd, weights, actual_seq_q, actual_seq_k, block_table = generate_inputs(params)

        # Compute golden (pass BNSD format key on CPU)
        cpu_result, topk_value = compute_golden(
            params, query, key_bnsd, weights, actual_seq_q, actual_seq_k, block_table,
        )

        # Run kernel (pass PA_BSND or BSND format key on NPU)
        npu_result, sparse_value = lightning_indexer_wrapper(
            query, key_npu, weights, actual_seq_q, actual_seq_k, block_table,
            params["layout_query"], params["layout_key"],
            params["sparse_count"], params["sparse_mode"], params["return_value"],
            params["qk_dtype"], params["weight_dtype"],
        )
        npu_result, _ = torch.sort(npu_result)

        # Compare using result_compare_method
        params_list = [
            params["batch_size"], params["q_seq"], params["k_seq"],
            params.get("q_t_size", 1), params.get("k_t_size", 1),
            params["q_head_num"], params["k_head_num"], params["head_dim"],
            params["block_size"], params["block_num"],
            params["qk_dtype"], params["weight_dtype"], params["actual_seq_dtype"],
            params["act_seq_q"], params["act_seq_k"],
            params["layout_query"], params["layout_key"],
            params["sparse_count"], params["sparse_mode"],
            params["query_datarange"], params["key_datarange"],
            params["weights_datarange"], params["return_value"],
        ]

        result, fulfill_percent = check_result(
            cpu_result, npu_result, topk_value, sparse_value, params_list,
        )

        if result == "Pass":
            print(f"  [PRECISION_PASS] {case_name}: {fulfill_percent:.2f}%")
            return True
        else:
            print(f"  [PRECISION_FAIL] {case_name}: {fulfill_percent:.2f}%")
            return False
    except Exception as e:
        print(f"  [ERROR] {case_name}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="lightning_indexer test")
    parser.add_argument("--level", type=str, default="L0",
                        choices=["L0", "L1", "L2", "Boundary", "all"])
    args, _ = parser.parse_known_args()

    tilelang.disable_cache()

    # L0 test cases
    l0_cases = [
        "l0_li_small_bf16",
        "l0_li_small_fp16_nomask",
        "l0_li_bsnd_bsnd",
        "l0_li_default_a2_bf16",
        "l0_li_default_a2_fp16",
        "l0_li_sparse_count_153",
        "l0_li_large_b_48",
        "l0_li_n1_64",
    ]

    if args.level in ["L0", "all"]:
        print("=" * 60)
        print("L0 Test Suite")
        print("=" * 60)
        passed = 0
        failed = 0
        for case_name in l0_cases:
            result = run_l0_case(case_name)
            if result is True:
                passed += 1
            elif result is False:
                failed += 1
        print(f"\nL0 Summary: {passed} passed, {failed} failed, "
              f"{len(l0_cases) - passed - failed} skipped")

        if failed == 0 and passed > 0:
            print("Test Passed!")
            print("Kernel Output Match!")
        else:
            print("Test Failed!")
            sys.exit(1)


if __name__ == "__main__":
    main()
