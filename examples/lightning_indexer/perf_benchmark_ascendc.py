#!/usr/bin/env python3
"""
lightning_indexer AscendC 算子性能基准采集脚本

用法:
  msprof op --kernel-name="Lightning" --application="python3 perf_benchmark_ascendc.py" \
    --output=./perf_output --aic-metrics=Default --launch-count=20 --warm-up=5

支持四种 Layout 场景，通过命令行参数选择:
  python3 perf_benchmark_ascendc.py --case=BSND_BSND
  python3 perf_benchmark_ascendc.py --case=BSND_PA_BSND
  python3 perf_benchmark_ascendc.py --case=TND_TND
  python3 perf_benchmark_ascendc.py --case=TND_PA_BSND
  python3 perf_benchmark_ascendc.py --case=ALL   # 跑全部四种
"""

import argparse
import math
import numpy as np
import torch
import torch_npu


def dtype_from_str(s):
    return torch.float16 if s == "FP16" else torch.bfloat16


def build_case_params(case_name):
    """四种 Layout 代表性用例参数"""
    cases = {
        "BSND_BSND": {
            # LI_L0_16_64_1_5_3072_128_BSND_BSND_FP16_000002
            "B": 16, "S1": 5, "S2": 3072, "N1": 64, "N2": 1, "D": 128,
            "block_size": 128, "dtype": "FP16",
            "layout_query": "BSND", "layout_key": "BSND",
            "sparse_count": 2048, "sparse_mode": 3,
            "act_seq_q": [5] * 16,
            "act_seq_k": [3072] * 16,
            "return_value": False,
        },
        "BSND_PA_BSND": {
            # LI_L0_pa_2_8_1_1_2048_128_BSND_PA_BSND_FP16_000008
            "B": 2, "S1": 1, "S2": 2048, "N1": 8, "N2": 1, "D": 128,
            "block_size": 128, "dtype": "FP16",
            "layout_query": "BSND", "layout_key": "PA_BSND",
            "sparse_count": 2048, "sparse_mode": 3,
            "act_seq_q": [1, 1],
            "act_seq_k": [2048, 2048],
            "return_value": False,
        },
        "TND_TND": {
            # LI_L0_8_24_1_5_3072_128_TND_FP16_000032
            "B": 8, "S1": 5, "S2": 3072, "N1": 24, "N2": 1, "D": 128,
            "block_size": 256, "dtype": "FP16",
            "layout_query": "TND", "layout_key": "TND",
            "sparse_count": 2048, "sparse_mode": 0,
            # TND 前缀和
            "act_seq_q": [5, 10, 15, 20, 25, 30, 35, 40],
            "act_seq_k": [3072, 6144, 9216, 12288, 15360, 18432, 21504, 24576],
            "return_value": False,
        },
        "TND_PA_BSND": {
            # LI_L0_pa_20_64_1_3_512_128_TND_PA_BSND_BF16_000021
            "B": 20, "S1": 3, "S2": 512, "N1": 64, "N2": 1, "D": 128,
            "block_size": 16, "dtype": "BF16",
            "layout_query": "TND", "layout_key": "PA_BSND",
            "sparse_count": 315, "sparse_mode": 0,
            # TND query 前缀和
            "act_seq_q": [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 60],
            # PA_BSND key 非前缀和
            "act_seq_k": [512] * 20,
            "return_value": False,
        },
    }
    return cases[case_name]


def build_inputs(params):
    """构造输入张量"""
    B = params["B"]
    S1 = params["S1"]
    S2 = params["S2"]
    N1 = params["N1"]
    N2 = params["N2"]
    D = params["D"]
    block_size = params["block_size"]
    dtype = dtype_from_str(params["dtype"])
    layout_query = params["layout_query"]
    layout_key = params["layout_key"]
    act_seq_q = params["act_seq_q"]
    act_seq_k = params["act_seq_k"]

    # Query
    if layout_query == "BSND":
        query = torch.randn(B, S1, N1, D, dtype=dtype, device="npu")
        weights = torch.randn(B, S1, N1, dtype=dtype, device="npu") * 0.1
    else:  # TND
        T1 = act_seq_q[-1] if layout_query == "TND" else sum(act_seq_q)
        query = torch.randn(T1, N1, D, dtype=dtype, device="npu")
        weights = torch.randn(T1, N1, dtype=dtype, device="npu") * 0.1

    # Key
    if layout_key == "BSND":
        key = torch.randn(B, S2, N2, D, dtype=dtype, device="npu")
        block_table = None
    elif layout_key == "TND":
        T2 = act_seq_k[-1] if layout_key == "TND" else sum(act_seq_k)
        key = torch.randn(T2, N2, D, dtype=dtype, device="npu")
        block_table = None
    elif layout_key == "PA_BSND":
        # 构建 PA key: [block_num, block_size, N2, D]
        k_max_s2 = max(act_seq_k)
        block_num = math.ceil(k_max_s2 / block_size) * B
        # 确保 block_num 足够
        total_blocks_needed = sum(math.ceil(ak / block_size) for ak in act_seq_k)
        block_num = max(block_num, total_blocks_needed + 10)
        key = torch.randn(block_num, block_size, N2, D, dtype=dtype, device="npu")

        # 构建 block_table: [B, maxBlockNumPerSeq]
        max_block_num_per_seq = math.ceil(k_max_s2 / block_size)
        block_table_np = np.full((B, max_block_num_per_seq), fill_value=-1, dtype=np.int32)
        block_id_list = np.arange(total_blocks_needed, dtype=np.int32)
        np.random.shuffle(block_id_list)
        cur_block_id = 0
        for b_idx in range(B):
            cur_act_k = act_seq_k[b_idx]
            cur_block_num = math.ceil(cur_act_k / block_size)
            for i_block in range(cur_block_num):
                block_table_np[b_idx][i_block] = block_id_list[cur_block_id]
                cur_block_id += 1
        block_table = torch.from_numpy(block_table_np).to(torch.int32).npu()
    else:
        raise ValueError(f"Unsupported layout_key: {layout_key}")

    # actual_seq_lengths
    actual_seq_q = torch.tensor(act_seq_q, dtype=torch.int32, device="npu")
    actual_seq_k = torch.tensor(act_seq_k, dtype=torch.int32, device="npu")

    return query, key, weights, actual_seq_q, actual_seq_k, block_table


def run_benchmark(params, num_iters=30):
    """运行性能基准"""
    print(f"\n{'='*80}")
    print(f"Case: {params['layout_query']}_{params['layout_key']}")
    print(f"  B={params['B']} S1={params['S1']} S2={params['S2']} N1={params['N1']} D={params['D']}")
    print(f"  block_size={params['block_size']} dtype={params['dtype']} "
          f"sparse_count={params['sparse_count']} sparse_mode={params['sparse_mode']}")
    print(f"  iterations={num_iters}")
    print(f"{'='*80}")

    # 构造输入
    query, key, weights, act_seq_q, act_seq_k, block_table = build_inputs(params)
    print(f"  query shape: {query.shape}, dtype: {query.dtype}")
    print(f"  key shape: {key.shape}, dtype: {key.dtype}")
    print(f"  weights shape: {weights.shape}, dtype: {weights.dtype}")
    if block_table is not None:
        print(f"  block_table shape: {block_table.shape}")

    # 调用参数
    kwargs = dict(
        actual_seq_lengths_query=act_seq_q,
        actual_seq_lengths_key=act_seq_k,
        layout_query=params["layout_query"],
        layout_key=params["layout_key"],
        sparse_count=params["sparse_count"],
        sparse_mode=params["sparse_mode"],
        return_value=params["return_value"],
    )
    if block_table is not None:
        kwargs["block_table"] = block_table

    # 预热 (5 次)
    print("  Warming up (5 iters)...")
    for _ in range(5):
        _ = torch_npu.npu_lightning_indexer(query, key, weights, **kwargs)
    torch.npu.synchronize()

    # 正式采集 (num_iters 次)
    print(f"  Benchmarking ({num_iters} iters)...")
    start_event = torch.npu.Event(enable_timing=True)
    end_event = torch.npu.Event(enable_timing=True)

    start_event.record()
    for i in range(num_iters):
        result = torch_npu.npu_lightning_indexer(query, key, weights, **kwargs)
    end_event.record()
    torch.npu.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / num_iters

    # 输出结果
    if isinstance(result, tuple):
        out_shape = result[0].shape
    else:
        out_shape = result.shape
    print(f"\n  Result:")
    print(f"    output shape: {out_shape}")
    print(f"    total time: {total_ms:.3f} ms")
    print(f"    avg per iter: {avg_ms:.3f} ms")
    print(f"    throughput: {1000.0/avg_ms:.1f} ops/s")

    return {
        "case": f"{params['layout_query']}_{params['layout_key']}",
        "B": params["B"], "S1": params["S1"], "S2": params["S2"], "N1": params["N1"],
        "avg_ms": avg_ms,
        "total_ms": total_ms,
        "iters": num_iters,
    }


def main():
    parser = argparse.ArgumentParser(description="lightning_indexer AscendC perf benchmark")
    parser.add_argument("--case", type=str, default="ALL",
                        choices=["ALL", "BSND_BSND", "BSND_PA_BSND", "TND_TND", "TND_PA_BSND"],
                        help="Which case to benchmark")
    parser.add_argument("--iters", type=int, default=30, help="Number of iterations")
    args = parser.parse_args()

    torch.npu.set_device(0)
    np.random.seed(42)
    torch.manual_seed(42)

    if args.case == "ALL":
        cases = ["BSND_BSND", "BSND_PA_BSND", "TND_TND", "TND_PA_BSND"]
    else:
        cases = [args.case]

    results = []
    for case_name in cases:
        params = build_case_params(case_name)
        result = run_benchmark(params, num_iters=args.iters)
        results.append(result)

    # 汇总
    print(f"\n{'='*80}")
    print("Benchmark Summary (AscendC torch_npu.npu_lightning_indexer)")
    print(f"{'='*80}")
    print(f"{'Case':<20} {'B':>4} {'S1':>6} {'S2':>6} {'N1':>4} {'Avg(ms)':>10} {'Ops/s':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['case']:<20} {r['B']:>4} {r['S1']:>6} {r['S2']:>6} {r['N1']:>4} "
              f"{r['avg_ms']:>10.3f} {1000.0/r['avg_ms']:>10.1f}")

    # 保存结果
    import json
    with open("/home/tilelang-ascend/examples/lightning_indexer/perf_baseline_ascendc.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: examples/lightning_indexer/perf_baseline_ascendc.json")


if __name__ == "__main__":
    main()
