"""Quick test for the lightning_indexer kernel."""
import sys
sys.path.insert(0, 'examples/lightning_indexer')

import torch
import tilelang
import numpy as np

tilelang.disable_cache()
from lightning_indexer import lightning_indexer

# L0 case 3: small BSND_PA_BSND, bf16
B, S1, S2, N1, D = 1, 1, 256, 8, 128
G = N1
SPARSE_COUNT = 128
MAX_S2 = 256
block_size = 128
max_block_num = 2
sparse_mode = 3
return_value = False

print(f'Test: B={B}, S1={S1}, S2={S2}, N1={N1}, D={D}, G={G}')
print(f'  SPARSE_COUNT={SPARSE_COUNT}, sparse_mode={sparse_mode}')
print('Compiling...')
kernel = lightning_indexer(
    B, S1, S2, N1, D, G, SPARSE_COUNT, MAX_S2,
    'BSND', 'BSND', block_size, max_block_num,
    sparse_mode, return_value,
    input_dtype='bfloat16', calc_dtype='float',
)
print('Compilation SUCCESS')

# Generate test data
np.random.seed(42)
query = torch.tensor(np.random.uniform(-1, 1, (B, S1, N1, D))).to(torch.bfloat16).npu()
key = torch.tensor(np.random.uniform(-1, 1, (B, S2, 1, D))).to(torch.bfloat16).npu()
weights = torch.tensor(np.random.uniform(-1, 1, (B, S1, 1, G))).float().npu()
act_seq_q = torch.tensor([1], dtype=torch.int32).npu()
act_seq_k = torch.tensor([256], dtype=torch.int32).npu()
block_table = torch.tensor([[0, 1]], dtype=torch.int32).npu()

print('Running...')
output, output_value = kernel(query, key, weights, act_seq_q, act_seq_k, block_table)
torch.npu.synchronize()
print('Output shape:', output.shape, 'dtype:', output.dtype)

# Compute golden
q_f = query.cpu().float()  # [B, S1, N1, D]
k_f = key.cpu().float()    # [B, S2, 1, D]
w_f = weights.cpu().float().squeeze(2)  # [B, S1, G]

# QK: [B, S1, N1, S2] = Q[B,S1,N1,D] @ K[B,S2,1,D]^T
k_squeeze = k_f.squeeze(2)  # [B, S2, D]
qk = torch.einsum('bsnd,btd->bsnt', q_f, k_squeeze)  # [B, S1, N1, S2]
qk_relu = qk.clamp_min(0)
weighted = qk_relu * w_f.unsqueeze(-1)  # [B, S1, N1, S2] * [B, S1, N1, 1]
scores = weighted.sum(dim=2)  # [B, S1, S2]

# Apply mask (sparse_mode=3: rightDownCausal)
act_q_val = 1
act_k_val = 256
for s1_idx in range(act_q_val):
    cutoff = act_k_val - act_q_val + s1_idx + 1
    scores[:, s1_idx, cutoff:] = float('-inf')

# TopK
topk_vals, topk_indices = torch.sort(-scores, dim=-1, stable=True)
topk_indices = topk_indices[..., :SPARSE_COUNT]

# Compare (set matching)
npu_out = output.cpu()[0, 0, 0, :].numpy()
golden_out = topk_indices[0, 0, :].numpy()

npu_set = set(npu_out[npu_out >= 0])
golden_set = set(golden_out[golden_out >= 0])
intersection = npu_set & golden_set
ratio = len(intersection) / max(len(golden_set), 1)

print(f'NPU output (first 10): {npu_out[:10]}')
print(f'Golden output (first 10): {golden_out[:10]}')
print(f'Set match ratio: {ratio:.4f} ({len(intersection)}/{len(golden_set)})')
if ratio > 0.99:
    print('TEST PASSED')
else:
    print('TEST FAILED')
