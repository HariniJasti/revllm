"""CPU-only correctness checks (no GPU needed). Output is saved in results/cpu_checks.txt.
    python tools/cpu_checks.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import revllm as R

torch.set_num_threads(max(1, os.cpu_count() or 1))

print("== 1. gradient check vs plain autograd (float64, d=128, 4 layers, T=64) ==")
for m in ["revnet", "midpoint"]:
    r = R.grad_check(m)
    print(f"  {m:9s} rel_err={r['rel_err']:.2e}")
for e in [0.02, 0.1, 0.5]:
    for k in [1, 4, 8, 16]:
        r = R.grad_check("euler", euler_iters=k, emb_std=e)
        print(f"  euler emb_std={e:<4} iters={k:<2} rel_err={r['rel_err']:.2e}  "
              f"fixed-point step={r['euler_fixed_point_residual']:.1e}")

print("\n== 2. full-size model (21.05M), fp32, chunked CE: reversible vs autograd ==")
torch.manual_seed(0)
for mode in ["revnet", "midpoint"]:
    m = R.GPT(R.ModelConfig(mode=mode, ce_chunk=256))
    idx, tgt = torch.randint(0, 8192, (2, 256)), torch.randint(0, 8192, (2, 256))
    g = {}
    for ref in (True, False):
        m.zero_grad(set_to_none=True)
        m.c.ce_chunk = 0 if ref else 256
        m(idx, tgt, reference=ref).backward()
        g[ref] = torch.cat([p.grad.flatten() for p in m.parameters()])
    print(f"  {mode:9s} params={m.n_params()/1e6:.2f}M rel_err={((g[False]-g[True]).norm()/g[True].norm()).item():.2e}")

print("\n== 3. activation memory saved for backward (fp32, T=512, ce_chunk=512) ==")
for mode in ["standard", "euler", "midpoint", "revnet"]:
    m = R.GPT(R.ModelConfig(mode=mode, ce_chunk=512))
    out = []
    for B in (2, 4):
        idx = torch.randint(0, 8192, (B, 512))
        out.append(R.saved_activation_bytes(m, idx, idx) / 2**20)
    per = (out[1] - out[0]) / 2
    print(f"  {mode:9s} {per:7.2f} MB / sequence  ({per*1024/512:6.1f} KB / token)")
