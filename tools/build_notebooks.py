"""Generates the Colab notebooks. Each notebook embeds revllm.py via %%writefile so it
runs standalone (no repo clone needed). Re-run after editing revllm.py:
    python tools/build_notebooks.py
"""
import os
import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = open(os.path.join(ROOT, "revllm.py")).read()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = [
    md("## 0. Setup\nRuntime → Change runtime type → **GPU** (T4 is enough; L4/A100 are faster). "
       "Results and the tokenized data are cached on Google Drive (`USE_DRIVE=True`) so the "
       "notebooks can share them; set it to `False` to keep everything in `/content`."),
    code("""import os, sys
USE_DRIVE = os.environ.get("REVLLM_USE_DRIVE", "1") == "1"   # persist data + results on Google Drive
SMOKE = os.environ.get("REVLLM_SMOKE", "0") == "1"   # True = tiny end-to-end pipeline check (~2 min)

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB and USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK = "/content/drive/MyDrive/revllm-50m"
else:
    WORK = os.path.abspath("work")
os.makedirs(WORK, exist_ok=True)
DATA_DIR, RES_DIR = f"{WORK}/data", f"{WORK}/results"
print("work dir:", WORK)
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!pip -q install tokenizers datasets"""),
    code("%%writefile revllm.py\n" + SRC),
    code("""import importlib, json, math, torch
import revllm as R; importlib.reload(R)
import matplotlib.pyplot as plt
print(torch.__version__, R.gpu_info())

TOTAL_TOKENS = 50_000_000 if not SMOKE else 400_000
DATA_SOURCE = os.environ.get("REVLLM_SOURCE", "tinystories")   # or a local .txt for offline tests
data = R.prepare_data(DATA_DIR, source=DATA_SOURCE, vocab_size=8192,
                      n_train_tokens=51_000_000 if not SMOKE else 1_000_000,
                      n_val_tokens=1_000_000 if not SMOKE else 100_000)
BASE_KW = dict(vocab_size=data["tokenizer"].get_vocab_size())
if SMOKE:                          # tiny model so the smoke test also runs on CPU
    BASE_KW.update(d_model=128, n_layer=4, n_head=4, seq_len=128)
M = lambda **kw: R.ModelConfig(**{**BASE_KW, **kw})   # d=384, 10 layers, 6 heads, T=512, vocab 8192
print(M())
print("params:", f"{R.GPT(M()).n_params()/1e6:.2f}M",
      "| non-embedding:", f"{R.GPT(M()).n_params(non_embedding=True)/1e6:.2f}M")
HAS_GPU = torch.cuda.is_available()"""),
]

PLOT = """def plot_runs(runs, title=""):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for r in runs:
        h = r["history"]
        ax[0].plot([t/1e6 for t in h["tokens"]], h["loss"], label=r["run_name"], alpha=.8)
        ax[1].plot([t/1e6 for t in h["val_tokens"]], h["val_loss"], "o-", label=r["run_name"])
    for a, t in zip(ax, ["train loss", "val loss"]):
        a.set_xlabel("tokens (M)"); a.set_ylabel(t); a.grid(alpha=.3); a.legend()
    ax[0].set_ylim(top=min(6, ax[0].get_ylim()[1]))
    fig.suptitle(title); plt.tight_layout(); plt.show()

def show(r):
    keys = ["run_name", "steps", "tokens_trained", "final_train_loss", "final_val_loss",
            "tokens_per_sec", "peak_mem_alloc_gb", "peak_mem_reserved_gb", "activation_mem_gb", "train_time_s"]
    for k in keys:
        v = r.get(k)
        print(f"  {k:22s} {v:,.4f}" if isinstance(v, float) else f"  {k:22s} {v}")
    print("  sample:", r["sample"][:400])"""


def nb(cells, name):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "language_info": {"name": "python"}}
    nbf.write(n, os.path.join(ROOT, name))
    print("wrote", name)


# ---------------------------------------------------------------- 01 baseline
nb([md("""# 01 · Baseline: 20M GPT, 50M tokens, fixed batch size
Standard pre-LN transformer, ordinary backprop (all activations stored).

* Model: 21.0M params (d=384, 10 layers, 6 heads, ctx 512, byte-level BPE vocab 8192, tied head)
* Data: TinyStories, 50M training tokens (every token seen once)
* Fixed batch: **32 × 512 = 16,384 tokens/step → 3,052 steps**
* AdamW (0.9, 0.95), peak LR 1e-3, 5% warmup, cosine to 10%, wd 0.1, clip 1.0, bf16/fp16 autocast

We also probe the **largest batch the baseline can fit**, to compare with the reversible model later.""")]
   + SETUP + [
    md("## 1. How large a batch fits without reversibility? (reference)"),
    code("""if HAS_GPU:
    probe_std = R.probe_max_batch(M(mode="standard"), limit=4096 if not SMOKE else 64)
    R.save_json(probe_std, f"{RES_DIR}/probe_standard.json")"""),
    md("## 2. Train the baseline at a fixed batch size of 32"),
    code(PLOT),
    code("""BATCH = 32
base = R.train(M(mode="standard"),
               R.TrainConfig(run_name="01_baseline_b32", batch_size=BATCH, total_tokens=TOTAL_TOKENS,
                             lr=1e-3, out_dir=RES_DIR, log_every=50 if not SMOKE else 5),
               data)
show(base); plot_runs([base], "baseline")"""),
], "01_baseline.ipynb")

# ---------------------------------------------------------------- 02 reversible
nb([md("""# 02 · Reversible training: which integrator works?
A residual stack is an ODE integrator. If each block's *input* can be recomputed from its *output*,
activations need not be stored: backward rebuilds them layer by layer (custom `autograd.Function`
in `revllm.py`). We try three schemes, all with the same blocks/params (21.0M):

| variant | update | inverse | exact? |
|---|---|---|---|
| **euler** | $x_{n+1}=x_n+F_n(x_n)$ (= the baseline net) | fixed-point $x\\leftarrow x_{n+1}-F_n(x)$ | only if Lip$(F_n)<1$ |
| **midpoint** (leapfrog) | $x_{n+1}=x_{n-1}+F_n(x_n)$ | $x_{n-1}=x_{n+1}-F_n(x_n)$ | yes |
| **revnet** (coupling / symplectic Euler) | $y_1=x_1+\\mathrm{Attn}(x_2),\\; y_2=x_2+\\mathrm{MLP}(y_1)$ | $x_2=y_2-\\mathrm{MLP}(y_1),\\; x_1=y_1-\\mathrm{Attn}(x_2)$ | yes |

Steps: (1) gradient check vs. plain autograd, (2) 5M-token bake-off at batch 32,
(3) full 50M-token run of the winner at the **same batch size as the baseline (32)**.""")]
   + SETUP + [
    md("## 1. Gradient check (float64, small model) — are the reversible gradients right?"),
    code("""import pandas as pd
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = [R.grad_check(m, device=dev) for m in ["revnet", "midpoint"]]
for e in [0.02, 0.1, 0.5]:                      # Euler: inverse only converges if the residual
    for k in [1, 4, 8]:                          # stream is large relative to F's Lipschitz const.
        rows.append(R.grad_check("euler", device=dev, euler_iters=k, emb_std=e))
gc = pd.DataFrame(rows); R.save_json(rows, f"{RES_DIR}/grad_check.json"); gc"""),
    md("## 2. Bake-off: 5M tokens each, batch 32"),
    code(PLOT),
    code("""BATCH, BAKE_TOKENS = 32, (5_000_000 if not SMOKE else 200_000)
variants = {
    "standard (ref)":        M(mode="standard"),
    "euler (4 it)":          M(mode="euler", euler_iters=4),
    "euler (4 it, emb .5)":  M(mode="euler", euler_iters=4, emb_std=0.5),
    "midpoint":              M(mode="midpoint"),
    "revnet":                M(mode="revnet"),
}
bake = {}
for name, mc in variants.items():
    tag = "02_bake_" + name.split()[0] + ("_emb05" if "emb" in name else "")
    bake[name] = R.train(mc, R.TrainConfig(run_name=tag, batch_size=BATCH, total_tokens=BAKE_TOKENS,
                          lr=1e-3, eval_every_tokens=10**9, out_dir=RES_DIR,
                          log_every=50 if not SMOKE else 5), data)
tbl = pd.DataFrame([{ "variant": k, "val_loss": r["final_val_loss"], "train_loss": r["final_train_loss"],
    "tok/s": round(r["tokens_per_sec"]), "peak GB": r["peak_mem_alloc_gb"],
    "act GB": r["activation_mem_gb"], "diverged": r["diverged"],
    "euler residual": r["euler_recon_residual"]} for k, r in bake.items()])
plot_runs(list(bake.values()), "5M-token bake-off"); tbl"""),
    code("""# pick the reversible variant with the lowest validation loss
rev = {k: r for k, r in bake.items() if not k.startswith("standard") and not r["diverged"]}
best_name = min(rev, key=lambda k: rev[k]["final_val_loss"])
BEST = variants[best_name]
print("chosen variant:", best_name, BEST)
R.save_json({"name": best_name, "model": R.asdict(BEST), "bakeoff": tbl.to_dict("records")},
            f"{RES_DIR}/chosen_variant.json")"""),
    md("## 3. Full run: chosen reversible variant, 50M tokens, batch 32 (same as baseline)"),
    code("""rev_run = R.train(BEST, R.TrainConfig(run_name="02_reversible_b32", batch_size=BATCH,
                  total_tokens=TOTAL_TOKENS, lr=1e-3, out_dir=RES_DIR,
                  log_every=50 if not SMOKE else 5), data)
show(rev_run)
runs = [rev_run]
if os.path.exists(f"{RES_DIR}/01_baseline_b32.json"):
    runs.insert(0, json.load(open(f"{RES_DIR}/01_baseline_b32.json")))
plot_runs(runs, "baseline vs reversible @ batch 32")"""),
], "02_reversible.ipynb")

# ---------------------------------------------------------------- 03 max batch
nb([md("""# 03 · Reversible training at the maximum batch size
Same reversible model as notebook 02, but the batch is pushed to the largest size that fits
in GPU memory (found by a doubling + bisection probe of a full fwd/bwd/AdamW step).
The token budget stays at 50M, so a bigger batch means **fewer optimizer steps**; peak LR is
scaled by √(B/32) (capped at 3e-3) and warmup is 5% of steps (min 10).""")]
   + SETUP + [
    code(PLOT),
    code("""p = f"{RES_DIR}/chosen_variant.json"
BEST = R.ModelConfig(**json.load(open(p))["model"]) if os.path.exists(p) else M(mode="revnet")
print("variant:", BEST.mode)
probe_rev = R.probe_max_batch(BEST, limit=4096 if not SMOKE else 64) if HAS_GPU else \
    dict(mode=BEST.mode, max_batch=64, max_tokens_per_step=64 * BEST.seq_len, hit_limit=True,
         gb_per_sequence=None, trace=[], hw=R.gpu_info())      # CPU smoke-test stand-in
R.save_json(probe_rev, f"{RES_DIR}/probe_{BEST.mode}.json")
if os.path.exists(f"{RES_DIR}/probe_standard.json"):
    ps = json.load(open(f"{RES_DIR}/probe_standard.json"))
    print(f"max batch  standard: {ps['max_batch']}   reversible ({BEST.mode}): {probe_rev['max_batch']}"
          f"   -> {probe_rev['max_batch']/ps['max_batch']:.1f}x")"""),
    code("""B_MAX = probe_rev["max_batch"]
lr = min(3e-3, 1e-3 * math.sqrt(B_MAX / 32))
while True:                                   # back off in steps of 8 if the real run OOMs
    try:
        big = R.train(BEST, R.TrainConfig(run_name="03_reversible_bmax", batch_size=B_MAX,
                      total_tokens=TOTAL_TOKENS, lr=lr, eval_every_tokens=5_000_000,
                      out_dir=RES_DIR, log_every=5), data)
        break
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); B_MAX -= 8; print("OOM -> retry with B =", B_MAX)
show(big)
runs = [json.load(open(f"{RES_DIR}/{n}.json")) for n in ["01_baseline_b32", "02_reversible_b32"]
        if os.path.exists(f"{RES_DIR}/{n}.json")] + [big]
plot_runs(runs, "all runs")"""),
], "03_reversible_maxbatch.ipynb")

# ---------------------------------------------------------------- 03b LR ablation
ABL_TRAIN = """p = f"{RES_DIR}/chosen_variant.json"
BEST = R.ModelConfig(**json.load(open(p))["model"]) if os.path.exists(p) else M(mode="revnet")
import gc
# one granularity step (8 seq) below the probed max: at exactly 944 this run once hit an OOM at
# step 100/104 from allocator fragmentation; 936 keeps tokens/step within 1% of run 03.
B_MAX = json.load(open(f"{RES_DIR}/probe_{BEST.mode}.json"))["max_batch"] - 8
print("variant:", BEST.mode, "| batch:", B_MAX)
while True:                                   # back off in steps of 8 if the run still OOMs
    oom = False
    try:
        abl = R.train(BEST, R.TrainConfig(run_name="03b_reversible_bmax_lr1e-3", batch_size=B_MAX,
                      total_tokens=TOTAL_TOKENS, lr=1e-3, eval_every_tokens=5_000_000,
                      out_dir=RES_DIR, log_every=5), data)
    except torch.cuda.OutOfMemoryError:
        oom = True
    if not oom:
        break
    gc.collect(); torch.cuda.empty_cache(); B_MAX -= 8; print("OOM -> retry with B =", B_MAX)
show(abl)
runs = [json.load(open(f"{RES_DIR}/{n}.json")) for n in ["01_baseline_b32", "02_reversible_b32", "03_reversible_bmax"]
        if os.path.exists(f"{RES_DIR}/{n}.json")] + [abl]
plot_runs(runs, "all runs + LR ablation")"""
nb([md("""# 03b · Max-batch LR ablation
Run 03 (max batch, peak LR 1e-3·√(B/32) → capped 3e-3) stalled near the unigram plateau.
This run keeps the **same reversible model and (to within one 8-sequence step) the same max batch** but uses the batch-32 peak
LR of **1e-3**, to separate "too few optimizer steps" from "LR too high".""")]
   + SETUP + [code(PLOT), code(ABL_TRAIN)], "03b_maxbatch_lr_ablation.ipynb")

# ---------------------------------------------------------------- 04 report
nb([md("""# 04 · Report
Collects every `results/*.json`, prints the summary table, and saves figures + `summary.md`
into the results folder (copy that folder into the repo's `results/`).""")]
   + [SETUP[0], SETUP[1].copy()] + [
    code("""import json, glob, os, pandas as pd, matplotlib.pyplot as plt
os.makedirs(f"{RES_DIR}/figs", exist_ok=True)
load = lambda n: json.load(open(f"{RES_DIR}/{n}.json")) if os.path.exists(f"{RES_DIR}/{n}.json") else None
rnd = lambda v, n: None if v is None else round(v, n)
main = [r for r in (load("01_baseline_b32"), load("02_reversible_b32"), load("03_reversible_bmax"), load("03b_reversible_bmax_lr1e-3")) if r]
rows = [{"run": r["run_name"], "mode": r["model"]["mode"], "batch": r["train"]["batch_size"],
         "tokens/step": r["tokens_per_step"], "steps": r["steps"], "peak LR": r["train"]["lr"],
         "final train loss": rnd(r["final_train_loss"], 4), "final val loss": rnd(r["final_val_loss"], 4),
         "tokens/s": round(r["tokens_per_sec"]), "peak mem alloc (GB)": rnd(r["peak_mem_alloc_gb"], 2),
         "peak mem reserved (GB)": rnd(r["peak_mem_reserved_gb"], 2),
         "activation mem (GB)": rnd(r["activation_mem_gb"], 2), "train time (min)": rnd(r["train_time_s"]/60, 1),
         "GPU": r["hw"]["device"]} for r in main]
df = pd.DataFrame(rows); df"""),
    code("""fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for r in main:
    h = r["history"]
    ax[0].plot([t/1e6 for t in h["tokens"]], h["loss"], label=r["run_name"], alpha=.8)
    ax[1].plot([t/1e6 for t in h["val_tokens"]], h["val_loss"], "o-", label=r["run_name"])
ax[0].set_ylim(1, 7)
for a, t in zip(ax[:2], ["train loss", "val loss"]):
    a.set_xlabel("tokens (M)"); a.set_ylabel(t); a.grid(alpha=.3); a.legend()
for f in sorted(glob.glob(f"{RES_DIR}/probe_*.json")):
    p = json.load(open(f)); ok = sorted([t for t in p["trace"] if t["ok"]], key=lambda t: t["batch"])
    ax[2].plot([t["batch"] for t in ok], [t["peak_gb"] for t in ok], "o-", label=f"{p['mode']} (max {p['max_batch']})")
ax[2].set_xscale("log", base=2); ax[2].set_xlabel("batch (sequences of 512)"); ax[2].set_ylabel("peak GB")
ax[2].grid(alpha=.3); ax[2].legend(); ax[2].set_title("memory vs batch")
plt.tight_layout(); plt.savefig(f"{RES_DIR}/figs/curves.png", dpi=130); plt.show()"""),
    code("""lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
extra = []
for n in ["probe_standard"] + [os.path.basename(f)[:-5] for f in glob.glob(f"{RES_DIR}/probe_*.json") if "standard" not in f]:
    p = load(n)
    if p: extra.append(f"- {n}: max batch **{p['max_batch']}** ({p['max_tokens_per_step']:,} tok/step), "
                       f"{(p['gb_per_sequence'] or 0)*1024:.1f} MB/sequence, hit probe limit: {p['hit_limit']}")
cv = load("chosen_variant")
if cv:
    extra.append(f"- chosen reversible variant: **{cv['name']}**")
    extra += ["  - bake-off " + json.dumps(b) for b in cv["bakeoff"]]
gc = load("grad_check")
if gc:
    extra += [f"- grad check {g['mode']} it={g['euler_iters']} emb={g['emb_std']}: rel_err={g['rel_err']:.2e}" for g in gc]
samples = [f"**{r['run_name']}**: {r['sample'][:300]!r}" for r in main]
md_txt = "\\n".join(lines + [""] + extra + ["", "### Samples", ""] + samples)
open(f"{RES_DIR}/summary.md", "w").write(md_txt); print(md_txt)"""),
    code("""# optional: zip the results folder for download
import shutil; shutil.make_archive(f"{WORK}/results", "zip", RES_DIR)
if IN_COLAB:
    from google.colab import files; files.download(f"{WORK}/results.zip")"""),
], "04_report.ipynb")
