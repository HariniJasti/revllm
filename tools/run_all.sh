#!/bin/bash
# Execute all notebooks headless (e.g. in a Colab/Kaggle terminal or `!bash tools/run_all.sh`).
# Outputs: executed/*.ipynb, work/results/*, progress.log.  REVLLM_SMOKE=1 for a quick check.
cd "$(dirname "$0")/.."
export REVLLM_LOG="$PWD/progress.log"
export REVLLM_USE_DRIVE="${REVLLM_USE_DRIVE:-0}"   # headless: keep data/results in ./work
mkdir -p executed
for nb in 01_baseline 02_reversible 03_reversible_maxbatch 03b_maxbatch_lr_ablation 04_report; do
  echo "=== $nb start $(date +%T)" | tee -a "$REVLLM_LOG"
  if jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
       --output-dir executed "$nb.ipynb" >> run.log 2>&1; then
    echo "=== $nb OK $(date +%T)" | tee -a "$REVLLM_LOG"
  else
    echo "=== $nb FAILED $(date +%T)" | tee -a "$REVLLM_LOG"; exit 1
  fi
done
python -c "import shutil,os; os.makedirs('bundle',exist_ok=True); [shutil.copytree(s,'bundle/'+s.replace('work/',''),dirs_exist_ok=True) for s in ('executed','work/results')]; shutil.copy('progress.log','bundle/'); shutil.make_archive('revllm_results','zip','bundle')"
echo "=== ALL DONE" | tee -a "$REVLLM_LOG"
