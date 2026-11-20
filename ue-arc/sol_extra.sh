#!/bin/bash
#SBATCH -A class_cse47267488fall2026   # CSE 472 course allocation
#SBATCH -p public
#SBATCH -q class
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00
#SBATCH -J arc-extra
#SBATCH -o logs/%x-%j.out
# Follow-up runs, all resumable (finished result files are skipped):
#   MODELS  victims to run (default: Qwen/Qwen3-8B-Base, the headroom check for Qwen3-8B's ceiling)
#   SETS    which training sets: any of zeroshot clean random protected hint protected_r0
# e.g.  sbatch sol_extra.sh
#       MODELS="meta-llama/Meta-Llama-3-8B" SETS="protected_r0" sbatch sol_extra.sh
set -eo pipefail
export PATH=/scratch/$USER/envs/ue/bin:$PATH
set -u
python -c 'import torch, transformers, peft' || { echo "ERROR: env missing; run setup_sol.sh"; exit 1; }
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false HF_HOME=${HF_HOME:-/scratch/$USER/hf}
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p results adapters logs
[ -f data/hint.jsonl ] || python make_hint.py
MODELS=${MODELS:-Qwen/Qwen3-8B-Base}
SETS=${SETS:-zeroshot clean random protected hint}
declare -A FILE=([clean]=data/train.jsonl [random]=data/random_k2.jsonl [protected]=protected/final.jsonl
                 [hint]=data/hint.jsonl [protected_r0]=protected/r0.jsonl)
for M in $MODELS; do
  T=$(basename "$M")
  for D in $SETS; do
    if [ "$D" = zeroshot ]; then
      [ -f results/${T}_zeroshot.json ] || python eval.py --model "$M" --data data/test.jsonl --out results/${T}_zeroshot.json
      continue
    fi
    for S in 1 2 3; do
      [ -f results/${T}_${D}_s${S}.json ] || \
        python train_lora.py --model "$M" --train "${FILE[$D]}" --seed $S \
          --out adapters/${T}_${D}_s${S} --eval_test data/test.jsonl --result results/${T}_${D}_s${S}.json
    done
  done
done
echo "ALL EXTRA RUNS DONE"
