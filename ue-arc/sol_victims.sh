#!/bin/bash
# Sol booking 3: victim fine-tunes on Random-edit and Protected data, both models x 3 seeds (12 runs).
# Resumable: any run whose result file exists is skipped.
# PROT picks which protected set the victims train on (default: last round). E.g. PROT=protected/r0.jsonl sbatch ...
#SBATCH -J arc-victims
#SBATCH -p general
#SBATCH -q public
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-05:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Sol: module env + scratch cache. Anywhere else (RunPod, Colab): `HF_HOME=... bash -e` in an env with requirements.txt
if command -v module >/dev/null; then module load mamba/latest; source activate ue; fi   # check: module avail mamba
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}   # weights pre-downloaded here from the login node
LLAMA=${LLAMA:-meta-llama/Meta-Llama-3-8B}
PROT=${PROT:-protected/final.jsonl}

for M in "$LLAMA" Qwen/Qwen3-8B; do
  T=$(basename "$M")
  for D in random:data/random_k2.jsonl protected:$PROT; do
    NAME=${D%%:*}; DATA=${D#*:}
    for S in 1 2 3; do
      [ -f results/${T}_${NAME}_s${S}.json ] || \
        python train_lora.py --model "$M" --train "$DATA" --seed $S \
          --out adapters/${T}_${NAME}_s${S} --eval_test data/test.jsonl \
          --result results/${T}_${NAME}_s${S}.json
    done
  done
done
