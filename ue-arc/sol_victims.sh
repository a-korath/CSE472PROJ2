#!/bin/bash
# Sol booking 3: victim fine-tunes on Random-edit and Protected data, both models x 3 seeds (12 runs).
# Resumable: any run whose result file exists is skipped.
# PROT picks which protected set the victims train on (default: last round); PNAME names its results so runs never collide.
# E.g. PROT=protected/r0.jsonl PNAME=protected_r0 sbatch sol_victims.sh   (hint bonus: PROT=data/hint.jsonl PNAME=hint)
#SBATCH -J arc-victims
#SBATCH -A class_cse47267488fall2026   # CSE 472 course allocation
#SBATCH -p general
#SBATCH -q class                       # class account default QoS (verified in the OnDemand form)
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-05:00
#SBATCH -o logs/%x-%j.out
set -eo pipefail   # -u is turned on after env activation: conda/mamba activate scripts read unset vars

# Sol: module env + scratch cache. Anywhere else (RunPod, Colab): `HF_HOME=... bash -e` in an env with requirements.txt
# Env: put its bin/ first on PATH. (`source activate` is a no-op in non-interactive shells on Sol, so it can't be used here.)
export PATH=/scratch/$USER/envs/ue/bin:$PATH
set -u
python -c 'import torch, transformers, peft' || { echo "ERROR: env /scratch/$USER/envs/ue missing or broken; run setup_sol.sh"; exit 1; }
mkdir -p results adapters logs
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false   # compute nodes: read the pre-downloaded cache only
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}   # weights pre-downloaded here from the login node
LLAMA=${LLAMA:-meta-llama/Meta-Llama-3-8B}
PROT=${PROT:-protected/final.jsonl}
PNAME=${PNAME:-protected}   # result-name for PROT; set it for any other set, e.g. PROT=protected/r0.jsonl PNAME=protected_r0

for M in "$LLAMA" Qwen/Qwen3-8B; do
  T=$(basename "$M")
  for D in random:data/random_k2.jsonl $PNAME:$PROT; do
    NAME=${D%%:*}; DATA=${D#*:}
    for S in 1 2 3; do
      [ -f results/${T}_${NAME}_s${S}.json ] || \
        python train_lora.py --model "$M" --train "$DATA" --seed $S \
          --out adapters/${T}_${NAME}_s${S} --eval_test data/test.jsonl \
          --result results/${T}_${NAME}_s${S}.json
    done
  done
done
