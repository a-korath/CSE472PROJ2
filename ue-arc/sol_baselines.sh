#!/bin/bash
# Sol booking 1: zero-shot + clean LoRA fine-tune x 3 seeds, for both models.
# Resumable: any run whose result file exists is skipped, so a cut-off slot loses nothing.
# CHECK the Sol-specific lines (partition, QoS, GPU flag, module name) against Sol docs before submitting.
#SBATCH -J arc-baselines
#SBATCH -A class_cse47267488fall2026   # CSE 472 course allocation
#SBATCH -p general
#SBATCH -q class                       # class account default QoS (verified in the OnDemand form)
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00
#SBATCH -o logs/%x-%j.out
set -eo pipefail   # -u is turned on after env activation: conda/mamba activate scripts read unset vars

# Sol: module env + scratch cache. Anywhere else (RunPod, Colab): `HF_HOME=... bash -e` in an env with requirements.txt
if command -v module >/dev/null; then module load mamba/latest; source activate ue || source activate /scratch/$USER/envs/ue; fi   # check: module avail mamba
set -u
mkdir -p results adapters logs
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false   # compute nodes: read the pre-downloaded cache only
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}   # weights pre-downloaded here from the login node

LLAMA=${LLAMA:-meta-llama/Meta-Llama-3-8B}   # gated; fallback: LLAMA=NousResearch/Meta-Llama-3-8B sbatch ...
for M in "$LLAMA" Qwen/Qwen3-8B; do
  T=$(basename "$M")
  [ -f results/${T}_zeroshot.json ] || \
    python eval.py --model "$M" --data data/test.jsonl --out results/${T}_zeroshot.json
  for S in 1 2 3; do
    [ -f results/${T}_clean_s${S}.json ] || \
      python train_lora.py --model "$M" --train data/train.jsonl --seed $S \
        --out adapters/${T}_clean_s${S} --eval_test data/test.jsonl \
        --result results/${T}_clean_s${S}.json
  done
done

# ~5 min smoke test of protect.py on the real surrogate (it has only ever run on Qwen3-0.6B), so booking 2 isn't the first try
[ -f protected_smoke/final.jsonl ] || \
  python protect.py --model "$LLAMA" --limit 16 --rounds 1 --train_epochs 1 --stop_acc 2 --out protected_smoke
