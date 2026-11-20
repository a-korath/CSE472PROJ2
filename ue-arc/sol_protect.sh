#!/bin/bash
# Sol booking 2: min-min protection with the Llama-3-8B surrogate.
# Resumable: protect.py restarts from the last saved round (r*.jsonl, ckpt_r*/), so just resubmit if the slot is cut.
# Same Sol-specific lines as sol_baselines.sh; check them there.
#SBATCH -J arc-protect
#SBATCH -A class_cse47267488fall2026   # CSE 472 course allocation
#SBATCH -p general
#SBATCH -q class                       # class account default QoS (verified in the OnDemand form)
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-12:00
#SBATCH -o logs/%x-%j.out
set -eo pipefail   # -u is turned on after env activation: conda/mamba activate scripts read unset vars

# Sol: module env + scratch cache. Anywhere else (RunPod, Colab): `HF_HOME=... bash -e` in an env with requirements.txt
# Env: put its bin/ first on PATH. (`source activate` is a no-op in non-interactive shells on Sol, so it can't be used here.)
export PATH=/scratch/$USER/envs/ue/bin:$PATH
set -u
python -c 'import torch, transformers, peft' || { echo "ERROR: env /scratch/$USER/envs/ue missing or broken; run setup_sol.sh"; exit 1; }
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false   # compute nodes: read the pre-downloaded cache only
export HF_HOME=${HF_HOME:-/scratch/$USER/hf}   # weights pre-downloaded here from the login node
LLAMA=${LLAMA:-meta-llama/Meta-Llama-3-8B}

# protected/neighbours.json is built on the laptop (protect.py --neighbours_only) and copied up; built here if missing
# --stop_acc 1.01 = never stop early: run all 5 rounds and save every one (an 8B model can hit 99% train acc by
# memorization, so the 99% rule could fire at round 1 regardless). --k 2 pinned so it provably matches random_k2.jsonl.
python protect.py --model "$LLAMA" --train data/train.jsonl --out protected --k 2 --stop_acc 1.01
