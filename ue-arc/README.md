# Unlearnable ARC-Challenge — baseline code

## Setup
```bash
pip install -r requirements.txt
python data.py                       # -> data/{train,validation,test}.jsonl
```

## Dry run (laptop CPU or Colab; tiny model, 16 examples) — do this BEFORE using Sol time
```bash
python eval.py --model Qwen/Qwen3-0.6B --limit 16 --out results/dry_zeroshot.json
python train_lora.py --model Qwen/Qwen3-0.6B --limit 16 --epochs 1 --bs 4 \
    --out adapters/dry --eval_test data/test.jsonl --result results/dry_clean.json
```
Pass = both commands finish and print an accuracy. The numbers themselves mean nothing at this size.

## Sol booking 1
1. On the login node: `huggingface-cli login`, then
   `python -c "import nltk; nltk.download('wordnet')"` (compute nodes may have no internet), then
   `HF_HOME=/scratch/$USER/hf huggingface-cli download meta-llama/Meta-Llama-3-8B` and the same for `Qwen/Qwen3-8B`.
2. Copy `data/` and `protected/neighbours.json` up, check the Sol-specific lines in `sol_baselines.sh`, then
   `mkdir -p logs && sbatch sol_baselines.sh` (`logs/` must exist at submit time or SLURM drops the job's output).
   No Llama access yet? `LLAMA=NousResearch/Meta-Llama-3-8B sbatch sol_baselines.sh` (same weights, ungated; disclose it). Same for the other two scripts.

## Laptop (CPU, no GPU needed), done before booking 2
```bash
python protect.py --model NousResearch/Meta-Llama-3-8B --out protected --neighbours_only   # embeddings only, ~5 GB download
python random_edits.py --tokenizer NousResearch/Meta-Llama-3-8B --neighbours protected/neighbours.json --out data/random_k2.jsonl
python metrics.py data/train.jsonl data/random_k2.jsonl
```

## Sol booking 2: protection
`sbatch sol_protect.sh` writes `protected/r{0..5}.jsonl`, one checkpoint per round, `rounds.jsonl` (the loss curve) and
`final.jsonl`. Resubmit if the slot is cut; it resumes at the last saved round.

## Sol booking 3: victims
`sbatch sol_victims.sh` (`PROT=protected/r0.jsonl` to train on a different round), then
`python summarize.py results` for the table and `python metrics.py data/train.jsonl protected/r*.jsonl data/random_k2.jsonl`
for the stealthiness numbers and the trade-off plot.

## RTX 5070 (12 GB): backup only
An 8B model in bf16 is ~16 GB of weights and doesn't fit. The only way to use this card is 4-bit QLoRA, which breaks the
"same bf16 settings" rule, so it must be disclosed. It also needs a CUDA 12.8 build of torch (Blackwell).

## Fixed settings (state these in the report)
- Prompt: `Question: ...\nA. ...\nB. ...\nAnswer:` with the target ` X`; no chat template, same for all models.
- Loss: next-token cross-entropy on the answer-letter token only.
- LoRA: r=16, alpha=32, dropout=0.05, all attention + MLP projections; lr 1e-4, 3 epochs, batch 8, bf16.
- Evaluation: argmax over the next-token logits of the valid option letters.

## Off Sol (RunPod / Lambda / Colab Pro A100 or L4)
The `sol_*.sh` scripts also run as plain bash: `pip install -r requirements.txt`, then
`HF_HOME=$PWD/hf bash sol_baselines.sh` (and the same for the other two). On Colab, keep the repo on your mounted Drive
so results survive a disconnect. Resuming just means rerunning the same command.
