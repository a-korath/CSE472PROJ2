# Handoff for Arjun: where the project stands (Fri Nov 20, 3 pm)

**TL;DR.** All the core experiments are done, plus the bonus and a third victim. The min-min protection doesn't beat
random edits. The pre-registered "answer hint" bonus wipes out the whole fine-tuning gain on every model. Every result
is in this repo now. What's left: ~30 min of small runs/plots, then the report (due Tue Nov 24; we aim for Mon Nov 23)
and 2 slides (Nov 30).

Start with `git pull`, then read **`RESULTS.md`**. It has the full table, the edit statistics, the findings and the bonus outcome.

---

## 1. What I ran (Aryan, on Sol)

| Job | What | Status |
|---|---|---|
| 2110499 | Baselines: zero-shot + clean fine-tune × 3 seeds for Llama-3-8B and Qwen3-8B, plus a 16-question protection smoke test on the real Llama | COMPLETED (26 min) |
| tunnel 2100054 | **Bonus**: hint victims, Llama + Qwen3-8B × 3 seeds; hint readability metrics | done |
| 2120261 | **Qwen3-8B-Base** (added as a headroom check): zero-shot, clean, random, protected, hint × 3 seeds | COMPLETED |

Your runs (protection 2078108, victims 2078235) are committed too: `protected/r0..r5.jsonl`, `rounds.jsonl`,
`results/*_{random,protected}_s*.json`, `logs/arjun/`.

## 2. The results

| Victim | Zero-shot | Clean | Random | Protected | Hint |
|---|---|---|---|---|---|
| Llama-3-8B | 77.47 | 82.62 ± 0.54 | 80.18 ± 0.50 | 80.03 ± 0.92 | **51.45 ± 11.88** |
| Qwen3-8B | 91.04 | 92.86 ± 0.05 | 92.04 ± 0.13 | 92.29 ± 0.40 | **89.90 ± 1.03** |
| Qwen3-8B-Base | 92.41 | 92.63 ± 0.26 | 92.18 ± 0.34 | 93.00 ± 0.17 | **80.75 ± 9.20** |

Raw numbers: `ue-arc/results/all_results.csv`. Per-run JSONs: `ue-arc/results/`.

## 3. What it means (the story for the report)

1. **Llama is the model where the comparison means anything.** Fine-tuning adds +5.15 points over zero-shot. Random
   and protected edits each keep about half of that, so **protected is no better than random**.
2. **Why min-min failed** (I checked this in the data, see RESULTS.md):
   - The surrogate hits ≥ 99% train accuracy after round 1, which means it **memorizes**.
   - The swaps are sample-wise: replacement words grouped by answer letter overlap only 0.08 (Jaccard), and the top-5 words cover ≤ 12% of edits. Li et al.'s effective noise had > 90%.
   - **No shared shortcut forms**, so a fresh victim has nothing to latch onto.
   - The optimizer did drift toward the answer: 23% of r5 edits land on the correct choice, against random's 13%. But with k = 2 it never formed a hint.
3. **Bonus = the positive result.** One `!` appended to the correct choice:
   - SBERT 0.995, PPL ratio 1.16, 1 edit per question.
   - Drives Llama **26 points below zero-shot**, and both Qwens below zero-shot.
   - It's exactly the consistent shortcut the optimizer failed to find.
   - Hypothesis (committed *before* results in `docs/bonus_hypothesis.md`, commit b554aff): (a) confirmed; (b) confirmed on Qwen, rejected on Llama by **overshoot**.
4. **Qwen ceiling (a limitation).** Both Qwen models already score ≥ 91% zero-shot, so fine-tuning adds ≤ 1.8 points
   (Base: +0.23). Qwen can't tell protected from random. Most likely ARC was in Qwen's pretraining data. Say so plainly.

## 4. Repo changes since your last commit (439906b)

- **Sol fixes, all verified on a real A100:**
  - class account and QoS `class` (Sol forces the class account onto the `public` partition anyway);
  - env activated via `PATH` — **this is the same fix you have on your Mac; it's already on main (235d772), so you don't need to push yours**;
  - `HF_HUB_OFFLINE=1` in jobs;
  - `setup_sol.sh` builds the env with mamba from `environment.yml` and reports the failing line if anything breaks.
- **`gpt2` → `openai-community/gpt2`** in `metrics.py` and setup (034ac87). The bare `gpt2` alias 404s with the new huggingface_hub. **Before running `metrics.py` on your account, download it once:** `hf download openai-community/gpt2`.
- **`sol_extra.sh`:** one resumable job for any set of models × training sets. Examples are in its header.
- **Bonus:** `make_hint.py`, `docs/bonus_hypothesis.md`, `data/hint.jsonl`.
- **`SOL_RUNBOOK.md`:** the full VPN → tunnel → setup → submit walkthrough that worked.
- Note: environment versions are newer than the laptop's: transformers 5.23, torch 2.14.1 (CUDA build), peft 0.22. The dry run matched your laptop numbers exactly (0.6875 / 0.625).

## 5. What's left, and a suggested split

**Small runs/plots (needed for the report):**
- [ ] **Readability for the protected sets.** These are the two TODO rows in RESULTS.md, ~2 min on any GPU node:
      `python metrics.py data/train.jsonl protected/r0.jsonl protected/final.jsonl` (after the gpt2 download above). **Arjun**
- [ ] **Round-0 victims** (Li et al.'s Error-min-0 test, ~12 min): `MODELS="meta-llama/Meta-Llama-3-8B" SETS="protected_r0" sbatch sol_extra.sh`. **Arjun**
- [ ] **Trade-off plot** (test accuracy vs SBERT, one point per set/round). **Aryan**
- [ ] **3–4 before/after example questions** (protected and hint). **Aryan**
- [ ] *(Optional)* 2 more Llama hint seeds, since the spread is 42–65%.
- [ ] *(Optional)* Hand-label 50 protected + 50 random edits (meaning kept / changed), both of us, for inter-rater agreement.

**Report (class section names; the guideline's content goes inside them):**

| Section | Owner |
|---|---|
| Abstract, Introduction | both |
| Related Works | Arjun |
| Model Description (= Design checkpoint: loss, surrogate, perturbation, filters, stop rule) | Aryan |
| Experiment: setup + results table (= Results checkpoint) | Arjun |
| Experiment: analysis (memorization mechanism, edit statistics, hint, Qwen ceiling) | Aryan |
| Future Work (hint robustness to punctuation stripping, class-wise min-min, other datasets) | Aryan |
| References (papers **and** every library/model), AI-use acknowledgment | Arjun |

Then we swap and edit each other's sections. **Slides:** 2 pages due Nov 30; Aryan presents and Arjun handles questions.

**Timeline:** Sat–Sun draft · Mon edits + submit · Tue buffer.

## 6. Using this from your Claude

Point it at this repo and say: *"Read HANDOFF_ARJUN.md and RESULTS.md, then help me with the items assigned to Arjun."*
Everything it needs is in the repo. The two Sol runs above need your Sol session; the rest is writing.
