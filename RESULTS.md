# Results: unlearnable ARC-Challenge (CSE 472 Project 2.5)

All numbers: clean ARC-Challenge **test** accuracy (1,172 questions), LoRA r=16/α=32 on all 7 projections, lr 1e-4,
3 epochs, batch 8, answer-letter-only loss; mean ± sample std over seeds 1–3. Raw per-run numbers:
`ue-arc/results/all_results.csv`. Runs: Aryan (baselines 2110499, hint + Qwen3-8B-Base 2120261 / tunnel),
Arjun (protection 2078108, victims 2078235). Updated 2026-11-20.

## Main table

| Victim | Zero-shot | Clean | Random edits | Protected (min-min, r5) | Hint (bonus) |
|---|---|---|---|---|---|
| **Llama-3-8B** (white-box: same as surrogate) | 77.47 | 82.62 ± 0.54 | 80.18 ± 0.50 | 80.03 ± 0.92 | **51.45 ± 11.88** |
| **Qwen3-8B** (black-box) | 91.04 | 92.86 ± 0.05 | 92.04 ± 0.13 | 92.29 ± 0.40 | **89.90 ± 1.03** |
| Qwen3-8B-Base (black-box, added as headroom check) | 92.41 | 92.63 ± 0.26 | 92.18 ± 0.34 | 93.00 ± 0.17 | **80.75 ± 9.20** |

Fine-tuning gain (clean − zero-shot): Llama **+5.15**, Qwen3-8B +1.82, Qwen3-8B-Base +0.23.

Round-0 protected set (Li et al. Error-min-0: perturbations from the untrained surrogate, before any min-min training),
Llama only: **79.81 ± 1.30** (seeds 78.67 / 79.52 / 81.23), the same as r5 (80.03) and random (80.18). Running more rounds did not help.

## Readability and edit statistics (training set, 1,119 questions)

| Set | Edits / question | Questions edited | Where edits land (question / correct choice / wrong choice) | SBERT ↑ | GPT-2 PPL ratio ↓ |
|---|---|---|---|---|---|
| Random (k ≤ 2) | 1.87 | 100% | 51% / 13% / 37% | 0.970 | 1.13 |
| Protected r0 | 1.26 | 82% | 52% / 16% / 32% | 0.974 | 1.09 |
| Protected r5 (final) | 1.18 | 79% | 53% / 23% / 24% | 0.976 | 1.08 |
| Hint ("!" on correct choice) | 1.00 | 100% | 0% / 100% / 0% | 0.995 | 1.16 |

Pattern check (Li et al. Table 1/2 analogue), replacement words grouped by answer letter: Jaccard overlap between
letters 0.08 for every protected round (random 0.14); the top-5 words cover 4–12% of edits (Li et al.: >90% for
effective noise). **The protected edits are sample-wise; no class-wise shortcut formed.**

Surrogate (rounds.jsonl): train accuracy 99.2% after round 1 and ≥ 99.7% after; protected loss ≪ clean loss every
round (r5: 0.0068 vs 0.213). Smoke test on 16 questions: surrogate 100% after one round.

## Findings

1. **Min-min word swaps give no protection beyond random edits.** Llama (the surrogate's own family): protected 80.03
   vs random 80.18; both keep ~50% of the +5.15 fine-tuning gain. Qwen: same picture.
2. **Why: the surrogate memorizes instead of forming a shortcut.** An 8B model fits the 1,119 training questions
   (≥ 99% train accuracy) within one round, so each per-example swap lowers *that surrogate's* loss without creating
   a consistent pattern a fresh victim could latch onto (Jaccard 0.08, top-5 share ≤ 12%). The optimizer drifts
   toward the answer (23% of r5 edits on the correct choice vs 13% random) but never forms a hint within k = 2.
3. **The answer hint is a strong shortcut, and it transfers.** One "!" on the correct choice drives Llama 26 points
   *below* zero-shot and both Qwen models below zero-shot, at SBERT 0.995 / PPL ratio 1.16, without any gradients.
   High seed variance (Llama 42–65%, Qwen-Base 72–90%) shows victims become unstable when the shortcut is absent.
4. **Qwen ceiling.** Both Qwen3-8B variants already score ≥ 91% zero-shot, so fine-tuning adds ≤ 1.8 points: there is
   almost no learnable gain to protect, and Qwen cannot discriminate protected from random. Qwen3-8B-Base did not
   fix this (92.41 zero-shot, +0.23 gain). Likely ARC exposure during Qwen pretraining; state as a limitation.

## Bonus hypothesis outcome (pre-registered in docs/bonus_hypothesis.md, commit b554aff)

- (a) hint ≤ protected on Qwen3-8B: **confirmed** (89.90 vs 92.29).
- (b) hint within 2 points of zero-shot on both models: **Qwen3-8B confirmed** (−1.14); **Llama rejected by
  overshoot** (−26.0: the hint does not just remove the gain, it damages the model).

## Still to do
- [x] `python metrics.py data/train.jsonl protected/r0.jsonl protected/final.jsonl` → `results/metrics_protected.jsonl`.
- [x] Round-0 victims on Llama (job 2122972): 79.81 ± 1.30.
- [ ] Trade-off plot (accuracy vs SBERT; one point per set/round).
- [ ] Hand-label 50 protected + 50 random edits (meaning kept / changed), both teammates.
- [ ] Report + 2-slide deck (Nov 30).
