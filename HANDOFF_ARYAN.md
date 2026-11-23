# Handoff for Aryan: final-report review fixes (Mon Nov 23)

**TL;DR:** Arjun ran a full review of `report/main.pdf` against the run files and the cited papers. **Every number in
Table 1 and the text was correct.** The changes are wording and citation fixes, plus one new paragraph on the mentor's
TextCloak paper, which supports our result. `main.pdf`, the figures and `report-overleaf.zip` are rebuilt. Still 8
pages. Please read §1 below (5 min), then do the items in §3.

## 1. What changed in `report/main.tex`

| Where | Before | After | Why |
|---|---|---|---|
| §2 Related Works, TextCloak paragraph | Only "it changes far more than two words" | Adds: TextCloak's authors argue low surrogate loss need not degrade a fine-tuned LLM because of pretraining priors; their **Textual UE (Li et al.) baseline changes ARC-Challenge accuracy by at most 2.6 points** across four LLMs it was not optimized against (their Table 3) | The mentor's own paper independently backs our negative result and our diagnosis. Not citing it was the biggest gap. |
| §2, "When Priors Backfire" | "pretrained models can learn genuine features…" | "…, **in image classification**, …"; closing sentence now "agrees with both findings" | That paper is about images only; the old wording over-reached. |
| §2, Li et al. patterns | "zero Jaccard overlap; five words cover most examples" | "in their most effective, round-0 sets, zero Jaccard overlap, and five words cover **60–100%**" | Zero overlap holds only for Error-min-0 (their Table 1); top-5 share is 0.60–1.00 (their Table 2). |
| §3 Perturbation | "4,350 editable words" | "4,350 **distinct** editable words, covering about 95% of word occurrences" + one line saying the neighbour lists came from the `NousResearch/Meta-Llama-3-8B` mirror (same weights) | 4,350 is the vocabulary, not the occurrence count (~43,700). The mirror must be disclosed (course rule: cite every model used). |
| §3 Perturbation | "…so we use it." | "…so we use it, **as does Zhao et al. (2026, Eq. 4)**" | TextCloak writes Li et al.'s score in the same difference form. |
| §4.1 Fine-tuning | "identical for every run, **including the surrogate** … 3 epochs" | "identical for every **victim** run … The surrogate uses the same settings except that it trains **2 epochs per round**" | Contradicted §3 and `protect.py` (`--train_epochs 2`). |
| §4.1 Evaluation | none | "compare conditions with Welch's t-test over seeds (SciPy)" | Needed for the new p-values. |
| §4.2 White-box | "differences smaller than one standard deviation" | Adds Welch p-values: **protected vs random p = 0.82, round 0 vs random p = 0.68; protected vs clean p = 0.02, random vs clean p = 0.005**, plus "three seeds give little power: no detectable difference, not proof of none" | Answers checklist §4 box 1 with a test instead of an eyeball. |
| §4.2 Surrogate + Fig. 2 caption | Surrogate "memorizes the training questions" | Adds: loss on the **unedited** versions of the same questions stays **0.21–0.54** in rounds 1–5 (`rounds.jsonl`), so it fits **the edited text, not the questions** | Sharper and actually what the data shows (checklist §4 box 2). Same change in the abstract and the one-sentence answer. |
| §4.3 Analysis | "Li et al.: over 90%" | "Li et al.: 60–100%", plus "Jaccard alone is not decisive (their AG-News Error-min-1 also scores 0.08), but the top-five share is" | Wrong number fixed; our 0.08 Jaccard equals one of theirs, so the top-5 share is the real evidence. |
| §4.3 | none | "Zhao et al. (2026) make the same argument for LLMs" | See TextCloak row above. |
| §4.4 Bonus | "about 10,000 choices … 39% end with a period" | "none of the **10,357** choices … about **40%**" | Exact count; 39.5% rounds to 40. |
| §4.4 Outcome | "PPL ratio 1.16 (**within our 1.2 limit**)" | "a **mean** PPL ratio of 1.16 (the per-question filters were not applied to the hint set)" | The 1.2 limit is per question and was never enforced for the hint. |
| Abstract + §4.4 | "below zero-shot on all three victims" | "… **in every seed**" | Verified: all 9 hint runs are below their zero-shot. |
| Acknowledgments | | Adds Accelerate, SciPy, Matplotlib | Course rule: cite every package used. |

## 2. Other files
- `references.bib`
  - Added `gugger2022accelerate`, `virtanen2020scipy` and `hunter2007matplotlib`.
  - `li2026priors` now cites its venue, ICLR 2026, instead of the arXiv preprint.
  - Braces keep the capitals in `{TextCloak}` and `{Try} {ARC}`.
- `make_figures.py`: the random-edit metrics are now read from `ue-arc/logs/local_random.log` instead of three hand-typed
  numbers. The regenerated `table_main.tex` is **byte-identical** to before.
- `figures/*`, `main.pdf` and `report-overleaf.zip` are regenerated.
  - The build is pdfLaTeX with Times, as before: 0 undefined references, 0 overfull boxes, 0 BibTeX warnings.
  - The body is now ~6.25 pages: Acknowledgments start at the bottom of p. 6 and spill onto p. 7. References take p. 7–8.

## 3. What Arjun verified (you can tick these in REVIEW_CHECKLIST.md)
- **§3 Table 1:** all 16 accuracies and stds recomputed from the 46 `results/*.json` files (all n = 1172).
- **§3 text numbers:**
  - Data: splits 1,119/299/1,172; 1,117 four-choice + 1 three-choice + 1 five-choice questions.
  - Edits: shares 23/16/13%; 21% of questions unedited; 1.18 vs 1.87 edits per question.
  - Metrics and surrogate: Jaccard 0.071–0.084 (random 0.135); `rounds.jsonl` values; hint seed ranges.
- **§3 random-edit readability:** `data/random_k2.jsonl` was committed once (initial commit) and never changed, and
  its metrics line matches Table 1. Same file the victims trained on.
- **§4 scoring formula:** matches `protect.py` (difference form, top-20, filter, exact re-score, accept only if the loss
  drops, gradient recomputed per accepted swap).
- **§4 stop rule:** `sol_protect.sh` passes `--stop_acc 1.01`.
- **§4 pre-registration:** `b554aff` is dated 2026-11-20 11:58; the first hint results were committed at 14:54 the same day.
- **Citations checked against the PDFs:**
  - Li et al. 2023: LSTM/BiDAF surrogates, cos ≥ 0.5, one substitution, Eq. 5, stop on test metrics.
  - TextCloak: title, authors, arXiv number.
  - When Priors Backfire: authors; ICLR 2026.

## 4. Not done: needs a human
1. **Arjun's AI-use statement:** the statement covers Claude (app + Claude Code). Arjun must confirm he used no other
   tool *for this project*.
2. **Hand-labelling (checklist §5):** the 50 + 50 meaning-change labels are still undone; §4.3 says so honestly. About
   30 minutes if we want the rate in the report.
3. **Li et al. 2023 published version:** it appeared at TrustNLP 2023 (pp. 249–259). TextCloak cites it as "Li and Liu",
   but arXiv lists three authors, so we kept the arXiv entry. Switch only if you can confirm the workshop author list.
4. **Survey citation** (`li2025survey`, arXiv 2503.23536): authors not re-checked.
5. **Your read-through:** please read the TextCloak sentences in §2 and §4.3 yourself. The mentor will grade this, so
   check you're happy with how his paper is described.
6. **Not committed or pushed yet:** the changes are local in Arjun's clone. Once Arjun pushes, pull before editing, or
   we'll conflict.
7. **Submission:** checklist §7 (who uploads to Canvas, group vs individual) is unchanged.

## 5. Build note
If `latexmk -pdf main.tex` fails on a fresh TinyTeX with "eso-pic.sty / multirow.sty / enumitem.sty / pcrr7t not
found", run `tlmgr update --self && tlmgr install courier eso-pic multirow enumitem`. Overleaf needs nothing: upload the
zip.
