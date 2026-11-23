# Pre-submission review: CSE 472 Project 2.5 final report

**File to submit:** `report/main.pdf` (repo) = `CSE472_Project2_Report.pdf`
**Deadline:** Tue Nov 24 (confirm the exact time on Canvas). Internal target: submit today (Mon Nov 23).
**How to fix something:** edit `report/main.tex` and run `latexmk -pdf main.tex` in `report/`, or upload
`report/report-overleaf.zip` to Overleaf. If you change a number, change the run file and rerun
`python make_figures.py`. Never type a number into the table by hand.

Each of us reads the **whole** report once (about 20 min), then does the checks marked with their name.
Tick a box only after actually checking it. Write anything you disagree with under "Open issues" at the bottom.

---

## 1. Submission requirements (Aryan)

- [ ] PDF, 6–8 pages in an ML conference style. Currently 8 pages: 6 body + 2 references (ICLR 2027 style).
- [ ] Class sections are all present (announcement of Nov 2): Abstract, Introduction, Related Works, Model
      Description, Experiment, Future Work, References.
- [ ] Guideline sections are all present (mentor's handout, p. 6–7), as subsections:
      Abstract/Intro with the research question **and a one-sentence answer** (end of §1 para 2) ·
      Related Work · Method (= §3) · Experimental Setup (§4.1) · Results (§4.2) · Analysis and Limitations (§4.3) ·
      Bonus (§4.4).
- [ ] **Design checkpoint** is in §3: loss, surrogate, perturbation, *and the reasoning behind each* ("Why:" lines).
- [ ] **Results checkpoint** is in §4.2: results table for white-box and black-box, with GPT-2 perplexity and SBERT
      (Table 1), plus the trade-off plot (Fig. 1) and examples (Table 2).
- [ ] References include every paper **and** every package, model, dataset and tool used (course rule).
- [ ] The generative-AI acknowledgment is accurate (course rule). It is in Acknowledgments.
- [ ] Running header reads "CSE 472 Social Media Mining, Project 2.5 (Fall 2026)", not "Published at ICLR".

## 2. Names, people, wording (both)

- [ ] Author line: names spelled right, order OK (currently "Aryan Patyal & Arjun Korath").
- [ ] Affiliation "School of Computing and Augmented Intelligence, Arizona State University" is right.
      Add ASU emails? (none currently)
- [ ] Mentor named correctly: **Chengshuai Zhao**. He is first author of TextCloak (Huan Liu is last author). Check
      the TextCloak citation and the sentence describing it in §2 are fair to his paper.
- [ ] **Arjun:** the AI-use statement covers how *you* used AI too (it currently says "Anthropic's Claude (Claude Opus
      models, via the Claude app and Claude Code)"). Add any other tool you used.
- [ ] Title is OK: "Unlearnable Text for LLM Fine-Tuning: Min–Min Word Swaps Fail, a One-Character Hint Works".

## 3. Every number matches the data (Arjun: table; Aryan: text)

Source of truth: `ue-arc/results/all_results.csv`, `ue-arc/results/metrics_*.json*`, `ue-arc/protected/rounds.jsonl`.

- [ ] **Arjun:** Table 1 matches. Quick check:
      `cd report && python make_figures.py` prints the means; compare with the PDF.
- [ ] **Aryan:** the numbers in the text match Table 1:

| Where | Claim | Check against |
|---|---|---|
| Abstract | protected 80.03, random 80.18, clean 82.62, zero-shot 77.47, round-0 79.81, hint Llama 51.45 | Table 1 |
| §4.2 | Llama gain +5.15; Qwen3-8B +1.82; Qwen3-8B-Base +0.23 | clean − zero-shot |
| §4.2 | SBERT 0.976 vs 0.970; PPL ratio 1.08 vs 1.13; edits 1.18 vs 1.87; 21% of questions unedited | `metrics_protected.jsonl`, RESULTS.md |
| §4.2 / Fig. 2 | protected loss 0.0068 vs clean 0.213 at r5; surrogate train acc 99.2% then ≥ 99.7% | `rounds.jsonl` |
| §4.3 | Jaccard 0.07–0.08 (random 0.14); top-5 share 4–12%; 23% / 13% / 16% of edits on the correct choice | RESULTS.md analysis |
| §4.3 | "about 2.5 points" lost to edits on Llama | 82.62 − 80.18 = 2.44; 82.62 − 80.03 = 2.59 |
| §4.4 | hint SBERT 0.995, PPL ratio 1.16; Qwen3-8B −1.14 vs zero-shot; Llama −26 | `metrics_hint.json`, Table 1 |
| §4.4 | seed ranges: Llama hint 42.5–64.9, Qwen3-8B-Base hint 72.0–90.4 | `all_results.csv` |
| §4.4 | about 10,000 choices, none end with "!", 39% end with "." | counted over train+val+test |
| §3 | 50 neighbours, 4,350 editable words, k = 2, top-20 shortlist | `protect.py`, PROGRESS.md |
| §4.1 | 1,119 / 299 / 1,172 questions; 1,117 four-choice + 1 three + 1 five | `data/*.jsonl` |
| §4.1 | LoRA r 16, α 32, dropout 0.05, 7 projections; lr 1e-4, 3 epochs, bs 8, warm-up 5%, clip 1.0, max len 512 | `train_lora.py` |

- [ ] Random-edit readability (SBERT 0.970, PPL 106.5 / ×1.13) came from Arjun's laptop run (PROGRESS.md), not a
      Sol run. **Arjun:** confirm it was measured on the same `data/random_k2.jsonl` that the victims trained on.

## 4. Claims we must be comfortable defending (both)

Read these sentences carefully. They are where a grader or the mentor will push.

- [ ] **"Protected is no better than random"** (§4.2): we say "within seed noise: differences smaller than one standard
      deviation". We did not run a significance test. OK as worded?
- [ ] **"The surrogate memorizes"** (§4.3): the evidence is ≥ 99% training accuracy after one round, plus the
      sample-wise edit statistics. It is an explanation, not a proven cause. Is the wording careful enough?
- [ ] **Qwen ceiling and pretraining exposure** (§4.3): we say "plausibly because ARC-style questions are in their
      pretraining data". This is a hypothesis; we did not test it. Keep, soften, or cut?
- [ ] **"Removes the entire fine-tuning gain on all three victims"** (abstract): true, since hint < zero-shot for all three.
      But the Llama result has huge variance (42–65%). Are we happy stating it with 3 seeds?
- [ ] **Bonus hypothesis was pre-registered:** commit `b554aff` (`docs/bonus_hypothesis.md`) is dated **before** any hint
      result. Confirm with `git log --format='%h %ad %s' -- docs/bonus_hypothesis.md`.
- [ ] **Our scoring formula vs Li et al.** (§3): we say Li et al.'s Eq. 5 uses `e_sᵀ∇L` only and ours uses
      `(e_new − e_old)ᵀ∇L`. **Arjun:** this is your code; confirm it is described correctly.
- [ ] **Stop rule** (§3): "Li et al. stop when test metrics stop changing" (their Sec. 2.2). We ran a fixed 5 rounds
      with early stopping disabled. Matches what actually ran (`sol_protect.sh`, `--stop_acc 1.01`)?
- [ ] **Related work** (§2): we cite "When Priors Backfire" (Li et al., 2026) as consistent with our finding. Fair?

## 5. Weaknesses we chose to state openly (decide together)

These are in the report on purpose; reporting them honestly is safer than a grader finding them.

- [ ] **Meaning damage example:** Table 2 shows a correct answer changed "Wood" → "Water", and "atomic mass" →
      "molecular mass". Keep (recommended), or replace with a milder example?
- [ ] **Hand-labelling not done:** §4.3 says we planned to label 50 protected + 50 random edits and didn't.
      Option: spend ~30 min labelling them now and add the rate. Otherwise keep the sentence as is.
- [ ] **Hint is easy to strip:** stated as a limitation and as future work. OK.

## 6. Look and feel (Aryan)

- [ ] Figures are readable when printed (Fig. 1 points carry distinct marker shapes, not just colours).
- [ ] No text overflows the margins; tables don't break across pages; no "??" anywhere (search the PDF for "??").
- [ ] Spell-check pass on the PDF.

## 7. Submit

- [ ] Both of us have ticked the boxes above, or listed open issues and resolved them.
- [ ] Final rebuild: `cd report && latexmk -pdf main.tex`, then commit and push.
- [ ] Decide who uploads to Canvas, and whether it's a group or individual submission. Confirm the file opens after upload.
- [ ] After submitting: 2-page slide PDF due **Nov 30, 11:59 pm**; 2-minute talk **Dec 1** (Aryan presents,
      Arjun takes questions).

---

## Open issues

| # | Who | Issue | Decision |
|---|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
