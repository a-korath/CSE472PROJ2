# Bonus hypothesis (written before any bonus result exists)

**Method.** Answer hint: append "!" to the correct choice of every training question (1 edit per question, no
optimization, no surrogate). Adapted from Li et al. (2023), Section 6, where class-wise symbols ("!" / "@") made text
unlearnable for BERT/RoBERTa fine-tuning.

**Hypothesis.** A model-agnostic answer hint protects at least as well as our min-min word swaps on Llama-3-8B, and
transfers better to Qwen3-8B, because it does not depend on the surrogate's gradients.

**Success criterion.** Mean clean-test accuracy (3 seeds) of the victim fine-tuned on the hint set is
(a) <= that of the victim fine-tuned on the protected set, for Qwen3-8B (black-box), and
(b) within 2 points of zero-shot for both models.
Reported either way, including if rejected.

**Known limitation.** An attacker who strips trailing punctuation removes the hint; robustness to such normalization
is future work.
