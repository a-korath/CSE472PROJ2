"""Builds every figure/table number in the report from the run files (no typed numbers).
python make_figures.py   (run from report/)"""
import csv, json, statistics as st, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "../ue-arc/"
acc = collections.defaultdict(list)
for r in csv.DictReader(open(R + "results/all_results.csv")):
    acc[(r["model"], r["train_set"])].append(100 * float(r["test_accuracy"]))
met = {json.loads(l)["file"]: json.loads(l) for l in open(R + "results/metrics_protected.jsonl")}
met["data/hint.jsonl"] = json.load(open(R + "results/metrics_hint.json"))
met["random"] = json.loads(open(R + "logs/local_random.log").read().strip().splitlines()[-1])  # random_k2 metrics, Arjun's laptop run
PPL = {"clean": met["data/hint.jsonl"]["ppl_orig"], "random": met["random"]["ppl"], "protected_r0": met["protected/r0.jsonl"]["ppl"], "protected": met["protected/final.jsonl"]["ppl"], "hint": met["data/hint.jsonl"]["ppl"]}
SB = {"clean": (1.0, 1.0), "random": (met["random"]["sbert"], met["random"]["ppl_ratio"]),
      "protected_r0": (met["protected/r0.jsonl"]["sbert"], met["protected/r0.jsonl"]["ppl_ratio"]),
      "protected": (met["protected/final.jsonl"]["sbert"], met["protected/final.jsonl"]["ppl_ratio"]),
      "hint": (met["data/hint.jsonl"]["sbert"], met["data/hint.jsonl"]["ppl_ratio"])}
STY = {"clean": ("#2a78d6", "o", "Original"), "random": ("#eb6834", "s", "Random edits"),
       "protected_r0": ("#1baf7a", "^", "Protected r0"), "protected": ("#eda100", "D", "Protected r5"),
       "hint": ("#e87ba4", "*", "Hint (bonus)")}
MODELS = [("Meta-Llama-3-8B", "Llama-3-8B (white-box)"), ("Qwen3-8B", "Qwen3-8B (black-box)"),
          ("Qwen3-8B-Base", "Qwen3-8B-Base (black-box)")]

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e5e4df", "grid.linewidth": 0.5})
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
for ax, (m, title) in zip(axes, MODELS):
    z = acc[(m, "zeroshot")][0]
    ax.axhline(z, color="#8a8984", lw=1, ls="--", zorder=1)
    ax.text(0.9835, z, " zero-shot", fontsize=6.5, color="#52514e", va="top", ha="center")
    for s, (c, mk, lab) in STY.items():
        if (m, s) not in acc: continue
        v = acc[(m, s)]; mu, sd = st.mean(v), (st.stdev(v) if len(v) > 1 else 0)
        ax.errorbar(SB[s][0], mu, yerr=sd, fmt=mk, color=c, ms=7 if mk == "*" else 5, mec="white", mew=0.6,
                    ecolor=c, elinewidth=1, capsize=2, zorder=3, label=lab)
    ax.set_title(title, fontsize=8.5, color="#0b0b0b")
    ax.set_xlim(0.965, 1.004)
axes[0].set_ylabel("Clean test accuracy (%)")
axes[1].set_xlabel("SBERT similarity of training text to the original (higher = meaning kept)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("figures/tradeoff.pdf", bbox_inches="tight"); fig.savefig("figures/tradeoff.png", dpi=200, bbox_inches="tight")

# surrogate rounds
rows = [json.loads(l) for l in open(R + "protected/rounds.jsonl")]
srch = [r for r in rows if r["phase"] == "search"]; ev = {r["round"]: r["surrogate_train_acc"] for r in rows if r["phase"] == "eval"}
fig, ax = plt.subplots(figsize=(3.3, 2.0))
x = [r["round"] for r in srch]
ax.plot(x, [r["clean_loss"] for r in srch], "-o", color="#2a78d6", lw=2, ms=4, label="clean questions")
ax.plot(x, [r["protected_loss"] for r in srch], "-s", color="#eb6834", lw=2, ms=4, label="protected questions")
for xi, r in zip(x, srch):
    if xi in ev: ax.annotate(f"{100*ev[xi]:.1f}%", (xi, r["clean_loss"]), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6, color="#52514e")
ax.set_xlabel("min–min round"); ax.set_ylabel("surrogate answer loss"); ax.set_xticks(x)
ax.set_ylim(-0.03, 0.9); ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2); fig.tight_layout(); fig.savefig("figures/rounds.pdf", bbox_inches="tight")

# numbers for the tables (printed; pasted nowhere by hand: the .tex reads table_main.tex)
def ms(m, s):
    v = acc.get((m, s));
    if not v: return "--"
    return f"{v[0]:.2f}" if len(v) == 1 else f"{st.mean(v):.2f} $\\pm$ {st.stdev(v):.2f}"
lines = []
names = {"zeroshot": "None (zero-shot)", "clean": "Original", "random": "Random edits", "protected_r0": "Protected, round 0",
         "protected": "Protected, round 5 (final)", "hint": "Answer hint (bonus)"}
for m, title in MODELS:
    first = True
    for s in ["zeroshot", "clean", "random", "protected_r0", "protected", "hint"]:
        if (m, s) not in acc: continue
        sb, pr = ("--", "--") if s == "zeroshot" else (f"{SB[s][0]:.3f}", f"{PPL[s]:.1f} ({SB[s][1]:.2f})")
        mname = title.split(" (")[0] if first else ""
        first = False
        lines.append(f"{mname} & {names[s]} & {ms(m, s)} & {pr} & {sb} \\\\")
    lines.append("\\midrule")
open("table_main.tex", "w").write("\n".join(lines[:-1]) + "\n")
for m, _ in MODELS:
    z, c = acc[(m, "zeroshot")][0], st.mean(acc[(m, "clean")])
    print(m, "zero", round(z, 2), "clean", round(c, 2), "gain", round(c - z, 2),
          {s: round(st.mean(acc[(m, s)]), 2) for s in STY if (m, s) in acc})
print(SB)
