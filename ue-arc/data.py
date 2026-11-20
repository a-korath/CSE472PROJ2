"""ARC-Challenge loading, label normalization, and the single prompt format used everywhere.

Run once:  python data.py   -> writes data/{train,validation,test}.jsonl
"""
import json
import os

LETTERS = "ABCDE"  # ARC-Challenge has 3-5 options; labels come as A-E or 1-5


def load_arc(split):
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    rows = []
    for r in ds:
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue  # malformed row; skip
        rows.append({
            "id": r["id"],
            "question": r["question"],
            "choices": list(r["choices"]["text"]),
            # position of the correct option -> letter is re-derived as A, B, C...
            "answer_idx": labels.index(r["answerKey"]),
        })
    return rows


def format_prompt(ex):
    """Question + lettered options + 'Answer:'. The model's next token should be ' A'/' B'/..."""
    lines = [f"Question: {ex['question']}"]
    for i, c in enumerate(ex["choices"]):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def answer_letter(ex):
    return LETTERS[ex["answer_idx"]]


def save_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path, limit=None):
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


if __name__ == "__main__":
    for split in ["train", "validation", "test"]:
        rows = load_arc(split)
        save_jsonl(rows, f"data/{split}.jsonl")
        print(f"{split}: {len(rows)} examples")
    print("\nExample prompt:\n" + format_prompt(rows[0]) + " " + answer_letter(rows[0]))
