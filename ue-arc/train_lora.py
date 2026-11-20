"""LoRA fine-tune = the attacker. Loss is on the answer letter only (design decision 1).
The SAME settings are used for every victim run; protect.py will reuse this loop as the surrogate.

python train_lora.py --model meta-llama/Meta-Llama-3-8B --train data/train.jsonl --seed 1 \
    --out adapters/llama_clean_s1 --eval_test data/test.jsonl --result results/llama_clean_s1.json
"""
import argparse
import json
import math
import os
import random

import torch

from data import answer_letter, format_prompt, load_jsonl
from eval import evaluate, load_model, write_result

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def set_seed(s):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def encode(tok, ex, max_len=512):
    """input = prompt + ' X'; labels = -100 everywhere except the answer-letter token."""
    p = tok(format_prompt(ex))["input_ids"]
    a = tok(" " + answer_letter(ex), add_special_tokens=False)["input_ids"]
    p = p[-(max_len - len(a)):]
    return p + a, [-100] * len(p) + a


def collate(batch, pad_id):
    L = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), L), pad_id)
    lab = torch.full((len(batch), L), -100)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        ids[i, :len(x)] = torch.tensor(x)
        lab[i, :len(y)] = torch.tensor(y)
        att[i, :len(x)] = 1
    return ids, att, lab


def add_lora(model):
    from peft import LoraConfig, get_peft_model

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))


def train(model, tok, rows, epochs=3, lr=1e-4, bs=8, seed=1, log_path=None):
    model.train()
    data = [encode(tok, r) for r in rows]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(data) / bs)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
    rng, step, log = random.Random(seed), 0, []
    for ep in range(epochs):
        rng.shuffle(data)
        for i in range(0, len(data), bs):
            ids, att, lab = (t.to(model.device) for t in collate(data[i:i + bs], tok.pad_token_id))
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            log.append({"step": step, "epoch": ep, "loss": loss.item()})
            if step % 20 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f}", flush=True)
            step += 1
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "w") as f:
            for r in log:
                f.write(json.dumps(r) + "\n")
    return log


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--out", required=True, help="adapter directory")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--eval_test", default=None, help="evaluate right after training (saves a model load)")
    p.add_argument("--result", default=None)
    p.add_argument("--limit", type=int, default=None, help="dry runs only")
    a = p.parse_args()

    set_seed(a.seed)
    model, tok = load_model(a.model)
    model = add_lora(model)
    model.print_trainable_parameters()
    rows = load_jsonl(a.train, a.limit)
    train(model, tok, rows, a.epochs, a.lr, a.bs, a.seed, log_path=os.path.join(a.out, "loss.jsonl"))
    model.save_pretrained(a.out)

    if a.eval_test:
        test = load_jsonl(a.eval_test, a.limit)
        acc = evaluate(model, tok, test)
        write_result(a.result or os.path.join(a.out, "result.json"),
                     {"model": a.model, "train": a.train, "seed": a.seed, "n": len(test),
                      "accuracy": acc, "lora": LORA, "epochs": a.epochs, "lr": a.lr, "bs": a.bs})
