#!/usr/bin/env python3
"""Measure Spark-X2.5-1.7B thinking length vs accuracy.

Loads the whole model on one GPU. device_map=auto on 2xT4 splits weights
and then generate() fails with a cuda:0 vs cuda:1 matmul error.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "XHToken/Spark-X2.5-1.7B"
COD = (
    "Think step by step, but only keep a minimum draft for each thinking step, "
    "with 5 words at most. Put the final answer after the thinking."
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
BOX_RE = re.compile(r"\\boxed\{([^}]*)\}")
MODES = ("think", "no_think", "cod")

PROBLEMS = [
    {
        "id": "gsm8k_easy",
        "answer": "18",
        "q": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    },
    {
        "id": "gsm8k_med",
        "answer": "540",
        "q": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    },
    {
        "id": "gsm8k_word",
        "answer": "70",
        "q": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    },
    {"id": "math_arith", "answer": "36", "q": "What is 12 * 3?"},
    {
        "id": "math_frac",
        "answer": "5/6",
        "q": "Compute 1/2 + 1/3. Give the answer as a simplified fraction.",
    },
    {"id": "aime_lite", "answer": "16", "q": "Find the remainder when 2^10 is divided by 17."},
    {"id": "algebra", "answer": "4", "q": "If 3x + 5 = 17, what is x?"},
    {
        "id": "count",
        "answer": "45",
        "q": "How many 2-element subsets does {1,2,3,4,5,6,7,8,9,10} have?",
    },
]


def patch_rope_validation() -> None:
    try:
        from transformers.modeling_rope_utils import RopeParametersMixin

        RopeParametersMixin.validate_rope = lambda self: None
        print("patched rope validation", flush=True)
    except Exception as e:
        print("no rope patch needed", type(e).__name__, e, flush=True)


def normalize(s: str) -> str:
    return str(s).strip().replace(",", "").replace("$", "")


def extract_answer(text: str) -> str:
    boxed = BOX_RE.findall(text)
    if boxed:
        return normalize(boxed[-1])
    after = THINK_RE.sub("", text)
    nums = re.findall(r"-?\d+/\d+|-?\d+(?:\.\d+)?", after)
    return normalize(nums[-1]) if nums else ""


def count_think_tokens(text: str, tokenizer) -> int:
    match = THINK_RE.search(text)
    if match:
        return len(tokenizer.encode(match.group(1), add_special_tokens=False))
    if "<think>" in text.lower():
        return len(tokenizer.encode(text, add_special_tokens=False))
    return 0


def apply_template(tok, messages, enable_thinking: bool) -> str:
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )


def load_model(device: torch.device, dtype, temp: float, top_p: float):
    patch_rope_validation()
    print("loading", MODEL_ID, "on", device, flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()
    if getattr(model, "generation_config", None) is not None:
        # Spark ships top_k=-1 ("disabled"); HF generate() requires k > 0.
        model.generation_config.top_k = 50
        model.generation_config.temperature = temp
        model.generation_config.top_p = top_p
    print("loaded in", round(time.time() - t0, 1), "s", flush=True)
    print(
        "device",
        next(model.parameters()).device,
        "dtype",
        next(model.parameters()).dtype,
        "params B",
        sum(p.numel() for p in model.parameters()) / 1e9,
        flush=True,
    )
    return tok, model


def run_one(tok, model, device, problem, mode, max_new, temp, top_p, top_k):
    if mode == "think":
        messages = [{"role": "user", "content": problem["q"]}]
        enable_thinking = True
    elif mode == "no_think":
        messages = [{"role": "user", "content": problem["q"]}]
        enable_thinking = False
    elif mode == "cod":
        messages = [{"role": "user", "content": COD + "\n\n" + problem["q"]}]
        enable_thinking = True
    else:
        raise ValueError(mode)

    prompt = apply_template(tok, messages, enable_thinking)
    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    n_in = int(inputs["input_ids"].shape[1])
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    dt = time.time() - t0
    gen_ids = out[0][n_in:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    n_out = int(gen_ids.shape[0])
    n_think = count_think_tokens(text, tok)
    pred = extract_answer(text)
    gold = normalize(problem["answer"])
    correct = bool(pred) and (pred == gold or pred.endswith(gold) or gold.endswith(pred))
    return {
        "id": problem["id"],
        "mode": mode,
        "n_in": n_in,
        "n_out": n_out,
        "n_think": n_think,
        "n_answer": max(n_out - n_think, 0),
        "hit_cap": n_out >= max_new - 2,
        "seconds": round(dt, 2),
        "tok_s": round(n_out / dt, 2) if dt > 0 else 0,
        "pred": pred,
        "gold": gold,
        "correct": correct,
        "text_head": text[:600],
        "text_tail": text[-400:] if len(text) > 400 else text,
    }


def agg(rows, mode):
    xs = [r for r in rows if r["mode"] == mode]
    if not xs:
        return {}
    return {
        "n": len(xs),
        "acc": sum(r["correct"] for r in xs) / len(xs),
        "mean_out": sum(r["n_out"] for r in xs) / len(xs),
        "mean_think": sum(r["n_think"] for r in xs) / len(xs),
        "mean_answer": sum(r["n_answer"] for r in xs) / len(xs),
        "hit_cap": sum(r["hit_cap"] for r in xs),
        "mean_s": sum(r["seconds"] for r in xs) / len(xs),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/kaggle/working/spark_reason_probe.json")
    p.add_argument("--max-new", type=int, default=1536)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tok, model = load_model(device, dtype, args.temperature, args.top_p)

    rows, errors = [], []
    for problem in PROBLEMS:
        for mode in MODES:
            print(f"=== {problem['id']} / {mode} ===", flush=True)
            try:
                row = run_one(
                    tok,
                    model,
                    device,
                    problem,
                    mode,
                    args.max_new,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                )
                rows.append(row)
                print(
                    {k: row[k] for k in ["n_out", "n_think", "correct", "pred", "gold", "seconds", "hit_cap"]},
                    flush=True,
                )
            except Exception as e:
                err = {
                    "id": problem["id"],
                    "mode": mode,
                    "error": repr(e),
                    "tb": traceback.format_exc()[-1500:],
                }
                errors.append(err)
                print("ERROR", err["error"], flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary = {
        "ok": len(errors) == 0 and len(rows) == len(PROBLEMS) * len(MODES),
        "model": MODEL_ID,
        "max_new_tokens": args.max_new,
        "device": str(device),
        "n_rows": len(rows),
        "n_errors": len(errors),
        "by_mode": {m: agg(rows, m) for m in MODES},
        "rows": rows,
        "errors": errors,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print("WROTE", out_path, flush=True)
    print(
        json.dumps(
            {"ok": summary["ok"], "by_mode": summary["by_mode"], "n_errors": summary["n_errors"]},
            indent=2,
        ),
        flush=True,
    )
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
