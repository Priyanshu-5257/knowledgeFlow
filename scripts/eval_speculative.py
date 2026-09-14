#!/usr/bin/env python3
"""Measure HuggingFace assisted-generation acceptance vs Instruct target."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_common import INSTRUCT_ID, kaggle_out, load_spark, patch_rope_validation

PROMPTS = [
    "The capital of France is",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "If 3x + 5 = 17, what is x?",
    "Compute 1/2 + 1/3 as a simplified fraction.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain in one sentence what speculative decoding is.",
]


def resolve_draft(raw: str) -> str:
    if raw and Path(raw).exists():
        return raw
    for pat in (
        "/kaggle/input/**/spark-x25-draft-0.5B-instruct-kd/config.json",
        "/kaggle/input/**/spark-x25-draft-0.5B-base-kd/config.json",
    ):
        hits = glob.glob(pat, recursive=True)
        if hits:
            return str(Path(hits[0]).parent)
    raise SystemExit("no draft checkpoint found")


def chat_prompt(tok, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    try:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": True},
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default=INSTRUCT_ID)
    p.add_argument("--draft-path", default="")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--out", default="speculative_eval.json")
    args = p.parse_args()

    # HF assisted generation requires target and draft on the same device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    draft_path = resolve_draft(args.draft_path)
    print("target", args.target, "draft", draft_path, "device", device, flush=True)

    tok, target = load_spark(args.target, device, dtype=torch.float16, eval_mode=True)
    patch_rope_validation()
    from transformers import AutoModelForCausalLM

    draft = AutoModelForCausalLM.from_pretrained(
        draft_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=None
    )
    draft.to(device)
    draft.eval()

    rows = []
    for i, q in enumerate(PROMPTS):
        prompt = chat_prompt(tok, q)
        inputs = tok(prompt, return_tensors="pt")
        t_inputs = {k: v.to(device) for k, v in inputs.items()}
        n_in = int(t_inputs["input_ids"].shape[1])

        t0 = time.time()
        with torch.no_grad():
            vanilla = target.generate(
                **t_inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        vanilla_s = time.time() - t0
        vanilla_n = int(vanilla.shape[1] - n_in)

        t1 = time.time()
        with torch.no_grad():
            assisted = target.generate(
                **t_inputs,
                max_new_tokens=args.max_new,
                do_sample=False,
                assistant_model=draft,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        assisted_s = time.time() - t1
        assisted_n = int(assisted.shape[1] - n_in)
        same = bool(torch.equal(vanilla[0], assisted[0]) if vanilla.shape == assisted.shape else False)
        row = {
            "id": i,
            "prompt": q[:80],
            "vanilla_tokens": vanilla_n,
            "vanilla_s": round(vanilla_s, 3),
            "assisted_tokens": assisted_n,
            "assisted_s": round(assisted_s, 3),
            "speedup": round(vanilla_s / assisted_s, 3) if assisted_s > 0 else None,
            "lossless": same,
            "vanilla_text": tok.decode(vanilla[0][n_in:], skip_special_tokens=True)[:240],
            "assisted_text": tok.decode(assisted[0][n_in:], skip_special_tokens=True)[:240],
        }
        rows.append(row)
        print({k: row[k] for k in ("id", "vanilla_s", "assisted_s", "speedup", "lossless", "vanilla_tokens")}, flush=True)

    speeds = [r["speedup"] for r in rows if r["speedup"]]
    summary = {
        "ok": True,
        "draft": draft_path,
        "n": len(rows),
        "mean_speedup": sum(speeds) / len(speeds) if speeds else None,
        "lossless_frac": sum(r["lossless"] for r in rows) / len(rows),
        "rows": rows,
    }
    out = Path(kaggle_out(args.out))
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("ok", "mean_speedup", "lossless_frac", "n")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
