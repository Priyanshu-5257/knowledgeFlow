#!/usr/bin/env python3
"""Build a 4-layer ~474M Spark2_5 draft by copying Base teacher weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_common import (
    BASE_ID,
    DEFAULT_LAYER_MAP,
    DRAFT_LAYER_TYPES,
    count_params,
    fix_generation_config,
    kaggle_out,
    patch_rope_validation,
    pick_device,
)


def parse_layer_map(raw: str) -> list[int]:
    parts = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if len(parts) != 4:
        raise ValueError(f"layer map must have 4 indices, got {parts}")
    return parts


def copy_draft_weights(teacher, draft, layer_map: list[int]) -> dict:
    src = teacher.state_dict()
    dst = draft.state_dict()
    copied, missing = [], []
    new_sd = {}

    for key, tensor in dst.items():
        if ".layers." in key:
            bits = key.split(".")
            try:
                li = bits.index("layers")
                src_layer = layer_map[int(bits[li + 1])]
            except (ValueError, IndexError) as e:
                raise KeyError(f"cannot rewrite layer key {key}") from e
            src_key = ".".join(bits[: li + 1] + [str(src_layer)] + bits[li + 2 :])
            if src_key not in src:
                missing.append((key, src_key))
                continue
            new_sd[key] = src[src_key]
            copied.append((key, src_key))
        else:
            if key in src and src[key].shape == tensor.shape:
                new_sd[key] = src[key]
                copied.append((key, key))
            elif key == "lm_head.weight" and "model.embedding.weight" in src:
                new_sd[key] = src["model.embedding.weight"]
                copied.append((key, "model.embedding.weight"))
            else:
                missing.append((key, key))

    incompatible = draft.load_state_dict(new_sd, strict=False)
    return {
        "copied": len(copied),
        "missing_src": missing,
        "load_missing": list(incompatible.missing_keys),
        "load_unexpected": list(incompatible.unexpected_keys),
        "layer_map": layer_map,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default=BASE_ID)
    p.add_argument("--layer-map", default=",".join(str(i) for i in DEFAULT_LAYER_MAP))
    p.add_argument("--out", default="spark-x25-draft-0.5B-init")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke-tokens", type=int, default=8)
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    layer_map = parse_layer_map(args.layer_map)
    out_dir = Path(kaggle_out(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_rope_validation()
    print("loading teacher", args.teacher, "on", device, flush=True)
    tok = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    teacher.to(device)
    teacher.eval()
    fix_generation_config(teacher)
    print("teacher params B", count_params(teacher) / 1e9, flush=True)

    cfg = AutoConfig.from_pretrained(args.teacher, trust_remote_code=True)
    cfg.num_hidden_layers = 4
    cfg.layer_types = list(DRAFT_LAYER_TYPES)
    draft = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=dtype)
    draft.to(device)
    report = copy_draft_weights(teacher, draft, layer_map)
    draft.eval()
    fix_generation_config(draft)
    n_params = count_params(draft)
    print("draft params B", n_params / 1e9, "count", n_params, flush=True)
    print("copy report", {k: report[k] for k in ("copied", "load_missing", "load_unexpected", "layer_map")}, flush=True)
    if report["missing_src"]:
        print("missing_src sample", report["missing_src"][:8], flush=True)

    if not (4.2e8 < n_params < 5.6e8):
        raise SystemExit(f"unexpected draft size {n_params}")
    if report["load_missing"]:
        raise SystemExit(f"incomplete copy, missing {report['load_missing'][:12]}")

    smoke = {"ok": False}
    if args.smoke_tokens > 0:
        prompt = "The capital of France is"
        inputs = tok(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = draft.generate(
                **inputs,
                max_new_tokens=args.smoke_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=50,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        smoke = {"ok": True, "prompt": prompt, "text": text}
        print("smoke", smoke, flush=True)

    tok.save_pretrained(out_dir)
    draft.save_pretrained(out_dir)
    cfg.save_pretrained(out_dir)
    meta = {
        "ok": True,
        "teacher": args.teacher,
        "layer_map": layer_map,
        "layer_types": list(DRAFT_LAYER_TYPES),
        "num_parameters": n_params,
        "copy": {k: report[k] for k in ("copied", "load_missing", "load_unexpected", "missing_src")},
        "smoke": smoke,
    }
    (out_dir / "draft_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    # also put a copy at kaggle working root for easy download
    root_meta = Path(kaggle_out("draft_build.json"))
    if root_meta.parent != out_dir:
        root_meta.write_text(json.dumps({"ok": True, "out": str(out_dir), **{k: meta[k] for k in ("num_parameters", "layer_map", "smoke")}}, indent=2))
    print("WROTE", out_dir, flush=True)


if __name__ == "__main__":
    main()
