#!/usr/bin/env python3
"""Build a Spark2_5 draft: layer-pruned 1.7B copy, or Qwen3.5-0.8B-like 24×1024 net."""

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
    QWENLIKE_LAYER_TYPES,
    apply_qwenlike_draft_config,
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
            new_sd[key] = src[src_key].to(dtype=tensor.dtype)
            copied.append((key, src_key))
        else:
            if key in src and src[key].shape == tensor.shape:
                new_sd[key] = src[key].to(dtype=tensor.dtype)
                copied.append((key, key))
            elif key == "lm_head.weight" and "model.embedding.weight" in src:
                new_sd[key] = src["model.embedding.weight"].to(dtype=tensor.dtype)
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


def init_embed_from_teacher(teacher, draft) -> str:
    """Width-halve teacher embeddings (2048 → 1024) by averaging pairs of dims."""
    src = teacher.get_input_embeddings().weight.data
    dst = draft.get_input_embeddings().weight
    if src.shape[0] != dst.shape[0]:
        return f"skip vocab mismatch {tuple(src.shape)} vs {tuple(dst.shape)}"
    if src.shape[1] != dst.shape[1] * 2:
        return f"skip hidden mismatch {tuple(src.shape)} vs {tuple(dst.shape)}"
    pooled = src.view(src.shape[0], dst.shape[1], 2).mean(dim=-1)
    dst.data.copy_(pooled.to(device=dst.device, dtype=dst.dtype))
    return f"pooled embed {tuple(src.shape)} -> {tuple(dst.shape)}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default=BASE_ID)
    p.add_argument("--arch", choices=["prune", "qwenlike"], default="qwenlike")
    p.add_argument("--layer-map", default=",".join(str(i) for i in DEFAULT_LAYER_MAP))
    p.add_argument("--out", default="spark-x25-draft-0.5B-init")
    p.add_argument("--device", default="auto")
    p.add_argument("--smoke-tokens", type=int, default=8)
    args = p.parse_args()

    device = pick_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    out_dir = Path(kaggle_out(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_rope_validation()
    print("loading teacher", args.teacher, "on", device, "arch", args.arch, flush=True)
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
    report = {"arch": args.arch}
    if args.arch == "qwenlike":
        apply_qwenlike_draft_config(cfg)
        draft = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=dtype)
        draft.to(device)
        report["embed_init"] = init_embed_from_teacher(teacher, draft)
        report["layer_map"] = None
        report["layer_types"] = list(QWENLIKE_LAYER_TYPES)
    else:
        cfg.num_hidden_layers = 4
        cfg.layer_types = list(DRAFT_LAYER_TYPES)
        layer_map = parse_layer_map(args.layer_map)
        draft = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=dtype)
        draft.to(device)
        report.update(copy_draft_weights(teacher, draft, layer_map))
        report["layer_types"] = list(DRAFT_LAYER_TYPES)
        if report.get("load_missing"):
            raise SystemExit(f"incomplete copy, missing {report['load_missing'][:12]}")

    draft.eval()
    fix_generation_config(draft)
    n_params = count_params(draft)
    print("draft params B", n_params / 1e9, "count", n_params, flush=True)
    print("init report", {k: report[k] for k in report if k != "missing_src"}, flush=True)
    if not (3.8e8 < n_params < 5.8e8):
        raise SystemExit(f"unexpected draft size {n_params}")

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
        "arch": args.arch,
        "num_parameters": n_params,
        "hidden_size": int(cfg.hidden_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(cfg.num_key_value_heads),
        "intermediate_size": int(cfg.intermediate_size),
        "layer_types": list(cfg.layer_types),
        "init": {k: report[k] for k in report if k != "missing_src"},
        "smoke": smoke,
    }
    (out_dir / "draft_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    root_meta = Path(kaggle_out("draft_build.json"))
    if root_meta.parent != out_dir:
        slim = {k: meta[k] for k in ("ok", "arch", "num_parameters", "hidden_size", "num_hidden_layers", "smoke")}
        slim["out"] = str(out_dir)
        root_meta.write_text(json.dumps(slim, indent=2))
    print("WROTE", out_dir, flush=True)


if __name__ == "__main__":
    main()
