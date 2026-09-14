"""Shared Spark-X2.5 load helpers. Pin transformers==4.57.1 on Kaggle."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "XHToken/Spark-X2.5-1.7B-Base"
INSTRUCT_ID = "XHToken/Spark-X2.5-1.7B"
TRANSFORMERS_PIN = "4.57.1"

DRAFT_LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
]
DEFAULT_LAYER_MAP = [0, 1, 2, 3]


def patch_rope_validation() -> None:
    try:
        from transformers.modeling_rope_utils import RopeParametersMixin

        RopeParametersMixin.validate_rope = lambda self: None
    except Exception:
        pass


def fix_generation_config(model, temperature: float = 1.0, top_p: float = 0.95, top_k: int = 50) -> None:
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    # Spark ships top_k=-1; HF generate() requires k>0. do_sample must be True
    # or save_pretrained rejects top_p/temperature.
    cfg.do_sample = True
    cfg.top_k = top_k
    cfg.temperature = temperature
    cfg.top_p = top_p


def load_spark(model_id: str, device: str, dtype=torch.float16, eval_mode: bool = True):
    patch_rope_validation()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    if eval_mode:
        model.eval()
    fix_generation_config(model)
    return tok, model


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def kaggle_out(path: str) -> str:
    if os.path.isdir("/kaggle/working"):
        if path.startswith("/kaggle/"):
            return path
        return os.path.join("/kaggle/working", path)
    return path


def teacher_student_devices():
    if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        return torch.device("cuda:0"), torch.device("cuda:1")
    dev = pick_device("auto")
    return dev, dev
