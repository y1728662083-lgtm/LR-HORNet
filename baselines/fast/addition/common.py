"""Shared, dependency-light helpers for the addition experiments."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ADDITION_ROOT = Path(__file__).resolve().parent
PAPER4_ROOT = ADDITION_ROOT.parent


def resolve_from_addition(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ADDITION_ROOT / path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def import_module_from_file(name: str, source: str | Path):
    source = Path(source).resolve()
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    # Decorators in third-party modules (notably timm.register_model) resolve
    # the defining module through sys.modules while the file is executing.
    # importlib.import_module performs this registration automatically; a
    # file-location import must do it explicitly before exec_module().
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def require_cuda_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda" and (device.index or 0) >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {device}, but only {torch.cuda.device_count()} GPU(s) are "
            f"visible (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
        )
    return device


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
