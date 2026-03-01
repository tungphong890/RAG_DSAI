"""
Device Configuration Utilities

Handles device detection and configuration for GPU/CPU execution.
Supports automatic device selection, environment variable overrides,
and GPU requirement enforcement for model inference.

Key Features:
- Automatic CUDA detection
- Environment variable configuration
- GPU requirement validation
- PyTorch device mapping
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class DeviceConfig:
    device: str  # 'cpu' or 'cuda'
    gpu_required: bool


def get_device_config(
    requested: Optional[str] = None,
    *,
    env_var: str = "RAG_DEVICE",
    gpu_required_env: str = "RAG_GPU_REQUIRED",
) -> DeviceConfig:
    """Resolve runtime device with environment variable support."""
    gpu_required = os.environ.get(gpu_required_env, "0").strip() in {
        "1",
        "true",
        "True",
    }

    raw = (requested or os.environ.get(env_var, "auto")).strip().lower()
    if raw not in {"auto", "cpu", "cuda"}:
        raw = "auto"

    if raw == "cpu":
        device = "cpu"
    elif raw == "cuda":
        if not torch.cuda.is_available():
            if gpu_required:
                raise RuntimeError(
                    "GPU required (RAG_DEVICE=cuda, RAG_GPU_REQUIRED=1) but CUDA is not available"
                )
            device = "cpu"
        else:
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if gpu_required and device != "cuda":
        raise RuntimeError(
            "GPU required (RAG_GPU_REQUIRED=1) but CUDA is not available"
        )

    return DeviceConfig(device=device, gpu_required=gpu_required)


def torch_device(device: str) -> torch.device:
    return torch.device(device)
