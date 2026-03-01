"""
Runtime settings and path resolution for backend modules.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR_CANDIDATES: List[Path] = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "shared" / "models" / "models",
]
DATA_DIR_CANDIDATES: List[Path] = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "shared" / "data" / "data",
]


def _first_existing(paths: List[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def resolve_env_path(env_var: str, fallback_candidates: List[Path]) -> Path:
    """Resolve a path from environment with deterministic fallbacks."""
    raw = os.environ.get(env_var, "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate
    return _first_existing(fallback_candidates)


MODELS_DIR = _first_existing(MODEL_DIR_CANDIDATES)
DATA_DIR = _first_existing(DATA_DIR_CANDIDATES)

DEFAULT_GGUF_PATH = resolve_env_path(
    "RAG_GGUF_PATH",
    [
        MODELS_DIR
        / "qwen2_5_1_5b_instruct"
        / "gguf"
        / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    ],
)
DEFAULT_GENERATOR_MODEL_PATH = resolve_env_path(
    "RAG_GENERATOR_MODEL_PATH",
    [MODELS_DIR / "qwen", MODELS_DIR / "qwen2_5_1_5b_instruct"],
)
DEFAULT_EMBEDDING_MODEL_PATH = resolve_env_path(
    "RAG_EMBEDDING_MODEL_PATH",
    [MODELS_DIR / "bge-m3"],
)
DEFAULT_INDEX_DIR = resolve_env_path(
    "RAG_INDEX_DIR",
    [DATA_DIR / "indexes"],
)
DEFAULT_DATA_JSONL = resolve_env_path(
    "RAG_DATA_JSONL",
    [
        DATA_DIR / "sample_data.jsonl",
        DATA_DIR / "ds_ai_knowledge.jsonl",
    ],
)

BACKEND_HOST = os.environ.get("RAG_BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1"
BACKEND_PORT = int(os.environ.get("RAG_BACKEND_PORT", "8000"))
REQUEST_TIMEOUT_S = int(os.environ.get("RAG_REQUEST_TIMEOUT_S", "120"))
