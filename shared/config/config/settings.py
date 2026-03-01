"""
Example configuration values for deployment environments.

The active runtime settings are resolved in `src/backend/settings.py`.
This file is a static reference template only.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
SHARED_MODELS_DIR = PROJECT_ROOT / "shared" / "models" / "models"
DATA_DIR = PROJECT_ROOT / "data"
SHARED_DATA_DIR = PROJECT_ROOT / "shared" / "data" / "data"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501

GGUF_MODEL_PATH = MODELS_DIR / "qwen2_5_1_5b_instruct" / "gguf" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
EMBEDDING_MODEL_PATH = MODELS_DIR / "bge-m3"
INDEX_DIR = DATA_DIR / "indexes"

FALLBACK_GGUF_MODEL_PATH = SHARED_MODELS_DIR / "qwen2_5_1_5b_instruct" / "gguf" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
FALLBACK_EMBEDDING_MODEL_PATH = SHARED_MODELS_DIR / "bge-m3"
FALLBACK_INDEX_DIR = SHARED_DATA_DIR / "indexes"
