# Deployment Checklist

## 1. Environment

- Python 3.10+ available
- GPU drivers/CUDA installed if GPU inference is required
- Virtual environment created:
  - `python -m venv .venv`
  - `.venv\Scripts\activate`
- Dependencies installed:
  - `pip install -r requirements.txt`

## 2. Runtime Paths

The backend resolves paths in this order:

- Environment variable value (if provided)
- Root project path (`models/`, `data/`)
- Shared fallback path (`shared/models/models/`, `shared/data/data/`)

Optional environment variables:

- `RAG_GGUF_PATH`
- `RAG_EMBEDDING_MODEL_PATH`
- `RAG_INDEX_DIR`
- `RAG_DATA_JSONL`
- `RAG_BACKEND_HOST`
- `RAG_BACKEND_PORT`
- `RAG_CORS_ORIGINS`

## 3. Launch

- Full app:
  - `run_app.bat`
- Backend only:
  - `python -m uvicorn src.backend.server:app --host 127.0.0.1 --port 8000`
- Frontend only:
  - `python -m streamlit run src/app.py`

## 4. Validation

- Health endpoint: `GET http://127.0.0.1:8000/healthz`
- Ask endpoint: `POST http://127.0.0.1:8000/ask`
- Frontend endpoint: `http://localhost:8501`

## 5. Pre-Release Checks

- `python -m compileall -q src`
- Verify model/index paths exist for target environment
- Run one real query through frontend and direct API
