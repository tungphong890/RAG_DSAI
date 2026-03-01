# RAG QnA Project

Retrieval-augmented QnA system with:

- FastAPI backend (`src/backend/server.py`)
- Streamlit frontend (`src/app.py`)
- Local hybrid retrieval (vector + BM25)
- Local generation support (llama.cpp and fallback path)

## Project Layout

```
src/
  app.py
  backend/
    server.py
    generator.py
    hybrid_retriever.py
    ingest.py
    reasoning/
run_app.bat
run_local_inference.ps1
models/
shared/
```

## Quick Start

1. Create environment:
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run app:
   - `run_app.bat`

Backend default URL: `http://127.0.0.1:8000`  
Frontend default URL: `http://localhost:8501`

## Configuration

Use environment variables if you need custom paths:

- `RAG_GGUF_PATH`
- `RAG_EMBEDDING_MODEL_PATH`
- `RAG_INDEX_DIR`
- `RAG_DATA_JSONL`
- `RAG_BACKEND_HOST`
- `RAG_BACKEND_PORT`
- `RAG_CORS_ORIGINS`

See `.env.example` and `shared/docs/DEPLOYMENT.md` for deployment details.

## GitHub Free Note

GitHub Free cannot store files larger than 2 GB, even with Git LFS.
This repository therefore excludes these four files from git tracking:

- `models/bge-m3/pytorch_model.bin`
- `models/bge-m3/onnx/model.onnx_data`
- `shared/models/models/bge-m3/pytorch_model.bin`
- `shared/models/models/bge-m3/onnx/model.onnx_data`
