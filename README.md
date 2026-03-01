# RAG QnA Project

Local Retrieval-Augmented Generation (RAG) question-answering system with:

- FastAPI backend API
- Streamlit frontend UI
- Hybrid retrieval (dense vector + BM25)
- Local LLM inference via GGUF (llama.cpp backend path)

This repository is structured for local execution on Windows and can run fully offline after model and data assets are prepared.

## 1. Repository Structure

```text
.
|-- src/
|   |-- app.py                      # Streamlit frontend
|   `-- backend/
|       |-- server.py               # FastAPI app
|       |-- settings.py             # Path/env resolution
|       |-- ingest.py               # Build/search FAISS index
|       |-- hybrid_retriever.py     # Retrieval layer
|       |-- generator.py            # Generation layer
|       `-- reasoning/              # Optional reasoning pipeline
|-- models/                         # Local model artifacts
|-- shared/data/data/               # Data + prebuilt index fallback
|-- run_app.bat                     # Start backend + frontend
|-- run_local_inference.ps1         # Backend-only launch helper
|-- requirements.txt
|-- .env.example
`-- README.md                       # Single project README
```

## 2. Prerequisites

1. Windows 10/11
2. Python 3.10+ (3.11 or 3.12 recommended)
3. Optional NVIDIA GPU + CUDA (for better performance)
4. At least 16 GB RAM recommended
5. Sufficient disk space for model files (at least 8 GB free)

## 3. Setup

1. Create virtual environment:
```powershell
python -m venv .venv
```
2. Activate virtual environment:
```powershell
.venv\Scripts\activate
```
3. Install dependencies:
```powershell
pip install -r requirements.txt
```

## 4. Run The System

1. Start full application (recommended):
```bat
run_app.bat
```
2. Backend URL: `http://127.0.0.1:8000`
3. Frontend URL: `http://localhost:8501`

`run_app.bat` starts:

- FastAPI backend in one terminal
- Streamlit frontend in the current terminal

If `run_app.bat` closes immediately, run it from an open terminal to inspect the error:
```powershell
cmd /k run_app.bat
```

## 5. Large Model Files (Google Drive)

GitHub Free cannot store files larger than 2 GB, even with Git LFS.  
Two required files are intentionally excluded from git.

Google Drive folder for large files:

- <https://drive.google.com/drive/folders/1OvIWoob7eM1VU-Z8XDPzzOSyfm1pXYEO?usp=sharing>

Expected Drive contents and destination paths:

| File in Drive | Approx size | Destination in repo | Required |
|---|---:|---|---|
| `pytorch_model.bin` | 2.115 GB | `models/bge-m3/pytorch_model.bin` | Yes |
| `model.onnx_data` | 2.111 GB | `models/bge-m3/onnx/model.onnx_data` | Yes |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | 1.041 GB | `models/qwen2_5_1_5b_instruct/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf` | Yes |
| `adapter_model.safetensors` | 0.150 GB | `models/final_adapter/adapter_model.safetensors` | Optional |

Download or copy these files from Drive into the exact paths above before running inference.

### Verify required model files

```powershell
$required = @(
  "models/bge-m3/pytorch_model.bin",
  "models/bge-m3/onnx/model.onnx_data",
  "models/qwen2_5_1_5b_instruct/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
$required | ForEach-Object {
  if (Test-Path $_) { "OK  - $_" } else { "MISS- $_" }
}
```

## 6. Data and Index

Default data/index fallback path:

- `shared/data/data/`

Important files:

- `shared/data/data/sample_data.jsonl`
- `shared/data/data/ds_ai_knowledge.jsonl`
- `shared/data/data/indexes/faiss.index`
- `shared/data/data/indexes/chunks.jsonl`

Rebuild the FAISS index:

```powershell
python -m src.backend.ingest build
```

## 7. Environment Variables

Supported environment variables:

- `RAG_GGUF_PATH`
- `RAG_EMBEDDING_MODEL_PATH`
- `RAG_INDEX_DIR`
- `RAG_DATA_JSONL`
- `RAG_BACKEND_HOST`
- `RAG_BACKEND_PORT`
- `RAG_CORS_ORIGINS`
- `RAG_ADAPTER_PATH`
- `RAG_LLM_BACKEND`

If not set, `src/backend/settings.py` uses project defaults.

## 8. Backend API

Main endpoints:

- `GET /healthz`
- `POST /ask`
- `POST /reload_generator`

Example health check:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/healthz | Select-Object -Expand Content
```

Example ask request:

```powershell
$body = @{
  question = "What is retrieval augmented generation?"
  top_k = 3
  mode = "hybrid"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## 9. Validation Checklist

1. Python import/syntax check:
```powershell
python -m compileall -q src
```
2. Backend startup check:
```powershell
python -m uvicorn src.backend.server:app --host 127.0.0.1 --port 8000
```
3. Open `http://127.0.0.1:8000/healthz` and confirm `"status":"ok"`
4. Run `run_app.bat` and open frontend page

## 10. License

This repository is licensed under the MIT License. See `LICENSE`.
