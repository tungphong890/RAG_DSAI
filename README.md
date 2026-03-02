# MochiChatbot RAG Application

MochiChatbot is a local Retrieval-Augmented Generation (RAG) desktop app. It runs a FastAPI backend and Streamlit UI behind a native desktop window, with background ingestion for new PDFs and links.

## What This Project Does

- Runs as a desktop app (`MochiChatbot.exe`) instead of opening a browser.
- Starts backend and UI services in the background.
- Watches for new content and updates the vector index automatically.
- Keeps runtime files in `app_data/` (logs, queue, processed, failed, index data).

## Current Project Structure

```text
.
|-- assets/
|   `-- app.ico
|-- dist/
|   `-- MochiChatbot.exe
|-- models/
|   |-- bge-m3/
|   `-- qwen2_5_1_5b_instruct/
|-- shared/
|   `-- data/data/
|       |-- ds_ai_knowledge.jsonl
|       |-- sample_data.jsonl
|       `-- indexes/
|-- src/
|   |-- app.py
|   `-- backend/
|       |-- server.py
|       |-- ingest.py
|       |-- hybrid_retriever.py
|       |-- generator.py
|       |-- settings.py
|       |-- tools.py
|       |-- device_manager.py
|       |-- device_utils.py
|       `-- reasoning/
|-- .env.example
|-- LICENSE
|-- requirements.txt
|-- run_app.bat
|-- run_app_launcher.py
|-- run_app_launcher.spec
`-- README.md
```

## Runtime Layout (Created Automatically)

When you run the launcher (source or EXE), it creates:

```text
app_data/
|-- incoming/                  # drop new PDF files here
|-- processed/                 # successfully ingested files
|-- failed/                    # failed files/links
|-- data/
|   `-- auto_ingest.jsonl      # launcher-managed ingestion corpus
|-- indexes/                   # index output target for launcher rebuild flow
|-- logs/
|   |-- launcher.log
|   |-- backend.log
|   |-- streamlit_stdout.log
|   |-- streamlit_stderr.log
|   |-- processed_links.json
|   `-- failed_links.json
|-- docs/
|   `-- manual_review_required.md
`-- EDIT_NOTES_FOR_OWNER.md
```

## Requirements

- Windows 10/11 (primary target for EXE)
- Python 3.10+
- Local model files present under `models/`

Install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pywebview requests uvicorn
```

## Run Options

### Option A: Desktop launcher from source

```powershell
python run_app_launcher.py
```

Expected behavior:

- One native app window opens.
- Backend + Streamlit run in hidden background processes.
- Closing the window stops child services.

### Option B: Prebuilt EXE

```powershell
.\dist\MochiChatbot.exe
```

If startup is slow, check logs under `app_data/logs/`.

### Option C: Dev mode (legacy local web flow)

```bat
run_app.bat
```

This starts backend and Streamlit for local development.

## Auto Ingestion Workflow

The launcher watcher runs on a fixed interval and checks:

- `app_data/incoming/*.pdf`
- `app_data/incoming_links.txt` (or JSON links input if configured)

For each cycle:

1. Detect new PDFs and links.
2. Extract text and append normalized records into `app_data/data/auto_ingest.jsonl`.
3. Trigger index rebuild via `src.backend.ingest` (import call first, subprocess fallback).
4. Move successful files to `processed/`.
5. Move failures to `failed/` and append details to `app_data/docs/manual_review_required.md`.

## Model Files

Place local model artifacts in `models/`.

Expected paths used by launcher/backend:

- `models/bge-m3/`
- `models/qwen2_5_1_5b_instruct/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf`

If models are missing, startup and answers can fail or fall back to reduced behavior.

## Environment Variables

Defaults are defined in code and `.env.example`.

Most relevant variables:

- `RAG_BACKEND_HOST`
- `RAG_BACKEND_PORT`
- `RAG_CORS_ORIGINS`
- `RAG_BACKEND_URL`
- `RAG_REQUEST_TIMEOUT_S`
- `RAG_GGUF_PATH`
- `RAG_EMBEDDING_MODEL_PATH`
- `RAG_INDEX_DIR`
- `RAG_DATA_JSONL`

## Build EXE

Build with PyInstaller spec:

```powershell
pyinstaller run_app_launcher.spec
```

Output:

- `dist/MochiChatbot.exe`

## Troubleshooting

### App does not open

- Check `app_data/logs/launcher.log`.
- Check `app_data/logs/streamlit_stderr.log`.
- Confirm model paths exist.

### "System not fully initialized"

- Backend is still loading models or failed startup.
- Review `app_data/logs/backend.log`.

### Ingestion not updating

- Confirm PDFs are in `app_data/incoming/`.
- Confirm links file exists at `app_data/incoming_links.txt`.
- Review `manual_review_required.md` for failures.

## API Endpoints

When backend is running:

- `GET /healthz`
- `POST /ask`
- `POST /reload_generator`

Quick check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/healthz
```

## License

MIT License. See `LICENSE`.
