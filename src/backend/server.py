"""
RAG backend API server.

Provides retrieval-augmented generation endpoints and optional reasoning mode.
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from .device_manager import get_device_config
    from .generator import FallbackExtractorGenerator, LlamaCppGenerator
    from .hybrid_retriever import HybridRetriever
    from .reasoning.pipeline import ReasoningPipeline
    from .settings import (
        BACKEND_HOST,
        BACKEND_PORT,
        DEFAULT_EMBEDDING_MODEL_PATH,
        DEFAULT_GGUF_PATH,
        DEFAULT_INDEX_DIR,
        PROJECT_ROOT,
    )
    from .tools import web_search
except ImportError:
    from device_manager import get_device_config
    from generator import FallbackExtractorGenerator, LlamaCppGenerator
    from hybrid_retriever import HybridRetriever
    from reasoning.pipeline import ReasoningPipeline
    from settings import (
        BACKEND_HOST,
        BACKEND_PORT,
        DEFAULT_EMBEDDING_MODEL_PATH,
        DEFAULT_GGUF_PATH,
        DEFAULT_INDEX_DIR,
        PROJECT_ROOT,
    )
    from tools import web_search

try:
    import torch
except Exception:
    torch = None                


_generator: Optional[Any] = None
_hybrid_retriever: Optional[HybridRetriever] = None
_reasoning: Optional[ReasoningPipeline] = None
_startup_error: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    mode: str = "hybrid"
    reasoning: Optional[bool] = None
    continue_request: Optional[bool] = False
    section_index: Optional[int] = None


app = FastAPI(title="Local RAG QA with Web Search", version="0.3.0")


cors_origins = [
    origin.strip()
    for origin in os.environ.get("RAG_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_path(raw_path: str, fallback_path: Path) -> Path:
    value = (raw_path or "").strip()
    if value:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        return candidate
    return fallback_path


def _resolve_paths() -> Dict[str, Path]:
    return {
        "gguf_path": _resolve_path(
            os.environ.get("RAG_GGUF_PATH", ""), DEFAULT_GGUF_PATH
        ),
        "embedding_model_path": _resolve_path(
            os.environ.get("RAG_EMBEDDING_MODEL_PATH", ""),
            DEFAULT_EMBEDDING_MODEL_PATH,
        ),
        "index_dir": _resolve_path(
            os.environ.get("RAG_INDEX_DIR", ""), DEFAULT_INDEX_DIR
        ),
    }


def _log_gpu_memory(event: str) -> None:
    if torch is None:
        return
    if not torch.cuda.is_available():
        return

    try:
        logs_dir = PROJECT_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.time(),
            "event": event,
            "reserved": int(torch.cuda.memory_reserved()),
            "allocated": int(torch.cuda.memory_allocated()),
        }
        with (logs_dir / "gpu_memory.log").open("a", encoding="utf-8") as f:
            f.write(f"{payload}\n")
    except Exception:
        pass


def reload_generator() -> Dict[str, Any]:
    """Hot reload the generator without restarting the API."""
    global _generator, _reasoning

    try:
        paths = _resolve_paths()
        gguf_path = paths["gguf_path"]
        if not gguf_path.exists():
            raise FileNotFoundError(f"GGUF model not found: {gguf_path}")

        device_cfg = get_device_config()
        _generator = LlamaCppGenerator(
            gguf_path=str(gguf_path),
            device=device_cfg.device,
            max_new_tokens=256,
            n_gpu_layers=28,
            n_ctx=2048,
            allow_ctx_fallback=True,
        )

        if _hybrid_retriever is not None:
            _reasoning = ReasoningPipeline(
                retriever=_hybrid_retriever, generator=_generator
            )

        return {
            "status": "success",
            "message": "Generator reloaded successfully",
            "gguf_path": str(gguf_path),
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to reload generator: {e}",
        }


@app.on_event("startup")
def _startup() -> None:
    global _generator, _hybrid_retriever, _reasoning, _startup_error

    print("[startup] Initializing RAG system...")
    try:
        _startup_error = None
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        paths = _resolve_paths()
        device_cfg = get_device_config()
        print(f"[startup] Device config: {device_cfg}")
        print(f"[startup] Index dir: {paths['index_dir']}")
        print(f"[startup] Embedding model: {paths['embedding_model_path']}")
        print(f"[startup] GGUF model: {paths['gguf_path']}")

        _log_gpu_memory("startup_begin")
        _hybrid_retriever = HybridRetriever(
            index_dir=str(paths["index_dir"]),
            embedding_model_path=str(paths["embedding_model_path"]),
            device=device_cfg.device,
            use_reranker=os.environ.get("RAG_USE_RERANKER", "0").strip()
            in {"1", "true", "True"},
        )

        if paths["gguf_path"].exists():
            try:
                _generator = LlamaCppGenerator(
                    gguf_path=str(paths["gguf_path"]),
                    device=device_cfg.device,
                    max_new_tokens=256,
                    n_gpu_layers=28,
                    n_ctx=2048,
                    allow_ctx_fallback=True,
                )
            except Exception as exc:
                print(f"[startup] GGUF load failed, using fallback generator: {exc}")
                _generator = FallbackExtractorGenerator()
        else:
            print("[startup] GGUF model not found, using fallback generator")
            _generator = FallbackExtractorGenerator()

        _reasoning = ReasoningPipeline(
            retriever=_hybrid_retriever, generator=_generator
        )
        _log_gpu_memory("startup_done")
        print("[startup] RAG system initialized successfully")
    except Exception as e:
        _generator = None
        _hybrid_retriever = None
        _reasoning = None
        _startup_error = str(e)
        print(f"[startup] Initialization failed: {e}")


def _not_ready_response(
    start_time: float, mode: str, top_k: int, error_message: str
) -> Dict[str, Any]:
    message = error_message or "System is still initializing. Please try again shortly."
    return {
        "status": "success",
        "answer": message,
        "processing_time": time.time() - start_time,
        "mode": mode,
        "top_k": max(1, top_k),
        "citations": [],
        "sources": [],
        "confidence": None,
    }


@app.post("/ask")
async def ask(req: AskRequest) -> Dict[str, Any]:
    global _generator, _hybrid_retriever, _reasoning

    start_time = time.time()
    try:
        mode = (req.mode or "hybrid").strip().lower()
        if mode not in {"local", "online", "hybrid"}:
            mode = "hybrid"

        if _generator is None or _hybrid_retriever is None:
            if mode in {"online", "hybrid"}:
                try:
                    online_answer = str(
                        web_search(req.question, max_results=max(1, req.top_k))
                    ).strip()
                    if online_answer:
                        return {
                            "status": "success",
                            "answer": online_answer,
                            "processing_time": time.time() - start_time,
                            "mode": mode,
                            "top_k": max(1, req.top_k),
                            "citations": [],
                            "sources": [],
                            "confidence": None,
                        }
                except Exception:
                    pass

            return _not_ready_response(
                start_time=start_time,
                mode=mode,
                top_k=req.top_k,
                error_message=_startup_error or "",
            )

        response_text = ""
        citations: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        confidence: Optional[float] = None

        if req.reasoning and _reasoning is not None and mode != "online":
            reasoning_result = _reasoning.run(req.question)
            reasoning_payload = reasoning_result.get("result", {})
            response_text = str(reasoning_payload.get("answer", "")).strip()
            confidence = float(reasoning_payload.get("confidence", 0.0) or 0.0)
            sources = list(reasoning_payload.get("snippets", []))
        else:
            retrieved_docs: List[Dict[str, Any]] = []
            if mode in {"local", "hybrid"}:
                retrieved_docs = _hybrid_retriever.search(
                    query=req.question,
                    top_k=max(1, req.top_k),
                )

            if retrieved_docs:
                gen_out = _generator.answer(
                    question=req.question, contexts=retrieved_docs
                )
                response_text = str(gen_out.get("answer", "")).strip()
                citations = list(gen_out.get("citations", []))
                sources = list(gen_out.get("sources", []))
                if gen_out.get("confidence") is not None:
                    confidence = float(gen_out.get("confidence"))

            if (not response_text) and mode in {"online", "hybrid"}:
                response_text = str(
                    web_search(req.question, max_results=max(1, req.top_k))
                )

        if not response_text:
            response_text = (
                "I could not find enough information to answer that question."
            )

        return {
            "status": "success",
            "answer": response_text,
            "processing_time": time.time() - start_time,
            "mode": mode,
            "top_k": max(1, req.top_k),
            "citations": citations,
            "sources": sources,
            "confidence": confidence,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "processing_time": time.time() - start_time,
        }


@app.post("/reload_generator")
def reload_generator_endpoint() -> Dict[str, Any]:
    return reload_generator()


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {
        "status": "ok",
        "ready": _generator is not None and _hybrid_retriever is not None,
        "generator_loaded": _generator is not None,
        "retriever_loaded": _hybrid_retriever is not None,
    }


@app.get("/")
async def root() -> HTMLResponse:
    return HTMLResponse("""
    <html>
      <head><title>RAG Backend</title></head>
      <body>
        <h1>RAG Backend Server</h1>
        <p>Server is running.</p>
        <ul>
          <li>POST /ask</li>
          <li>POST /reload_generator</li>
        </ul>
      </body>
    </html>
    """)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT, log_level="info")
