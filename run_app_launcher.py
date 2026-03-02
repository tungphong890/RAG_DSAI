"""
Purpose:
This launcher runs the RAG application as a desktop-style app by starting FastAPI and
Streamlit in background services and opening a native window with pywebview.

Public API:
- run(): starts the launcher lifecycle.

Inputs and outputs:
- Inputs are launcher configuration constants and optional runtime inputs in app_data:
  new PDF files under incoming/ and links under incoming_links.txt or incoming_links.json.
- Output is a native window plus runtime artifacts written under app_data/.

Side effects:
- Starts backend and Streamlit services.
- Extracts PDF/link content and updates a launcher-managed JSONL corpus.
- Rebuilds the FAISS index through src.backend.ingest import or CLI fallback.
- Writes logs, moves processed/failed items, and writes review/session notes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, TextIO

import requests
import uvicorn


UVICORN_MODULE = "src.backend.server:app"
STREAMLIT_FILE = "src/app.py"

STREAMLIT_HOST = "127.0.0.1"
STREAMLIT_PORT = 8501

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000

CHECK_INTERVAL_SECONDS = 30

WINDOW_TITLE = "MochiChatbot"

STREAMLIT_STARTUP_TIMEOUT_SECONDS = 180
STREAMLIT_HEALTH_POLL_SECONDS = 2
INGEST_SUBPROCESS_TIMEOUT_SECONDS = 1800

CREATE_NO_WINDOW = 0x08000000
CHILD_MODE_STREAMLIT = "--child-streamlit"
CHILD_MODE_INGEST = "--child-ingest"

URL_FETCH_RETRIES = 3
URL_FETCH_TIMEOUT_SECONDS = 20


def is_frozen() -> bool:
    """Return True when running from a frozen executable."""
    return bool(getattr(sys, "frozen", False))


def executable_root() -> Path:
    """Return runtime base directory beside executable or repository root."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def bundle_root() -> Path:
    """Return bundled resource root for frozen mode or repository root."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", executable_root()))
    return Path(__file__).resolve().parent


def resource_path(*parts: str) -> Path:
    """Return resource path using sys._MEIPASS fallback in frozen mode."""
    return bundle_root().joinpath(*parts)


def app_home_path(*parts: str) -> Path:
    """Return writable app_data path beside executable or source root."""
    return executable_root().joinpath("app_data", *parts)


REPO_ROOT = Path(__file__).resolve().parent
APP_HOME = app_home_path()
SINGLE_INSTANCE_LOCK = app_home_path("run_app_launcher.lock")

INCOMING_DIR = app_home_path("incoming")
PROCESSED_DIR = app_home_path("processed")
FAILED_DIR = app_home_path("failed")
LOG_DIR = app_home_path("logs")
DOCS_DIR = app_home_path("docs")
DATA_DIR = app_home_path("data")
INDEX_DIR = app_home_path("indexes")

LINKS_FILE = app_home_path("incoming_links.txt")
AUTO_JSONL_PATH = app_home_path("data", "auto_ingest.jsonl")
PROCESSED_LINKS_FILE = app_home_path("logs", "processed_links.json")
FAILED_LINKS_FILE = app_home_path("logs", "failed_links.json")
MANUAL_REVIEW_FILE = app_home_path("docs", "manual_review_required.md")
EDIT_NOTES_FILE = app_home_path("EDIT_NOTES_FOR_OWNER.md")

EXTERNAL_MODELS_DIR = executable_root() / "models"
PARENT_MODELS_DIR = executable_root().parent / "models"
BUNDLED_MODELS_DIR = resource_path("models")


def resolve_models_dir() -> Path:
    """Resolve models directory from external or bundled candidates."""
    for candidate in (EXTERNAL_MODELS_DIR, PARENT_MODELS_DIR, BUNDLED_MODELS_DIR):
        if candidate.exists():
            return candidate
    return EXTERNAL_MODELS_DIR


MODELS_DIR = resolve_models_dir()
EMBEDDING_MODEL_PATH = MODELS_DIR / "bge-m3"
GGUF_MODEL_PATH = (
    MODELS_DIR
    / "qwen2_5_1_5b_instruct"
    / "gguf"
    / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)


@dataclass
class ManagedSubprocess:
    """Store subprocess and log handles for clean shutdown."""

    process: subprocess.Popen[Any]
    stdout_handle: TextIO
    stderr_handle: TextIO


@dataclass
class SessionSummary:
    """Track processed items and failures for session notes."""

    processed_files: list[str] = field(default_factory=list)
    processed_links: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


@dataclass
class PendingBatch:
    """Represent extracted pending content that awaits index rebuild."""

    records: list[dict[str, Any]] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


class SessionRecorder:
    """Thread-safe recorder for session activity."""

    def __init__(self) -> None:
        self._summary = SessionSummary()
        self._lock = threading.Lock()

    def add_processed_file(self, path: Path) -> None:
        """Record a successfully processed file."""
        with self._lock:
            self._summary.processed_files.append(str(path))

    def add_processed_link(self, link: str) -> None:
        """Record a successfully processed link."""
        with self._lock:
            self._summary.processed_links.append(link)

    def add_failure(self, item: str, error: str) -> None:
        """Record a processing failure."""
        with self._lock:
            self._summary.failures.append(f"{item}: {error}")

    def snapshot(self) -> SessionSummary:
        """Return a copy of session summary values."""
        with self._lock:
            return SessionSummary(
                processed_files=list(self._summary.processed_files),
                processed_links=list(self._summary.processed_links),
                failures=list(self._summary.failures),
            )


def ensure_runtime_paths() -> None:
    """Create runtime writable directories."""
    for path in (
        APP_HOME,
        INCOMING_DIR,
        PROCESSED_DIR,
        FAILED_DIR,
        LOG_DIR,
        DOCS_DIR,
        DATA_DIR,
        INDEX_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def acquire_single_instance_lock() -> Optional[TextIO]:
    """Acquire a non-blocking file lock to prevent duplicate launcher instances."""
    SINGLE_INSTANCE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = SINGLE_INSTANCE_LOCK.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        handle.close()
        return None

    handle.seek(0)
    handle.truncate(0)
    handle.write(f"pid={os.getpid()} ts={now_utc()}\n")
    handle.flush()
    return handle


def release_single_instance_lock(handle: Optional[TextIO]) -> None:
    """Release file lock and close handle."""
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    finally:
        try:
            handle.close()
        except Exception:
            pass


def configure_logging() -> logging.Logger:
    """Configure launcher logger writing to app_data/logs."""
    logger = logging.getLogger("run_app_launcher")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s")

    info_handler = logging.FileHandler(LOG_DIR / "launcher.log", mode="a", encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    err_handler = logging.FileHandler(
        LOG_DIR / "launcher_error.log", mode="a", encoding="utf-8"
    )
    err_handler.setLevel(logging.ERROR)
    err_handler.setFormatter(formatter)
    logger.addHandler(err_handler)

    return logger


def now_utc() -> str:
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def configure_runtime_environment(logger: logging.Logger) -> None:
    """Configure environment defaults consumed by backend and ingest."""
    for candidate in (str(bundle_root()), str(executable_root())):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    python_path = os.environ.get("PYTHONPATH", "")
    python_path_parts = [str(bundle_root()), str(executable_root())]
    if python_path:
        python_path_parts.append(python_path)
    os.environ["PYTHONPATH"] = os.pathsep.join(python_path_parts)

    os.environ.setdefault("RAG_INDEX_DIR", str(INDEX_DIR))
    os.environ.setdefault("RAG_DATA_JSONL", str(AUTO_JSONL_PATH))
    if EMBEDDING_MODEL_PATH.exists():
        os.environ.setdefault("RAG_EMBEDDING_MODEL_PATH", str(EMBEDDING_MODEL_PATH))
    if GGUF_MODEL_PATH.exists():
        os.environ.setdefault("RAG_GGUF_PATH", str(GGUF_MODEL_PATH))

    logger.info("runtime: frozen=%s", is_frozen())
    logger.info("runtime: app_home=%s", APP_HOME)
    logger.info("runtime: models_dir=%s exists=%s", MODELS_DIR, MODELS_DIR.exists())
    logger.info(
        "runtime: embedding_model=%s exists=%s",
        EMBEDDING_MODEL_PATH,
        EMBEDDING_MODEL_PATH.exists(),
    )
    logger.info("runtime: index_dir=%s", INDEX_DIR)
    logger.info("runtime: data_jsonl=%s", AUTO_JSONL_PATH)


def build_uvicorn_log_config() -> dict[str, Any]:
    """Return uvicorn log config writing to launcher log files."""
    backend_log = str((LOG_DIR / "backend.log").resolve())
    backend_access_log = str((LOG_DIR / "backend_access.log").resolve())
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s [%(levelname)s] %(name)s %(message)s",
            },
            "access": {
                "format": "%(asctime)s [%(levelname)s] %(name)s %(message)s",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.FileHandler",
                "filename": backend_log,
                "mode": "a",
                "formatter": "default",
            },
            "access": {
                "class": "logging.FileHandler",
                "filename": backend_access_log,
                "mode": "a",
                "formatter": "access",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }


def resolve_backend_target(logger: logging.Logger) -> Any:
    """Resolve backend ASGI app object from UVICORN_MODULE with fallback."""
    try:
        module_name, app_name = UVICORN_MODULE.split(":", 1)
        module = importlib.import_module(module_name)
        app_obj = getattr(module, app_name)
        logger.info("backend: resolved app object from %s", UVICORN_MODULE)
        return app_obj
    except Exception:
                                                                                     
        logger.exception("backend: failed to import %s, using module string fallback", UVICORN_MODULE)
        return UVICORN_MODULE


def run_backend_server(logger: logging.Logger) -> None:
    """Run backend uvicorn server in current thread."""
    try:
        app_target = resolve_backend_target(logger)
        logger.info(
            "backend: starting uvicorn target=%s host=%s port=%s",
            UVICORN_MODULE,
            BACKEND_HOST,
            BACKEND_PORT,
        )
        uvicorn.run(
            app_target,
            host=BACKEND_HOST,
            port=BACKEND_PORT,
            log_level="info",
            log_config=build_uvicorn_log_config(),
            access_log=True,
        )
    except Exception:
        logger.exception("backend: uvicorn.run failed")


def start_backend_thread(logger: logging.Logger) -> threading.Thread:
    """Start backend server in a daemon thread."""
    thread = threading.Thread(
        target=run_backend_server,
        args=(logger,),
        daemon=True,
        name="backend-uvicorn-thread",
    )
    thread.start()
    return thread


def hidden_popen_kwargs(stdout_handle: TextIO, stderr_handle: TextIO) -> dict[str, Any]:
    """Return platform-specific hidden subprocess parameters."""
    kwargs: dict[str, Any] = {
        "cwd": str(executable_root()),
        "stdin": subprocess.DEVNULL,
        "stdout": stdout_handle,
        "stderr": stderr_handle,
    }
    if os.name == "nt":
        kwargs["creationflags"] = CREATE_NO_WINDOW
    else:
        kwargs["preexec_fn"] = os.setsid
    return kwargs


def start_streamlit_process(logger: logging.Logger) -> ManagedSubprocess:
    """Start streamlit in hidden subprocess with logs redirected."""
    stdout_handle = (LOG_DIR / "streamlit_stdout.log").open("a", encoding="utf-8")
    stderr_handle = (LOG_DIR / "streamlit_stderr.log").open("a", encoding="utf-8")

    streamlit_target = (executable_root() / STREAMLIT_FILE).resolve()
    if not streamlit_target.exists():
        candidates = [
            (executable_root().parent / STREAMLIT_FILE).resolve(),
            resource_path(*Path(STREAMLIT_FILE).parts).resolve(),
            resource_path(STREAMLIT_FILE).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                streamlit_target = candidate
                break

    streamlit_args = [
        str(streamlit_target),
        "--server.headless",
        "true",
        "--server.address",
        STREAMLIT_HOST,
        "--server.port",
        str(STREAMLIT_PORT),
        "--global.developmentMode",
        "false",
        "--server.fileWatcherType",
        "none",
        "--server.runOnSave",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]
    if is_frozen():
                                                                                          
        cmd = [sys.executable, CHILD_MODE_STREAMLIT, *streamlit_args]
    else:
        cmd = [sys.executable, "-m", "streamlit", "run", *streamlit_args]

    env = os.environ.copy()
    env.setdefault("RAG_INDEX_DIR", str(INDEX_DIR))
    env.setdefault("RAG_DATA_JSONL", str(AUTO_JSONL_PATH))
    env.setdefault("RAG_EMBEDDING_MODEL_PATH", str(EMBEDDING_MODEL_PATH))
    env.setdefault("RAG_GGUF_PATH", str(GGUF_MODEL_PATH))

    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            **hidden_popen_kwargs(stdout_handle, stderr_handle),
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        logger.exception("streamlit: failed to start hidden subprocess")
        raise

    logger.info(
        "streamlit: started pid=%s url=http://%s:%s",
        process.pid,
        STREAMLIT_HOST,
        STREAMLIT_PORT,
    )
    return ManagedSubprocess(process=process, stdout_handle=stdout_handle, stderr_handle=stderr_handle)


def terminate_process_group(process: subprocess.Popen[Any], logger: logging.Logger) -> None:
    """Terminate subprocess and process group with timeout and kill fallback."""
    if process.poll() is not None:
        return

    if os.name == "nt":
        process.terminate()
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            process.terminate()

    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        logger.warning("process: forcing kill pid=%s", process.pid)
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                process.kill()
        process.wait(timeout=10)


def close_managed_subprocess(
    managed: Optional[ManagedSubprocess], name: str, logger: logging.Logger
) -> None:
    """Terminate managed subprocess and close associated log handles."""
    if managed is None:
        return

    try:
        logger.info("%s: stopping pid=%s", name, managed.process.pid)
        terminate_process_group(managed.process, logger)
    except Exception:
        logger.exception("%s: shutdown failed", name)
    finally:
        for handle in (managed.stdout_handle, managed.stderr_handle):
            try:
                handle.flush()
            except Exception:
                pass
            try:
                handle.close()
            except Exception:
                pass


def wait_for_streamlit_ready(url: str, timeout_seconds: int, logger: logging.Logger) -> bool:
    """Poll streamlit URL until HTTP 200 or timeout."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info("healthcheck: streamlit ready at %s", url)
                return True
        except requests.RequestException:
            pass
        time.sleep(STREAMLIT_HEALTH_POLL_SECONDS)
    logger.warning("healthcheck: streamlit not ready within %ss", timeout_seconds)
    return False


def build_not_ready_html(target_url: str) -> str:
    """Return fallback HTML page when streamlit startup times out."""
    return f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Application Not Ready</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; }}
        code {{ background: #f3f3f3; padding: 2px 6px; }}
      </style>
    </head>
    <body>
      <h2>Application is starting</h2>
      <p>Streamlit is not ready yet. Logs are available under <code>app_data/logs/</code>.</p>
      <p>Expected UI URL: <code>{target_url}</code></p>
      <p>Close and relaunch if startup does not complete.</p>
    </body>
    </html>
    """


def open_native_window(streamlit_url: str, is_ready: bool, logger: logging.Logger) -> None:
    """Open pywebview native window and block until close."""
    try:
        import webview
    except Exception:
        logger.exception("window: pywebview import failed")
        raise

    if is_ready:
        webview.create_window(WINDOW_TITLE, url=streamlit_url, text_select=True)
    else:
        webview.create_window(
            WINDOW_TITLE,
            html=build_not_ready_html(streamlit_url),
            text_select=True,
        )
        logger.warning("window: opened fallback not-ready page")

    logger.info("window: entering event loop")
    webview.start(debug=False)
    logger.info("window: closed by user")


def load_json_set(path: Path, logger: logging.Logger) -> set[str]:
    """Load JSON list file into a string set."""
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("state: failed loading %s", path)
        return set()
    if not isinstance(payload, list):
        return set()
    return {str(item).strip() for item in payload if str(item).strip()}


def save_json_set(path: Path, values: Iterable[str], logger: logging.Logger) -> None:
    """Save string iterable to JSON list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sorted_values = sorted({value.strip() for value in values if value.strip()})
        path.write_text(
            json.dumps(sorted_values, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("state: failed writing %s", path)


def read_links(logger: logging.Logger) -> list[str]:
    """Read links from incoming_links.txt or incoming_links.json."""
    txt_path = LINKS_FILE
    json_path = LINKS_FILE.with_suffix(".json")

    candidate: Optional[Path] = None
    if txt_path.exists():
        candidate = txt_path
    elif json_path.exists():
        candidate = json_path

    if candidate is None:
        return []

    try:
        raw = candidate.read_text(encoding="utf-8").strip()
    except Exception:
        logger.exception("watcher: failed reading links file %s", candidate)
        return []

    if not raw:
        return []

    links: list[str] = []
    if candidate.suffix.lower() == ".json":
        try:
            payload = json.loads(raw)
            if isinstance(payload, list):
                links = [str(item).strip() for item in payload]
            elif isinstance(payload, dict):
                maybe_links = payload.get("links") or payload.get("urls") or []
                if isinstance(maybe_links, list):
                    links = [str(item).strip() for item in maybe_links]
        except Exception:
            logger.exception("watcher: invalid JSON links file %s", candidate)
            return []
    else:
        links = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    deduped: list[str] = []
    seen: set[str] = set()
    for link in links:
        if link and link not in seen:
            seen.add(link)
            deduped.append(link)
    return deduped


def append_manual_review(item: str, error_message: str, stack_trace: str, logger: logging.Logger) -> None:
    """Append failure entry to docs/manual_review_required.md."""
    try:
        MANUAL_REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
        if not MANUAL_REVIEW_FILE.exists():
            MANUAL_REVIEW_FILE.write_text(
                "# Manual Review Required\n\n"
                "This file is maintained by run_app_launcher.py when ingestion fails.\n\n",
                encoding="utf-8",
            )
        entry = (
            f"## {now_utc()}\n"
            f"- Item: `{item}`\n"
            f"- Error: {error_message}\n\n"
            f"```text\n{stack_trace}\n```\n\n"
        )
        with MANUAL_REVIEW_FILE.open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except Exception:
        logger.exception("manual_review: failed appending entry for %s", item)


def unique_destination(source: Path, destination_dir: Path) -> Path:
    """Return unique destination path preserving filename semantics."""
    candidate = destination_dir / source.name
    if not candidate.exists():
        return candidate
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return destination_dir / f"{source.stem}_{stamp}{source.suffix}"


def move_file_atomic(source: Path, destination_dir: Path) -> Path:
    """Move file atomically using os.replace into destination directory."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = unique_destination(source, destination_dir)
    os.replace(str(source), str(target))
    return target


def sha1_text(value: str) -> str:
    """Return short SHA1 digest for deterministic IDs."""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def bootstrap_auto_jsonl(logger: logging.Logger) -> None:
    """Create managed JSONL from available seed files if missing."""
    if AUTO_JSONL_PATH.exists() and AUTO_JSONL_PATH.stat().st_size > 0:
        return

    candidate_files = [
        executable_root() / "shared" / "data" / "data" / "ds_ai_knowledge.jsonl",
        executable_root() / "shared" / "data" / "data" / "sample_data.jsonl",
        executable_root().parent / "shared" / "data" / "data" / "ds_ai_knowledge.jsonl",
        executable_root().parent / "shared" / "data" / "data" / "sample_data.jsonl",
        REPO_ROOT / "shared" / "data" / "data" / "ds_ai_knowledge.jsonl",
        REPO_ROOT / "shared" / "data" / "data" / "sample_data.jsonl",
        resource_path("shared", "data", "data", "ds_ai_knowledge.jsonl"),
        resource_path("shared", "data", "data", "sample_data.jsonl"),
    ]

    seen_lines: set[str] = set()
    merged_lines: list[str] = []
    for file_path in candidate_files:
        if not file_path.exists():
            continue
        try:
            for line in file_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line in seen_lines:
                    continue
                seen_lines.add(line)
                merged_lines.append(line)
        except Exception:
            logger.exception("bootstrap: failed reading %s", file_path)

    AUTO_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if merged_lines:
        AUTO_JSONL_PATH.write_text("\n".join(merged_lines) + "\n", encoding="utf-8")
        logger.info("bootstrap: initialized auto JSONL with %s lines", len(merged_lines))
        return

    bootstrap_record = {
        "id": "bootstrap_local_rag",
        "source": "launcher_bootstrap",
        "source_type": "system",
        "text": (
            "Launcher bootstrap document. Add PDFs to app_data/incoming or links to "
            "app_data/incoming_links.txt to enrich the local knowledge base."
        ),
    }
    AUTO_JSONL_PATH.write_text(
        json.dumps(bootstrap_record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.warning("bootstrap: seed files not found, wrote minimal bootstrap JSONL")


def bootstrap_index_from_existing(logger: logging.Logger) -> bool:
    """Copy prebuilt index/chunks from shared paths when available."""
    target_index = INDEX_DIR / "faiss.index"
    target_chunks = INDEX_DIR / "chunks.jsonl"
    if target_index.exists() and target_chunks.exists():
        return True

    candidate_dirs = [
        executable_root() / "shared" / "data" / "data" / "indexes",
        executable_root().parent / "shared" / "data" / "data" / "indexes",
        REPO_ROOT / "shared" / "data" / "data" / "indexes",
        resource_path("shared", "data", "data", "indexes"),
    ]

    for src_dir in candidate_dirs:
        src_index = src_dir / "faiss.index"
        src_chunks = src_dir / "chunks.jsonl"
        if not (src_index.exists() and src_chunks.exists()):
            continue
        try:
            INDEX_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_index, target_index)
            shutil.copy2(src_chunks, target_chunks)
            logger.info("bootstrap: copied index from %s", src_dir)
            return True
        except Exception:
            logger.exception("bootstrap: failed copying index from %s", src_dir)

    return False


def invoke_ingest_via_import(
    jsonl_path: Path,
    index_dir: Path,
    embedding_model_path: Path,
    pending_sources: list[str],
) -> None:
    """Invoke ingest via module import call if possible."""
    module = importlib.import_module("src.backend.ingest")

    ingest_paths_fn = getattr(module, "ingest_paths", None)
    if callable(ingest_paths_fn) and pending_sources:
        ingest_paths_fn(pending_sources)
        return

    build_fn = getattr(module, "build", None)
    if callable(build_fn):
        try:
            build_fn(
                jsonl_path=str(jsonl_path),
                index_dir=str(index_dir),
                embedding_model_path=str(embedding_model_path),
            )
            return
        except TypeError:
            build_fn()
            return

    build_faiss_index_fn = getattr(module, "build_faiss_index", None)
    if callable(build_faiss_index_fn):
        build_faiss_index_fn(
            jsonl_path=str(jsonl_path),
            index_dir=str(index_dir),
            embedding_model_path=str(embedding_model_path),
        )
        return

    raise AttributeError("No supported ingest callable found")


def invoke_ingest_via_subprocess(
    jsonl_path: Path,
    index_dir: Path,
    embedding_model_path: Path,
    logger: logging.Logger,
) -> None:
    """Invoke ingest via hidden subprocess fallback."""
    stdout_handle = (LOG_DIR / "ingest_stdout.log").open("a", encoding="utf-8")
    stderr_handle = (LOG_DIR / "ingest_stderr.log").open("a", encoding="utf-8")
    env = os.environ.copy()
    env["RAG_DATA_JSONL"] = str(jsonl_path)
    env["RAG_INDEX_DIR"] = str(index_dir)
    env["RAG_EMBEDDING_MODEL_PATH"] = str(embedding_model_path)
    env["INCOMING_DIR"] = str(INCOMING_DIR)
    python_path_parts = [str(bundle_root()), str(executable_root())]
    if env.get("PYTHONPATH"):
        python_path_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

    if is_frozen():
                                                                              
        cmd = [
            sys.executable,
            CHILD_MODE_INGEST,
            "--jsonl",
            str(jsonl_path),
            "--index-dir",
            str(index_dir),
            "--embedding-model-path",
            str(embedding_model_path),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "src.backend.ingest",
            "build",
            "--jsonl",
            str(jsonl_path),
            "--index-dir",
            str(index_dir),
            "--embedding-model-path",
            str(embedding_model_path),
        ]

    logger.info("ingest: subprocess fallback started")
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            **hidden_popen_kwargs(stdout_handle, stderr_handle),
        )
        try:
            return_code = process.wait(timeout=INGEST_SUBPROCESS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            terminate_process_group(process, logger)
            raise TimeoutError(
                f"Ingest subprocess timed out after {INGEST_SUBPROCESS_TIMEOUT_SECONDS}s"
            )

        if return_code != 0:
            raise RuntimeError(f"Ingest subprocess failed with return code {return_code}")
    finally:
        stdout_handle.flush()
        stderr_handle.flush()
        stdout_handle.close()
        stderr_handle.close()


def invoke_ingest_build(
    jsonl_path: Path,
    index_dir: Path,
    embedding_model_path: Path,
    pending_sources: list[str],
    logger: logging.Logger,
) -> None:
    """Invoke ingest build import-first, then fallback subprocess."""
    try:
        invoke_ingest_via_import(
            jsonl_path=jsonl_path,
            index_dir=index_dir,
            embedding_model_path=embedding_model_path,
            pending_sources=pending_sources,
        )
        logger.info("ingest: import invocation succeeded")
        return
    except Exception:
        logger.exception("ingest: import invocation failed, falling back to subprocess")

    invoke_ingest_via_subprocess(
        jsonl_path=jsonl_path,
        index_dir=index_dir,
        embedding_model_path=embedding_model_path,
        logger=logger,
    )
    logger.info("ingest: subprocess fallback succeeded")


def ensure_initial_index(logger: logging.Logger) -> None:
    """Ensure index files exist for backend startup."""
    index_file = INDEX_DIR / "faiss.index"
    chunks_file = INDEX_DIR / "chunks.jsonl"
    if index_file.exists() and chunks_file.exists():
        return

    if bootstrap_index_from_existing(logger):
        return

    logger.warning("bootstrap: no prebuilt index found, building from managed JSONL")
    invoke_ingest_build(
        jsonl_path=AUTO_JSONL_PATH,
        index_dir=INDEX_DIR,
        embedding_model_path=EMBEDDING_MODEL_PATH,
        pending_sources=[],
        logger=logger,
    )


def prepare_child_runtime_paths() -> None:
    """Prepare sys.path and PYTHONPATH for child-role execution."""
    for candidate in (str(bundle_root()), str(executable_root())):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    python_path = os.environ.get("PYTHONPATH", "")
    parts = [str(bundle_root()), str(executable_root())]
    if python_path:
        parts.append(python_path)
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def call_ingest_build_callable(
    module: Any,
    jsonl_path: str,
    index_dir: str,
    embedding_model_path: str,
) -> None:
    """Call an ingest build function from module using common signatures."""
    build_fn = getattr(module, "build", None)
    if callable(build_fn):
        try:
            build_fn(
                jsonl_path=jsonl_path,
                index_dir=index_dir,
                embedding_model_path=embedding_model_path,
            )
            return
        except TypeError:
            build_fn()
            return

    build_faiss_index_fn = getattr(module, "build_faiss_index", None)
    if callable(build_faiss_index_fn):
        build_faiss_index_fn(
            jsonl_path=jsonl_path,
            index_dir=index_dir,
            embedding_model_path=embedding_model_path,
        )
        return

    main_fn = getattr(module, "main", None)
    if callable(main_fn):
        main_fn(
            [
                "build",
                "--jsonl",
                jsonl_path,
                "--index-dir",
                index_dir,
                "--embedding-model-path",
                embedding_model_path,
            ]
        )
        return

    raise AttributeError("No supported build function found in src.backend.ingest")


def run_streamlit_child(args: list[str]) -> int:
    """Run streamlit CLI in child mode and return process exit code."""
    if not args:
        raise ValueError("Missing streamlit script path for child mode.")

    streamlit_script = args[0]
    extra_args = args[1:]
    previous_argv = list(sys.argv)
    sys.argv = ["streamlit", "run", streamlit_script, *extra_args]
    try:
                                                                                                  
        import importlib.metadata as importlib_metadata

        original_version = importlib_metadata.version

        def safe_version(name: str) -> str:
            if name == "streamlit":
                try:
                    return original_version(name)
                except importlib_metadata.PackageNotFoundError:
                    return "0.0.0"
            return original_version(name)

        importlib_metadata.version = safe_version                            
        from streamlit.web import cli as stcli

        result = stcli.main()
        return int(result or 0)
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        return 1 if exc.code else 0
    finally:
        sys.argv = previous_argv


def run_ingest_child(args: list[str]) -> int:
    """Run ingest build in child mode and return process exit code."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--embedding-model-path", required=True)
    parsed = parser.parse_args(args)

    try:
        module = importlib.import_module("src.backend.ingest")
        call_ingest_build_callable(
            module=module,
            jsonl_path=parsed.jsonl,
            index_dir=parsed.index_dir,
            embedding_model_path=parsed.embedding_model_path,
        )
    except Exception:
        traceback.print_exc()
        return 1

    return 0


def maybe_run_child_mode() -> Optional[int]:
    """Run child role mode when requested; otherwise return None."""
    if len(sys.argv) <= 1:
        return None

    mode = sys.argv[1]
    if mode not in {CHILD_MODE_STREAMLIT, CHILD_MODE_INGEST}:
        return None

    prepare_child_runtime_paths()
    args = sys.argv[2:]
    try:
        if mode == CHILD_MODE_STREAMLIT:
            return run_streamlit_child(args)
        return run_ingest_child(args)
    except Exception:
        traceback.print_exc()
        return 1


def write_failed_link_record(link: str, error_message: str, logger: logging.Logger) -> None:
    """Write failed URL record under failed directory."""
    try:
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(ch if ch.isalnum() else "_" for ch in link)[:80]
        record = FAILED_DIR / f"failed_link_{stamp}_{safe}.txt"
        record.write_text(
            f"timestamp: {now_utc()}\nlink: {link}\nerror: {error_message}\n",
            encoding="utf-8",
        )
    except Exception:
        logger.exception("watcher: failed writing failed-link record for %s", link)

class DocumentWatcher(threading.Thread):
    """Watch incoming PDF files and links, then rebuild index in batches."""

    def __init__(
        self,
        stop_event: threading.Event,
        session: SessionRecorder,
        logger: logging.Logger,
    ) -> None:
        super().__init__(name="document-watcher", daemon=True)
        self.stop_event = stop_event
        self.session = session
        self.logger = logger
        self.processed_links = load_json_set(PROCESSED_LINKS_FILE, logger)
        self.failed_links = load_json_set(FAILED_LINKS_FILE, logger)

    def run(self) -> None:
        """Run periodic scan loop until stopped."""
        while not self.stop_event.is_set():
            try:
                self.scan_once()
            except Exception:
                self.logger.exception("watcher: unhandled error in scan cycle")
            self.stop_event.wait(CHECK_INTERVAL_SECONDS)

    def scan_once(self) -> None:
        """Run one watcher cycle: extract, batch rebuild, and finalize state."""
        pending_files = sorted(INCOMING_DIR.glob("*.pdf"))
        pending_links = self._new_links()
        if not pending_files and not pending_links:
            return

        self.logger.info(
            "watcher: found %s new files and %s new links",
            len(pending_files),
            len(pending_links),
        )

        batch = PendingBatch()

        for pdf_path in pending_files:
            if self.stop_event.is_set():
                break
            try:
                batch.records.extend(self._extract_pdf_records(pdf_path))
                batch.files.append(pdf_path)
                self.logger.info("watcher: extracted text from %s", pdf_path.name)
            except Exception as exc:
                self._mark_file_failed(pdf_path, exc, traceback.format_exc())

        for link in pending_links:
            if self.stop_event.is_set():
                break
            try:
                batch.records.append(self._extract_link_record(link))
                batch.links.append(link)
                self.logger.info("watcher: extracted text from link %s", link)
            except Exception as exc:
                self._mark_link_failed(link, exc, traceback.format_exc())

        if not batch.records:
            return

        temp_jsonl, new_count = self._build_merged_jsonl(batch.records)
        if new_count == 0:
            self.logger.info("watcher: no new unique records after dedupe")
            self._mark_batch_success_without_rebuild(batch)
            return

        self.logger.info("ingest: batch rebuild started with %s new records", new_count)
        try:
            sources = [str(path) for path in batch.files] + list(batch.links)
            invoke_ingest_build(
                jsonl_path=temp_jsonl,
                index_dir=INDEX_DIR,
                embedding_model_path=EMBEDDING_MODEL_PATH,
                pending_sources=sources,
                logger=self.logger,
            )
            os.replace(str(temp_jsonl), str(AUTO_JSONL_PATH))
            self.logger.info("ingest: batch rebuild succeeded")
            self._mark_batch_success_without_rebuild(batch)
        except Exception as exc:
            stack_trace = traceback.format_exc()
            self.logger.error("ingest: batch rebuild failed with error %s", exc)
            self.logger.error("ingest: batch stacktrace\n%s", stack_trace)
            self._mark_batch_failed(batch, exc, stack_trace)
            if temp_jsonl.exists():
                try:
                    temp_jsonl.unlink()
                except Exception:
                    self.logger.exception("watcher: failed removing temporary jsonl %s", temp_jsonl)

    def _new_links(self) -> list[str]:
        links = read_links(self.logger)
        return [
            link
            for link in links
            if link not in self.processed_links and link not in self.failed_links
        ]

    def _extract_pdf_records(self, pdf_path: Path) -> list[dict[str, Any]]:
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            records: list[dict[str, Any]] = []
            stat = pdf_path.stat()
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = (page.get_text("text") or "").strip()
                if not text:
                    continue
                fingerprint = sha1_text(
                    f"{pdf_path.name}|{stat.st_size}|{stat.st_mtime_ns}|{page_idx + 1}"
                )
                records.append(
                    {
                        "id": f"pdf_{fingerprint[:16]}",
                        "source": f"{pdf_path.name}#page={page_idx + 1}",
                        "source_type": "pdf",
                        "text": text,
                    }
                )
            if not records:
                raise ValueError(f"No extractable text found in PDF: {pdf_path.name}")
            return records
        finally:
            doc.close()

    def _extract_link_record(self, link: str) -> dict[str, Any]:
        import trafilatura

        headers = {"User-Agent": "RAG-Launcher/1.0"}
        last_error: Optional[Exception] = None
        content = ""
        for _ in range(URL_FETCH_RETRIES):
            try:
                response = requests.get(
                    link,
                    headers=headers,
                    timeout=URL_FETCH_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                content = response.text
                break
            except Exception as exc:
                last_error = exc
                time.sleep(1)

        if not content:
            raise RuntimeError(f"Failed fetching URL after retries: {last_error}")

        extracted = trafilatura.extract(content, include_links=False, include_tables=False)
        text = (extracted or "").strip()
        if not text:
            raise ValueError(f"Unable to extract readable text from URL: {link}")

        fingerprint = sha1_text(link)
        return {
            "id": f"url_{fingerprint[:16]}",
            "source": link,
            "source_type": "url",
            "text": text,
        }

    def _build_merged_jsonl(self, records: list[dict[str, Any]]) -> tuple[Path, int]:
        temp_jsonl = AUTO_JSONL_PATH.parent / f".auto_ingest_{int(time.time() * 1000)}.tmp.jsonl"
        AUTO_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

        seen_ids: set[str] = set()
        new_count = 0

        with temp_jsonl.open("w", encoding="utf-8") as out:
            if AUTO_JSONL_PATH.exists():
                with AUTO_JSONL_PATH.open("r", encoding="utf-8") as existing:
                    for raw in existing:
                        line = raw.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            self.logger.warning("watcher: skipped invalid JSONL line in %s", AUTO_JSONL_PATH)
                            continue
                        record_id = str(obj.get("id", "")).strip()
                        if not record_id or record_id in seen_ids:
                            continue
                        seen_ids.add(record_id)
                        out.write(json.dumps(obj, ensure_ascii=False) + "\n")

            for record in records:
                record_id = str(record.get("id", "")).strip()
                if not record_id or record_id in seen_ids:
                    continue
                seen_ids.add(record_id)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_count += 1

        return temp_jsonl, new_count

    def _mark_batch_success_without_rebuild(self, batch: PendingBatch) -> None:
        for pdf_path in batch.files:
            try:
                moved = move_file_atomic(pdf_path, PROCESSED_DIR)
                self.session.add_processed_file(moved)
                self.logger.info("watcher: moved processed file to %s", moved)
            except Exception as exc:
                self._mark_file_failed(pdf_path, exc, traceback.format_exc())

        if batch.links:
            self.processed_links.update(batch.links)
            save_json_set(PROCESSED_LINKS_FILE, self.processed_links, self.logger)
            for link in batch.links:
                self.session.add_processed_link(link)
                self.logger.info("watcher: marked processed link %s", link)

    def _mark_batch_failed(self, batch: PendingBatch, exc: Exception, stack_trace: str) -> None:
        for pdf_path in batch.files:
            self._mark_file_failed(pdf_path, exc, stack_trace)
        for link in batch.links:
            self._mark_link_failed(link, exc, stack_trace)

    def _mark_file_failed(self, pdf_path: Path, exc: Exception, stack_trace: str) -> None:
        self.session.add_failure(str(pdf_path), str(exc))
        append_manual_review(str(pdf_path), str(exc), stack_trace, self.logger)
        try:
            moved = move_file_atomic(pdf_path, FAILED_DIR)
            self.logger.info("watcher: moved failed file to %s", moved)
        except Exception:
            self.logger.exception("watcher: failed moving %s to failed directory", pdf_path)

    def _mark_link_failed(self, link: str, exc: Exception, stack_trace: str) -> None:
        self.session.add_failure(link, str(exc))
        append_manual_review(link, str(exc), stack_trace, self.logger)
        write_failed_link_record(link, str(exc), self.logger)
        self.failed_links.add(link)
        save_json_set(FAILED_LINKS_FILE, self.failed_links, self.logger)


def write_owner_notes(session: SessionRecorder, logger: logging.Logger) -> None:
    """Append session summary to EDIT_NOTES_FOR_OWNER.md."""
    summary = session.snapshot()
    lines = [
        f"## Launcher Session {now_utc()}",
        "",
        f"- Processed files: {len(summary.processed_files)}",
        f"- Processed links: {len(summary.processed_links)}",
        f"- Failures: {len(summary.failures)}",
        "",
    ]

    if summary.processed_files:
        lines.append("### Processed Files")
        lines.extend([f"- {item}" for item in summary.processed_files])
        lines.append("")

    if summary.processed_links:
        lines.append("### Processed Links")
        lines.extend([f"- {item}" for item in summary.processed_links])
        lines.append("")

    if summary.failures:
        lines.append("### Failures")
        lines.extend([f"- {item}" for item in summary.failures])
        lines.append("")

    try:
        with EDIT_NOTES_FILE.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    except Exception:
        logger.exception("notes: failed writing %s", EDIT_NOTES_FILE)


def run() -> None:
    """Start launcher lifecycle and cleanup on window close."""
    ensure_runtime_paths()
    logger = configure_logging()
    logger.info("launcher: session start %s", now_utc())

    configure_runtime_environment(logger)
    bootstrap_auto_jsonl(logger)

    try:
        ensure_initial_index(logger)
    except Exception:
        logger.exception("launcher: initial index preparation failed")

    session = SessionRecorder()
    stop_event = threading.Event()
    backend_thread = start_backend_thread(logger)
    streamlit_proc: Optional[ManagedSubprocess] = None
    watcher: Optional[DocumentWatcher] = None

    try:
        streamlit_proc = start_streamlit_process(logger)
        watcher = DocumentWatcher(stop_event=stop_event, session=session, logger=logger)

                                                                                    
        watcher.scan_once()
        watcher.start()

        streamlit_url = f"http://{STREAMLIT_HOST}:{STREAMLIT_PORT}"
        ready = wait_for_streamlit_ready(
            url=streamlit_url,
            timeout_seconds=STREAMLIT_STARTUP_TIMEOUT_SECONDS,
            logger=logger,
        )
        if not ready:
            logger.warning("launcher: opening fallback page because streamlit is not ready")

        open_native_window(streamlit_url=streamlit_url, is_ready=ready, logger=logger)
    except Exception:
        logger.exception("launcher: fatal error")
    finally:
        logger.info("launcher: shutdown initiated")
        stop_event.set()
        if watcher is not None:
            watcher.join(timeout=CHECK_INTERVAL_SECONDS + 5)

        close_managed_subprocess(streamlit_proc, "streamlit", logger)
        write_owner_notes(session, logger)

                                                                
        logger.info("backend: daemon thread alive=%s", backend_thread.is_alive())
        logger.info("launcher: session end %s", now_utc())
        logging.shutdown()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    child_exit_code = maybe_run_child_mode()
    if child_exit_code is not None:
        sys.exit(child_exit_code)

    lock_handle = acquire_single_instance_lock()
    if lock_handle is None:
        sys.exit(0)
    try:
        run()
    finally:
        release_single_instance_lock(lock_handle)
