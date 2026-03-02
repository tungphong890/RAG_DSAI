"""
Document Ingestion Module

Handles PDF document processing, text chunking, and embedding generation.
Creates FAISS vector indexes for efficient semantic search of Data Science content.

Key Features:
- PDF text extraction and filtering
- Recursive text chunking with overlap
- BGE embedding generation
- FAISS index creation and management
- Batch processing for large document sets
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModel, AutoTokenizer

try:
    from .device_manager import get_device_config
    from .settings import (
        DEFAULT_DATA_JSONL as SETTINGS_DEFAULT_DATA_JSONL,
        DEFAULT_EMBEDDING_MODEL_PATH as SETTINGS_DEFAULT_EMBEDDING_MODEL_PATH,
        DEFAULT_INDEX_DIR as SETTINGS_DEFAULT_INDEX_DIR,
    )
except ImportError:
    from device_manager import get_device_config
    from settings import (
        DEFAULT_DATA_JSONL as SETTINGS_DEFAULT_DATA_JSONL,
        DEFAULT_EMBEDDING_MODEL_PATH as SETTINGS_DEFAULT_EMBEDDING_MODEL_PATH,
        DEFAULT_INDEX_DIR as SETTINGS_DEFAULT_INDEX_DIR,
    )

DEFAULT_EMBEDDING_MODEL_PATH = str(SETTINGS_DEFAULT_EMBEDDING_MODEL_PATH)
DEFAULT_DATA_JSONL = str(SETTINGS_DEFAULT_DATA_JSONL)
DEFAULT_INDEX_DIR = str(SETTINGS_DEFAULT_INDEX_DIR)


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    source: str
    text: str


class LocalBGEEmbedder:
    def __init__(
        self,
        model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.device = get_device_config(device).device

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        all_embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            token_embs = out.last_hidden_state
            attn = enc["attention_mask"].unsqueeze(-1).type_as(token_embs)
            summed = (token_embs * attn).sum(dim=1)
            counts = attn.sum(dim=1).clamp(min=1)
            embs = summed / counts
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.detach().cpu().to(torch.float32).numpy())

        return np.concatenate(all_embs, axis=0)


def load_jsonl_documents(jsonl_path: str) -> List[Dict[str, Any]]:
    path = Path(jsonl_path)
    docs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {jsonl_path}: {e}"
                ) from e

            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Missing/empty 'text' on line {line_no} in {jsonl_path}"
                )

            doc_id = str(obj.get("id") or f"doc_{len(docs)}")
            source = str(obj.get("source") or doc_id)
            docs.append({"id": doc_id, "source": source, "text": text})

    return docs


def chunk_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[ChunkRecord]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: List[ChunkRecord] = []
    for doc in documents:
        doc_id = str(doc["id"])
        source = str(doc["source"])
        texts = splitter.split_text(str(doc["text"]))
        for j, t in enumerate(texts):
            chunk_id = f"{doc_id}_chunk_{j}"
            chunks.append(
                ChunkRecord(chunk_id=chunk_id, doc_id=doc_id, source=source, text=t)
            )

    return chunks


def build_faiss_index(
    jsonl_path: str = DEFAULT_DATA_JSONL,
    index_dir: str = DEFAULT_INDEX_DIR,
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
    batch_size: int = 8,
) -> Tuple[str, str]:
    index_dir_path = Path(index_dir)
    index_dir_path.mkdir(parents=True, exist_ok=True)

    documents = load_jsonl_documents(jsonl_path)
    chunks = chunk_documents(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    embedder = LocalBGEEmbedder(model_path=embedding_model_path)
    embeddings = embedder.encode([c.text for c in chunks], batch_size=batch_size)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise RuntimeError("Embedding shape mismatch")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    index_path = str(index_dir_path / "faiss.index")
    meta_path = str(index_dir_path / "chunks.jsonl")

    faiss.write_index(index, index_path)

    with Path(meta_path).open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "source": c.source,
                        "text": c.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return index_path, meta_path


def load_index(
    index_dir: str = DEFAULT_INDEX_DIR,
) -> Tuple[faiss.Index, List[ChunkRecord]]:
    index_dir_path = Path(index_dir)
    index_path = index_dir_path / "faiss.index"
    meta_path = index_dir_path / "chunks.jsonl"

    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index at: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing chunk metadata at: {meta_path}")

    index = faiss.read_index(str(index_path))

    chunks: List[ChunkRecord] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chunks.append(
                ChunkRecord(
                    chunk_id=str(obj["chunk_id"]),
                    doc_id=str(obj["doc_id"]),
                    source=str(obj["source"]),
                    text=str(obj["text"]),
                )
            )

    return index, chunks


def search(
    query: str,
    top_k: int = 5,
    index_dir: str = DEFAULT_INDEX_DIR,
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
) -> List[Dict[str, Any]]:
    index, chunks = load_index(index_dir=index_dir)
    embedder = LocalBGEEmbedder(model_path=embedding_model_path)
    q = embedder.encode([query], batch_size=1)

    scores, idxs = index.search(q.astype(np.float32), top_k)
    out: List[Dict[str, Any]] = []
    for rank, (score, idx) in enumerate(zip(scores[0].tolist(), idxs[0].tolist())):
        if idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx]
        out.append(
            {
                "rank": rank,
                "score": float(score),
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source": c.source,
                "text": c.text,
            }
        )

    return out


def _cmd_build(args: argparse.Namespace) -> None:
    index_path, meta_path = build_faiss_index(
        jsonl_path=args.jsonl,
        index_dir=args.index_dir,
        embedding_model_path=args.embedding_model_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    print(f"Wrote FAISS index: {index_path}")
    print(f"Wrote chunk metadata: {meta_path}")


def _cmd_search(args: argparse.Namespace) -> None:
    results = search(
        query=args.query,
        top_k=args.top_k,
        index_dir=args.index_dir,
        embedding_model_path=args.embedding_model_path,
    )
    for r in results:
        preview = r["text"].replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:180] + "..."
        print(
            f"[{r['rank']}] score={r['score']:.4f} {r['source']} {r['chunk_id']} :: {preview}"
        )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Local JSONL -> BGE-M3 -> FAISS ingest + search"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build")
    p_build.add_argument("--jsonl", "--jsonl_path", default=DEFAULT_DATA_JSONL)
    p_build.add_argument("--index-dir", "--index_dir", default=DEFAULT_INDEX_DIR)
    p_build.add_argument(
        "--embedding-model-path",
        "--embedding_model_path",
        default=DEFAULT_EMBEDDING_MODEL_PATH,
    )
    p_build.add_argument("--chunk-size", "--chunk_size", type=int, default=1000)
    p_build.add_argument("--chunk-overlap", "--chunk_overlap", type=int, default=150)
    p_build.add_argument("--batch-size", "--batch_size", type=int, default=8)
    p_build.set_defaults(func=_cmd_build)

    p_search = sub.add_parser("search")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", "--top_k", type=int, default=5)
    p_search.add_argument("--index-dir", "--index_dir", default=DEFAULT_INDEX_DIR)
    p_search.add_argument(
        "--embedding-model-path",
        "--embedding_model_path",
        default=DEFAULT_EMBEDDING_MODEL_PATH,
    )
    p_search.set_defaults(func=_cmd_search)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
