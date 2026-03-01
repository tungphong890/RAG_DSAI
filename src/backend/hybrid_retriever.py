"""
Hybrid Retrieval System

Combines BM25 (sparse) and vector (dense) retrieval with cross-encoder reranking.
Optimized for production-grade RAG with sub-10s latency targets.

Key Features:
- BM25 keyword search for exact matches
- Vector semantic search for conceptual matches
- Cross-encoder reranking for precision
- Hybrid scoring with configurable weights
- GPU acceleration for embeddings
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer

try:
    from .device_manager import get_device_config
    from .settings import (
        DEFAULT_EMBEDDING_MODEL_PATH as SETTINGS_DEFAULT_EMBEDDING_MODEL_PATH,
        DEFAULT_INDEX_DIR as SETTINGS_DEFAULT_INDEX_DIR,
    )
except ImportError:
    from device_manager import get_device_config
    from settings import (
        DEFAULT_EMBEDDING_MODEL_PATH as SETTINGS_DEFAULT_EMBEDDING_MODEL_PATH,
        DEFAULT_INDEX_DIR as SETTINGS_DEFAULT_INDEX_DIR,
    )

DEFAULT_EMBEDDING_MODEL_PATH = str(SETTINGS_DEFAULT_EMBEDDING_MODEL_PATH)
DEFAULT_INDEX_DIR = str(SETTINGS_DEFAULT_INDEX_DIR)
DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Fast, lightweight reranker
)


class HybridRetriever:
    """
    Combines BM25 (keyword) + Vector (semantic) search with optional reranking
    """

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
        use_reranker: bool = True,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        device: Optional[str] = None,
    ):
        self.index_dir = Path(index_dir)
        self.device = get_device_config(device).device
        self.use_reranker = use_reranker

        # Load FAISS index and chunks
        print("[HybridRetriever] Loading FAISS index...")
        self.faiss_index = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.chunks = self._load_chunks()

        # Load embedding model
        print(f"[HybridRetriever] Loading embeddings from {embedding_model_path}...")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        self.tokenizer = AutoTokenizer.from_pretrained(
            embedding_model_path,
            local_files_only=True,
            use_fast=True,
        )
        self.embedding_model = AutoModel.from_pretrained(
            embedding_model_path,
            local_files_only=True,
        ).to(self.device)
        self.embedding_model.eval()

        # Build BM25 index
        print("[HybridRetriever] Building BM25 index...")
        self.bm25 = self._build_bm25_index()

        # Load reranker (optional)
        self.reranker = None
        env_use_reranker = os.environ.get("RAG_USE_RERANKER")
        if env_use_reranker is not None:
            use_reranker = env_use_reranker.strip() in {"1", "true", "True"}

        if use_reranker:
            print(f"[HybridRetriever] Loading reranker: {reranker_model}...")
            try:
                self.reranker = CrossEncoder(
                    reranker_model, max_length=512, device=self.device
                )
                print("[HybridRetriever] Reranker loaded successfully")
            except Exception as e:
                print(f"[HybridRetriever] Warning: Could not load reranker: {e}")
                print("[HybridRetriever] Continuing without reranking...")
                self.use_reranker = False

        print(f"[HybridRetriever] Ready! {len(self.chunks)} chunks indexed")

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunk metadata from JSONL"""
        chunks = []
        meta_path = self.index_dir / "chunks.jsonl"
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
        return chunks

    def _build_bm25_index(self) -> BM25Okapi:
        """Build BM25 index from chunk texts"""
        # Tokenize all chunks (simple whitespace + lowercase)
        tokenized_corpus = [chunk["text"].lower().split() for chunk in self.chunks]
        return BM25Okapi(tokenized_corpus)

    @torch.no_grad()
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query using BGE-M3"""
        enc = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.embedding_model(**enc)

        # Mean pooling
        token_embs = out.last_hidden_state
        attn = enc["attention_mask"].unsqueeze(-1).type_as(token_embs)
        summed = (token_embs * attn).sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1)
        embs = summed / counts
        embs = torch.nn.functional.normalize(embs, p=2, dim=1)

        return embs.detach().cpu().to(torch.float32).numpy()

    def _vector_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Dense vector search using FAISS"""
        query_emb = self._encode_query(query)
        scores, indices = self.faiss_index.search(query_emb.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((int(idx), float(score)))
        return results

    def _bm25_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Sparse BM25 search"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices]
        return results

    def _hybrid_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        alpha: float = 0.5,
    ) -> List[Tuple[int, float]]:
        """
        Combine vector and BM25 scores using weighted fusion
        alpha: weight for vector search (1-alpha for BM25)
        """

        # Normalize scores to [0, 1]
        def normalize_scores(results: List[Tuple[int, float]]) -> Dict[int, float]:
            if not results:
                return {}
            scores = [s for _, s in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {idx: 1.0 for idx, _ in results}
            return {idx: (s - min_s) / (max_s - min_s) for idx, s in results}

        vec_scores = normalize_scores(vector_results)
        bm25_scores = normalize_scores(bm25_results)

        # Combine scores
        all_indices = set(vec_scores.keys()) | set(bm25_scores.keys())
        combined = []
        for idx in all_indices:
            vec_s = vec_scores.get(idx, 0.0)
            bm25_s = bm25_scores.get(idx, 0.0)
            combined_score = alpha * vec_s + (1 - alpha) * bm25_s
            combined.append((idx, combined_score))

        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined

    def _rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder"""
        if not self.reranker or not candidates:
            return candidates[:top_k]

        # Prepare query-document pairs
        pairs = [[query, c["text"]] for c in candidates]

        # Get reranker scores
        scores = self.reranker.predict(pairs)

        # Sort by reranker score
        for i, score in enumerate(scores):
            candidates[i]["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieval_k: int = 50,
        alpha: float = 0.5,
        use_reranker: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with optional reranking

        Args:
            query: Search query
            top_k: Final number of results to return
            retrieval_k: Number of candidates to retrieve before reranking
            alpha: Weight for vector search (0=BM25 only, 1=vector only, 0.5=balanced)
            use_reranker: Override instance reranker setting

        Returns:
            List of ranked chunks with scores
        """
        # Step 1: Retrieve candidates using hybrid search
        vector_results = self._vector_search(query, top_k=retrieval_k)
        bm25_results = self._bm25_search(query, top_k=retrieval_k)

        # Step 2: Fuse scores
        fused_results = self._hybrid_fusion(vector_results, bm25_results, alpha=alpha)

        # Step 3: Prepare candidates
        candidates = []
        for rank, (idx, score) in enumerate(fused_results[:retrieval_k]):
            chunk = self.chunks[idx]
            candidates.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                }
            )

        # Step 4: Rerank (optional)
        use_rerank = use_reranker if use_reranker is not None else self.use_reranker
        if use_rerank and self.reranker:
            candidates = self._rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        # Update ranks after reranking
        for i, c in enumerate(candidates):
            c["rank"] = i

        return candidates


def search_hybrid(
    query: str,
    top_k: int = 5,
    index_dir: str = DEFAULT_INDEX_DIR,
    embedding_model_path: str = DEFAULT_EMBEDDING_MODEL_PATH,
    use_reranker: bool = True,
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Convenience function for one-off hybrid searches
    For production, create a HybridRetriever instance and reuse it
    """
    retriever = HybridRetriever(
        index_dir=index_dir,
        embedding_model_path=embedding_model_path,
        use_reranker=use_reranker,
    )
    return retriever.search(query, top_k=top_k, alpha=alpha)


if __name__ == "__main__":
    # Test the hybrid retriever
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hybrid_retriever.py 'your query here'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\nQuery: {query}\n")

    results = search_hybrid(query, top_k=5, use_reranker=True, alpha=0.5)

    for r in results:
        preview = r["text"].replace("\n", " ")
        if len(preview) > 150:
            preview = preview[:150] + "..."

        rerank_info = (
            f" | rerank={r.get('rerank_score', 0):.3f}" if "rerank_score" in r else ""
        )
        print(f"[{r['rank']}] score={r['score']:.4f}{rerank_info}")
        print(f"    {r['source']} :: {r['chunk_id']}")
        print(f"    {preview}\n")
