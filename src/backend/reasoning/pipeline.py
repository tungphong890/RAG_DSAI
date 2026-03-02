"""
Reasoning Pipeline

Advanced reasoning system for complex question answering.
Decomposes queries, verifies claims, and builds evidence chains
for enhanced accuracy in Data Science and AI topics.

Key Features:
- Query decomposition into subquestions
- Claim verification against sources
- Evidence chain construction
- Complexity assessment
- ELI5 explanations
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .decomposer import QueryDecomposer
from .types import ReasoningOutput, Snippet
from .verifier import ClaimVerifier


def _sentence_claims(text: str, max_claims: int = 6) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    claims: List[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if len(s.split()) < 8:
            continue
        claims.append(s)
        if len(claims) >= max_claims:
            break
    return claims


class ReasoningPipeline:
    def __init__(
        self,
        retriever: Any,
        generator: Any,
        *,
        decomposer: Optional[QueryDecomposer] = None,
        verifier: Optional[ClaimVerifier] = None,
        subq_top_n: int = 5,
        retrieval_k: int = 50,
        alpha: float = 0.6,
        use_reranker: bool = True,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.decomposer = decomposer or QueryDecomposer()
        self.verifier = verifier or ClaimVerifier()
        self.subq_top_n = subq_top_n
        self.retrieval_k = retrieval_k
        self.alpha = alpha
        self.use_reranker = use_reranker

    def _retrieve_for_subquestion(self, subq: str) -> List[Dict[str, Any]]:
        if hasattr(self.retriever, "search"):
            try:
                return self.retriever.search(
                    query=subq,
                    top_k=self.subq_top_n,
                    retrieval_k=self.retrieval_k,
                    alpha=self.alpha,
                    use_reranker=self.use_reranker,
                )
            except TypeError:
                return self.retriever.search(subq, top_k=self.subq_top_n)
        raise RuntimeError("Retriever missing search()")

    def run(self, question: str) -> Dict[str, Any]:
        dec = self.decomposer.decompose(question)

        per_subq: List[Dict[str, Any]] = []
        snippets: List[Snippet] = []
        ctx_for_llm: List[Dict[str, Any]] = []

        seen_chunk_ids: set[str] = set()
        snippet_counter = 0

        for i, subq in enumerate(dec.subquestions):
            results = self._retrieve_for_subquestion(subq)
            per_subq.append({"subquestion": subq, "results": results})

            for r in results:
                chunk_id = str(r.get("chunk_id", ""))
                if chunk_id and chunk_id in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk_id)

                snippet_id = f"S{snippet_counter}"
                snippet_counter += 1

                sn = Snippet(
                    snippet_id=snippet_id,
                    doc_id=str(r.get("doc_id", "")),
                    chunk_id=str(r.get("chunk_id", "")),
                    source=str(r.get("source", "")),
                    text=str(r.get("text", "")),
                    score=float(r.get("score", 0.0) or 0.0),
                    char_start=0,
                    char_end=min(len(str(r.get("text", ""))), 240),
                )
                snippets.append(sn)
                ctx_for_llm.append(r)

                                                        
        gen_out = self.generator.answer(question, ctx_for_llm)
        answer_text = str(gen_out.get("answer", ""))

                                    
        evidence_chain = []
        for sn in sorted(snippets, key=lambda x: x.score, reverse=True)[:6]:
            evidence_chain.append(
                f"{sn.snippet_id} supports key context from {sn.source} ({sn.chunk_id})."
            )

                                                          
        eli5 = answer_text
        if len(eli5) > 500:
            eli5 = eli5[:500].rstrip() + "..."
        technical = answer_text

        claims = _sentence_claims(answer_text)
        verified = [self.verifier.verify(c, snippets) for c in claims]

                                                                         
        gen_conf = float(gen_out.get("confidence", 0.0) or 0.0)
        supported = sum(1 for v in verified if v.verdict == "Supported")
        support_rate = supported / max(1, len(verified))
        confidence = 0.6 * gen_conf + 0.4 * support_rate
        confidence = max(0.0, min(1.0, confidence))

        out = ReasoningOutput(
            answer=answer_text,
            eli5=eli5,
            technical=technical,
            evidence_chain=evidence_chain,
            snippets=snippets,
            claims=verified,
            confidence=confidence,
        )

        return {
            "question": question,
            "complexity": dec.complexity,
            "subquestions": dec.subquestions,
            "per_subquestion_retrieval": per_subq,
            "result": out.to_dict(),
        }
