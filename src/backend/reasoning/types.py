from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


@dataclass(frozen=True)
class Snippet:
    snippet_id: str
    doc_id: str
    chunk_id: str
    source: str
    text: str
    score: float

    char_start: int = 0
    char_end: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snippet_id": self.snippet_id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "text": self.text,
            "score": float(self.score),
            "char_start": int(self.char_start),
            "char_end": int(self.char_end),
        }


ClaimVerdict = Literal["Supported", "Not-Supported", "Inconclusive"]


@dataclass(frozen=True)
class VerifiedClaim:
    claim: str
    verdict: ClaimVerdict
    snippet_id: Optional[str]
    doc_id: Optional[str]
    chunk_id: Optional[str]
    quote: str
    char_start: int
    char_end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim": self.claim,
            "verdict": self.verdict,
            "snippet_id": self.snippet_id,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "quote": self.quote,
            "char_start": int(self.char_start),
            "char_end": int(self.char_end),
        }


@dataclass(frozen=True)
class ReasoningOutput:
    answer: str
    eli5: str
    technical: str
    evidence_chain: List[str]
    snippets: List[Snippet]
    claims: List[VerifiedClaim]
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "eli5": self.eli5,
            "technical": self.technical,
            "evidence_chain": self.evidence_chain,
            "snippets": [s.to_dict() for s in self.snippets],
            "claims": [c.to_dict() for c in self.claims],
            "confidence": float(self.confidence),
        }
