from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from .types import Snippet, VerifiedClaim


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def _overlap_score(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta))


class ClaimVerifier:
    def __init__(
        self, supported_threshold: float = 0.35, not_supported_threshold: float = 0.15
    ) -> None:
        self.supported_threshold = supported_threshold
        self.not_supported_threshold = not_supported_threshold

    def verify(self, claim: str, snippets: Iterable[Snippet]) -> VerifiedClaim:
        best: Optional[Tuple[Snippet, float]] = None
        for s in snippets:
            score = _overlap_score(claim, s.text)
            if best is None or score > best[1]:
                best = (s, score)

        if best is None:
            return VerifiedClaim(
                claim=claim,
                verdict="Inconclusive",
                snippet_id=None,
                doc_id=None,
                chunk_id=None,
                quote="",
                char_start=0,
                char_end=0,
            )

        snippet, score = best
        if score >= self.supported_threshold:
            verdict = "Supported"
        elif score <= self.not_supported_threshold:
            verdict = "Not-Supported"
        else:
            verdict = "Inconclusive"

        quote = snippet.text
        char_start = 0
        char_end = min(len(quote), 240)
        return VerifiedClaim(
            claim=claim,
            verdict=verdict,
            snippet_id=snippet.snippet_id,
            doc_id=snippet.doc_id,
            chunk_id=snippet.chunk_id,
            quote=quote[:240],
            char_start=char_start,
            char_end=char_end,
        )
