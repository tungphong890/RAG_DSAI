from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

Complexity = Literal["SIMPLE", "COMPLEX"]


@dataclass(frozen=True)
class Decomposition:
    complexity: Complexity
    subquestions: List[str]


class QueryDecomposer:
    def __init__(self, max_subquestions: int = 4) -> None:
        self.max_subquestions = max_subquestions

    def classify(self, question: str) -> Complexity:
        q = (question or "").strip().lower()
        if not q:
            return "SIMPLE"

        tokens = q.split()
        if len(tokens) >= 14:
            return "COMPLEX"

        triggers = [
            " and ",
            " then ",
            " vs ",
            " versus ",
            " compare ",
            " difference ",
            " differences ",
            " first ",
            " second ",
            " steps ",
            " multi-hop ",
        ]
        if any(t in q for t in triggers):
            return "COMPLEX"
        return "SIMPLE"

    def decompose(self, question: str) -> Decomposition:
        q = (question or "").strip()
        if not q:
            return Decomposition(complexity="SIMPLE", subquestions=[])

        complexity = self.classify(q)
        if complexity == "SIMPLE":
            return Decomposition(complexity=complexity, subquestions=[q])

        lowered = q.lower()
        seps = [" vs ", " versus ", " and ", ";", "."]
        parts: List[str] = [q]
        for sep in seps:
            if sep in lowered:
                if sep.strip() in {".", ";"}:
                    parts = [p.strip() for p in q.split(sep) if p.strip()]
                else:
                    parts = [p.strip() for p in q.split(sep) if p.strip()]
                break

        subqs: List[str] = []
        if len(parts) <= 1:
            subqs = [
                f"What are the key facts needed to answer: {q}",
                f"What are the main steps or components involved in: {q}",
            ]
        else:
            if " vs " in lowered or " versus " in lowered:
                if len(parts) >= 2:
                    subqs = [
                        f"Explain {parts[0]}.",
                        f"Explain {parts[1]}.",
                        f"Compare {parts[0]} and {parts[1]}.",
                    ]
                else:
                    subqs = [q]
            else:
                subqs = [p if p.endswith("?") else p + "?" for p in parts]

        subqs = [s.strip() for s in subqs if s.strip()]
        subqs = subqs[: self.max_subquestions]
        if not subqs:
            subqs = [q]

        return Decomposition(complexity=complexity, subquestions=subqs)
