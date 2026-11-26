"""Finding the sections that can answer a question.

Two rankings, combined:

- **BM25**, the classic keyword score: rewards rare question words, and does not let a long
  section win just by being long.
- **Vector similarity**: cosine between the question's embedding and each section's embedding.

``mode`` picks one of them or both. ``hybrid`` merges the two rankings with Reciprocal Rank
Fusion (RRF), which only uses positions, so the two scores never need to be on the same scale.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from belay.embeddings import Embedder, cosine
from belay.knowledge import Section
from belay.text import tokens

MODES = ("bm25", "vector", "hybrid")


@dataclass(frozen=True)
class Hit:
    """A retrieved section with every score that led to its rank (kept for traces and evals)."""

    section: Section
    score: float  # the score used for ranking in the chosen mode
    bm25: float
    similarity: float
    matched_terms: int  # different question words found in the section


class BM25:
    """Okapi BM25 over the given sections. k1 and b are the usual textbook defaults."""

    def __init__(self, sections: list[Section], k1: float = 1.5, b: float = 0.75):
        self._docs = [Counter(tokens(f"{s.title} {s.text}")) for s in sections]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lengths) / len(self._lengths)
        self._k1, self._b = k1, b
        n = len(self._docs)
        df = Counter(word for doc in self._docs for word in doc)
        # The "+1" inside the log keeps the IDF positive even for words in most sections
        self._idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def scores(self, query_terms: list[str]) -> list[float]:
        result = []
        for doc, length in zip(self._docs, self._lengths, strict=True):
            score = 0.0
            for term in set(query_terms):
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self._k1 * (1 - self._b + self._b * length / self._avg)
                    score += self._idf[term] * tf * (self._k1 + 1) / norm
            result.append(score)
        return result


class Retriever:
    """Ranks the sections one caller may see. Built once per access level, reused per question."""

    def __init__(self, sections: list[Section], embedder: Embedder, mode: str = "hybrid", rrf_k: int = 60):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.sections = sections
        self.mode = mode
        self._rrf_k = rrf_k
        self._bm25 = BM25(sections)
        self._embedder = embedder
        texts = [f"{s.title}. {s.text}" for s in sections]
        embedder.fit(texts)
        self._vectors = embedder.embed(texts)
        self._terms = [set(tokens(t)) for t in texts]

    def search(self, question: str, top_k: int = 3) -> list[Hit]:
        """The ``top_k`` best sections, best first. Ties keep knowledge base order (stable)."""
        query_terms = tokens(question)
        bm25 = self._bm25.scores(query_terms)
        query_vector = self._embedder.embed([question])[0]
        similarity = [cosine(query_vector, v) for v in self._vectors]

        if self.mode == "bm25":
            ranking = bm25
        elif self.mode == "vector":
            ranking = similarity
        else:
            ranking = self._fuse(bm25, similarity)

        order = sorted(range(len(self.sections)), key=lambda i: -ranking[i])
        unique_terms = set(query_terms)
        return [
            Hit(self.sections[i], ranking[i], bm25[i], similarity[i], len(unique_terms & self._terms[i]))
            for i in order[:top_k]
        ]

    def _fuse(self, *rankings: list[float]) -> list[float]:
        """Reciprocal Rank Fusion: each ranking adds 1 / (k + position). Zero scores add nothing."""
        fused = [0.0] * len(self.sections)
        for scores in rankings:
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            for position, i in enumerate(order, start=1):
                if scores[i] > 0:
                    fused[i] += 1 / (self._rrf_k + position)
        return fused
