"""Turning text into vectors, and comparing vectors with cosine similarity.

Two interchangeable embedders (Strategy pattern):

- ``TfidfEmbedder``: sparse TF-IDF vectors built from the knowledge base itself. No model, no
  download, same numbers on every run, so CI can gate on them.
- ``OllamaEmbedder``: dense vectors from a local embedding model (``nomic-embed-text``). It can
  match words that never appear in the article ("fast option" vs "express"), which TF-IDF can't.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

import httpx

from belay.text import tokens

Vector = dict[int, float]  # sparse: dimension -> value; a dense vector uses every index


class Embedder(Protocol):
    """Anything that can turn a batch of texts into vectors of the same space."""

    name: str

    def fit(self, corpus: list[str]) -> None:
        """Learns from the documents (vocabulary, weights). A no-op for pre-trained models."""

    def embed(self, texts: list[str]) -> list[Vector]: ...


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity: 1.0 = same direction, 0.0 = nothing in common.

    Works on sparse vectors: only the dimensions both vectors use contribute to the dot product.
    """
    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(dim, 0.0) for dim, value in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


class TfidfEmbedder:
    """Term frequency x inverse document frequency, over the knowledge base's own words.

    A word that appears in every section ("euros") gets a low weight; a rare one ("freeze") a high
    one. Words the knowledge base never uses are unknown and simply ignored.
    """

    name = "tfidf"

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []

    def fit(self, corpus: list[str]) -> None:
        documents = [set(tokens(text)) for text in corpus]
        vocab = sorted(set().union(*documents))
        self._vocab = {word: i for i, word in enumerate(vocab)}
        n = len(documents)
        # Smoothed IDF (as in scikit-learn): never zero, never divides by zero
        self._idf = [math.log((1 + n) / (1 + sum(word in d for d in documents))) + 1 for word in vocab]

    def embed(self, texts: list[str]) -> list[Vector]:
        if not self._vocab:
            raise RuntimeError("TfidfEmbedder.fit() must run before embed()")
        vectors = []
        for text in texts:
            counts = Counter(w for w in tokens(text) if w in self._vocab)
            vectors.append({self._vocab[w]: c * self._idf[self._vocab[w]] for w, c in counts.items()})
        return vectors


class OllamaEmbedder:  # pragma: needs ollama
    """Dense vectors from a local Ollama embedding model, through its ``/api/embed`` endpoint."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.name = f"ollama:{model}"
        self._url = base_url.rstrip("/") + "/api/embed"
        self._model = model
        self._timeout = timeout

    def fit(self, corpus: list[str]) -> None:
        """Nothing to learn: the model is pre-trained."""

    def embed(self, texts: list[str]) -> list[Vector]:
        response = httpx.post(self._url, json={"model": self._model, "input": texts}, timeout=self._timeout)
        response.raise_for_status()
        return [dict(enumerate(vector)) for vector in response.json()["embeddings"]]
