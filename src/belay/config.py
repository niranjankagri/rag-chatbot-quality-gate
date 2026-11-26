"""Settings, read from environment variables so CI and a local run differ only in the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    knowledge_dir: Path = ROOT / "knowledge"
    generator: str = "extractive"  # extractive | ollama
    retriever: str = "hybrid"  # bm25 | vector | hybrid
    embedder: str = "tfidf"  # tfidf | ollama
    ollama_url: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"
    top_k: int = 3
    # Evidence rule: the best section must share this many question words, or be this similar
    min_terms: int = 2
    min_similarity: float = 0.75
    rate_limit: int = 60  # requests per client per window; 0 switches the limit off
    rate_window_s: float = 60.0
    trace_path: Path | None = None

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ.get
        trace = env("BELAY_TRACE_PATH")
        return cls(
            knowledge_dir=Path(env("BELAY_KNOWLEDGE_DIR", str(cls.knowledge_dir))),
            generator=env("BELAY_GENERATOR", cls.generator),
            retriever=env("BELAY_RETRIEVER", cls.retriever),
            embedder=env("BELAY_EMBEDDER", cls.embedder),
            ollama_url=env("OLLAMA_URL", cls.ollama_url),
            model=env("BELAY_MODEL", cls.model),
            embed_model=env("BELAY_EMBED_MODEL", cls.embed_model),
            top_k=int(env("BELAY_TOP_K", cls.top_k)),
            min_terms=int(env("BELAY_MIN_TERMS", cls.min_terms)),
            min_similarity=float(env("BELAY_MIN_SIMILARITY", cls.min_similarity)),
            rate_limit=int(env("BELAY_RATE_LIMIT", cls.rate_limit)),
            rate_window_s=float(env("BELAY_RATE_WINDOW_S", cls.rate_window_s)),
            trace_path=Path(trace) if trace else None,
        )
