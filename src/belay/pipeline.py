"""The chat flow, from question to guarded answer.

    question -> input guard -> retrieve (public sections only) -> enough evidence?
             -> generator -> output guard (redaction, citation check) -> answer + trace

``build_pipeline`` is the factory: it picks the retriever mode, embedder and generator from the
settings, so the API, the tests and the evaluations all build the assistant the same way.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from belay.config import Settings
from belay.embeddings import Embedder, OllamaEmbedder, TfidfEmbedder
from belay.generators import Draft, ExtractiveGenerator, Generator, OllamaGenerator, Usage
from belay.guards import REFUSAL, InputGuard, OutputGuard
from belay.knowledge import KnowledgeBase, Section
from belay.retrieval import Hit, Retriever
from belay.tracing import Tracer

NOT_FOUND_ANSWER = "I don't have that information. Please ask the team at the front desk."


@dataclass(frozen=True)
class ChatResult:
    """Everything the API returns, plus the internals the trace keeps."""

    status: str  # answered | not_found | refused
    answer: str
    citations: list[Section]
    contexts: list[Section]  # the sections the generator was given
    trace_id: str
    latency_ms: float
    generator: str
    usage: Usage = field(default_factory=Usage)
    redactions: list[str] = field(default_factory=list)


class ChatPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        *,
        top_k: int = 3,
        min_terms: int = 2,
        min_similarity: float = 0.75,
        tracer: Tracer | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self._input_guard = InputGuard()
        self._output_guard = OutputGuard()
        self._top_k = top_k
        self._min_terms = min_terms
        self._min_similarity = min_similarity
        self._tracer = tracer or Tracer(None)

    def evidence(self, hits: list[Hit]) -> list[Hit]:
        """The hits good enough to answer from. If the best one isn't, the answer is "not found".

        Keyword overlap OR embedding similarity counts as evidence; requiring both would make a
        dense embedder useless for reworded questions, which is the reason to have one.
        """
        enough = [h for h in hits if h.matched_terms >= self._min_terms or h.similarity >= self._min_similarity]
        return enough if hits and enough and enough[0] is hits[0] else []

    def answer(self, question: str) -> ChatResult:
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        verdict = self._input_guard.check(question)
        hits: list[Hit] = []
        evidence: list[Hit] = []
        redactions: list[str] = []
        if verdict:
            status, text, citations, draft = "refused", REFUSAL, [], Draft(REFUSAL, [])
        else:
            hits = self.retriever.search(question, self._top_k)
            evidence = self.evidence(hits)
            draft = self.generator.generate(question, evidence)
            if draft.not_found:
                status, text, citations = "not_found", NOT_FOUND_ANSWER, []
            else:
                status = "answered"
                text, redactions = self._output_guard.redact(draft.text)
                citations = self._output_guard.valid_citations(draft.citations, [h.section.id for h in evidence])
                if not citations:
                    # No source it really saw: the answer can't be checked, so it isn't given
                    status, text = "not_found", NOT_FOUND_ANSWER
                    redactions.append("uncited_answer")

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        by_id = {h.section.id: h.section for h in evidence}
        result = ChatResult(
            status=status,
            answer=text,
            citations=[by_id[c] for c in citations],
            contexts=[h.section for h in evidence],
            trace_id=trace_id,
            latency_ms=latency_ms,
            generator=self.generator.name,
            usage=draft.usage,
            redactions=redactions,
        )
        self._tracer.record(
            trace_id=trace_id,
            question=question,
            status=status,
            guard_rule=verdict.rule if verdict else None,
            retrieved=[
                {
                    "id": h.section.id,
                    "score": round(h.score, 4),
                    "bm25": round(h.bm25, 3),
                    "similarity": round(h.similarity, 3),
                    "terms": h.matched_terms,
                }
                for h in hits
            ],
            evidence=[h.section.id for h in evidence],
            answer=text,
            citations=citations,
            redactions=redactions,
            generator=self.generator.name,
            latency_ms=latency_ms,
            usage={"prompt_tokens": draft.usage.prompt_tokens, "completion_tokens": draft.usage.completion_tokens},
        )
        return result


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "tfidf":
        return TfidfEmbedder()
    if settings.embedder == "ollama":
        return OllamaEmbedder(settings.ollama_url, settings.embed_model)
    raise ValueError(f"unknown embedder {settings.embedder!r}")


def build_generator(settings: Settings) -> Generator:
    if settings.generator == "extractive":
        return ExtractiveGenerator()
    if settings.generator == "ollama":
        return OllamaGenerator(settings.ollama_url, settings.model)
    raise ValueError(f"unknown generator {settings.generator!r}")


def build_pipeline(settings: Settings | None = None) -> ChatPipeline:
    """Factory: the one place that turns settings into a working assistant."""
    settings = settings or Settings.from_env()
    kb = KnowledgeBase.load(settings.knowledge_dir)
    # The chat is public: staff sections are not even indexed, so no question can retrieve them
    retriever = Retriever(kb.visible_to("public"), build_embedder(settings), settings.retriever)
    return ChatPipeline(
        retriever,
        build_generator(settings),
        top_k=settings.top_k,
        min_terms=settings.min_terms,
        min_similarity=settings.min_similarity,
        tracer=Tracer(settings.trace_path),
    )
