"""BM25, TF-IDF vectors, cosine, rank fusion and the evidence rule."""

import math

import pytest

from belay.embeddings import TfidfEmbedder, cosine
from belay.knowledge import Section
from belay.retrieval import BM25, Retriever


def section(sid: str, text: str, title: str = "t") -> Section:
    return Section(sid, title, text, "public", "test.md")


SECTIONS = [
    section("A1", "Shoe rental costs 4 euros. Harness rental costs 3 euros.", "Rentals"),
    section("A2", "The gym opens at 07:00 on weekdays.", "Opening hours"),
    section("A3", "A freeze costs 5 euros per month.", "Freezing"),
]


def test_cosine_basics():
    assert cosine({0: 1.0}, {0: 2.0}) == pytest.approx(1.0)
    assert cosine({0: 1.0}, {1: 1.0}) == 0.0
    assert cosine({}, {0: 1.0}) == 0.0  # an empty vector is not an error
    assert cosine({0: 1.0, 1: 1.0}, {0: 1.0}) == pytest.approx(1 / math.sqrt(2))


def test_tfidf_gives_rare_words_more_weight():
    embedder = TfidfEmbedder()
    embedder.fit(["euros shoe", "euros freeze", "euros gym"])
    vector = embedder.embed(["euros shoe"])[0]
    weights = sorted(vector.values())
    assert weights[0] < weights[1]  # "euros" (everywhere) < "shoe" (once)


def test_tfidf_must_be_fitted_first():
    with pytest.raises(RuntimeError):
        TfidfEmbedder().embed(["x"])


def test_bm25_ranks_the_matching_section_first_and_ignores_unknown_words():
    scores = BM25(SECTIONS).scores(["shoe", "rental", "unknownword"])
    assert scores.index(max(scores)) == 0
    assert BM25(SECTIONS).scores(["unknownword"]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_every_mode_finds_the_obvious_section(mode):
    hits = Retriever(SECTIONS, TfidfEmbedder(), mode).search("How much is shoe rental?", top_k=2)
    assert hits[0].section.id == "A1"
    assert hits[0].matched_terms == 2  # "shoe", "rental"


def test_hybrid_fusion_uses_ranks_not_raw_scores():
    retriever = Retriever(SECTIONS, TfidfEmbedder(), "hybrid", rrf_k=60)
    top = retriever.search("freeze", top_k=1)[0]
    # First in both rankings: 1/61 + 1/61
    assert top.section.id == "A3" and top.score == pytest.approx(2 / 61)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        Retriever(SECTIONS, TfidfEmbedder(), "magic")
