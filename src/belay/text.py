"""Small text helpers shared by retrieval, the generators and the guards.

Deliberately simple and dependency free: the same words go in at indexing time and at question
time, so the retriever, the extractive generator and the evals all agree on what a "word" is.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
# A sentence ends at . ! or ? followed by a space and a capital letter or digit; "e.g." stays whole
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Words that carry no meaning for matching a question to a help article
STOPWORDS = frozenset(
    """a an and are as at be been but by can could do does did for from has have how i if in is it
    its me my of on or our so than that the their them then there these they this to up was we were
    what when where which who why will with would you your yours am any all also about after before
    should may might must not no yes per one get got much many more most other some such into out
    over only own same too very just""".split()
)


def stem(word: str) -> str:
    """Strips a few English endings so "classes" matches "class" and "cancelled" matches "cancel".

    A light suffix stripper, not a real stemmer: it only touches words longer than 4 letters, so
    short words like "pass" or "kids" stay as they are. It does not link "rent" and "rental": that
    is the kind of gap the evaluation is meant to show, not one to hide with special cases.
    """
    if len(word) <= 4:
        return word
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return word[:-2]  # classes -> class, passes -> pass
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]  # shoes -> shoe, members -> member
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            # cancelling -> cancell -> cancel, stopped -> stopp -> stop
            if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
                word = word[:-1]
            return word
    return word


def tokens(text: str) -> list[str]:
    """Lower-cased, stemmed content words in their original order (stopwords removed)."""
    return [stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


def sentences(text: str) -> list[str]:
    """Splits a paragraph into sentences; keeps times like 07:00 and prices like 4.50 intact."""
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
