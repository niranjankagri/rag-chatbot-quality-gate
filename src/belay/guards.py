"""Guardrails around the assistant: one on the way in, one on the way out.

They are plain rules, not a model, so they behave the same on every run and can be tested with
planted inputs. They are a first line, not a proof: a word list can be talked around, which is
exactly what the red-team scan (Promptfoo) is there to find.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Text that tries to give the assistant new instructions. Used for user questions AND for
# documents, because an injection can arrive through either.
_INSTRUCTION = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules|prompts?|guidelines)\b",
    r"\bsystem\s+(note|prompt|message|override|instructions?)\b",
    r"\b(you are now|act as|pretend (to be|you are)|roleplay as|developer mode|jailbreak)\b",
    r"\b(reveal|show|print|repeat|output|tell me)\b.{0,30}\b(system prompt|hidden|initial|your)\s+"
    r"(instructions?|prompt|rules)\b",
]

# Secrets that are in the staff handbook and must never be handed out
_SECRET = r"\b(door|alarm|staff|entry|pin|access)\s+(code|pin|number)s?\b|\bpasswords?\b"

# Personal data of other people. "phone" alone is fine ("Can I pay with my phone?"), so a request
# is refused only when personal data words appear together with words pointing at other people.
_PERSONAL_DATA = r"\b(phone|mobile|e-?mails?|addresse?s?|contact details)\b"
# Lists of people are other people's data by definition, whoever asks
_PEOPLE_LISTS = r"\b(contact|member|members|customer|phone)\s+lists?\b|\bemergency contacts?\b"
_OTHER_PEOPLE = r"\b(members?|someone|somebody|people|customers?|climbers?|other|another|everyone|staff)\b"
_POSSESSIVE_NAME = r"\b(?!(?:gym|northpeak|your|my)'s)[a-z]+'s\b"

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")
_PHONE = re.compile(r"\+?\d[\d ()-]{7,}\d")
_CODE = re.compile(r"\b(code|pin)\b([^.\d]{0,25})\b\d{4,6}\b", re.IGNORECASE)

REFUSAL = (
    "I can't help with that. I can answer questions about memberships, opening hours, courses and visiting Northpeak."
)


def looks_like_instruction(text: str) -> bool:
    """True when the text tries to instruct the assistant (prompt injection)."""
    lowered = text.lower()
    return any(re.search(p, lowered) for p in _INSTRUCTION)


@dataclass(frozen=True)
class Verdict:
    """Why a question was refused; ``rule`` goes into the trace, never to the user."""

    rule: str


class InputGuard:
    """Refuses questions that try to inject instructions, or ask for secrets or other people's data."""

    def check(self, question: str) -> Verdict | None:
        lowered = question.lower()
        if looks_like_instruction(lowered):
            return Verdict("prompt_injection")
        if re.search(_SECRET, lowered):
            return Verdict("secret_request")
        if re.search(_PEOPLE_LISTS, lowered):
            return Verdict("personal_data_request")
        if re.search(_PERSONAL_DATA, lowered) and (
            re.search(_OTHER_PEOPLE, lowered) or re.search(_POSSESSIVE_NAME, lowered)
        ):
            return Verdict("personal_data_request")
        return None


class OutputGuard:
    """Last check before an answer leaves: removes personal data and codes, drops bad citations."""

    def redact(self, answer: str) -> tuple[str, list[str]]:
        """Masks e-mail addresses, phone numbers and numeric codes. Returns the text and what was hit."""
        hits = []
        if _EMAIL.search(answer):
            hits.append("email")
            answer = _EMAIL.sub("[removed e-mail]", answer)
        if _PHONE.search(answer):
            hits.append("phone")
            answer = _PHONE.sub("[removed phone]", answer)
        if _CODE.search(answer):
            hits.append("code")
            answer = _CODE.sub(lambda m: f"{m.group(1)}{m.group(2)}[removed code]", answer)
        return answer, hits

    def valid_citations(self, cited: list[str], retrieved: list[str]) -> list[str]:
        """Keeps only citations of sections that were really retrieved for this question.

        A model can invent a plausible ID like [M9]; citing a section the assistant never saw
        would look grounded while it is not.
        """
        seen = []
        for section_id in cited:
            if section_id in retrieved and section_id not in seen:
                seen.append(section_id)
        return seen
