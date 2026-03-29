from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


def normalise_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def compact_text(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalise_text(value))


def tokenize_text(value: str | None) -> list[str]:
    return TOKEN_RE.findall(normalise_text(value))


def significant_tokens(
    value: str | None,
    ignored_tokens: set[str] | None = None,
) -> list[str]:
    ignored = ignored_tokens or set()
    return [
        token
        for token in tokenize_text(value)
        if token not in ignored and (len(token) >= 3 or any(char.isdigit() for char in token))
    ]


def question_mentions_candidate(
    question: str,
    candidate: str,
    ignored_tokens: set[str] | None = None,
) -> bool:
    question_key = normalise_text(question)
    candidate_key = normalise_text(candidate)
    if not candidate_key:
        return False

    if candidate_key in question_key:
        return True

    compact_question = compact_text(question)
    compact_candidate = compact_text(candidate)
    if compact_candidate and compact_candidate in compact_question:
        return True

    candidate_tokens = significant_tokens(candidate, ignored_tokens=ignored_tokens)
    question_tokens = set(tokenize_text(question))
    if candidate_tokens and all(token in question_tokens for token in candidate_tokens):
        return True

    return False


def resolve_candidate_matches(
    question: str,
    candidates: list[str],
    ignored_tokens: set[str] | None = None,
) -> list[str]:
    strong_matches: list[str] = []
    weak_matches: list[str] = []
    question_tokens = set(tokenize_text(question))
    seen: set[str] = set()

    for candidate in candidates:
        candidate_key = normalise_text(candidate)
        if not candidate_key or candidate_key in seen:
            continue
        seen.add(candidate_key)

        if question_mentions_candidate(question, candidate, ignored_tokens=ignored_tokens):
            strong_matches.append(candidate)
            continue

        candidate_tokens = significant_tokens(candidate, ignored_tokens=ignored_tokens)
        if candidate_tokens and candidate_tokens[0] in question_tokens:
            weak_matches.append(candidate)

    return strong_matches or weak_matches
