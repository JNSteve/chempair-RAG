from __future__ import annotations

import re


TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")

TYPO_NORMALISATIONS = (
    (r"\bcontaminents\b", "contaminants"),
    (r"\bcontaminent\b", "contaminant"),
    (r"\bconcrned\b", "concerned"),
    (r"\bconcernes\b", "concerns"),
    (r"\bcontmaination\b", "contamination"),
    (r"\bcontaimination\b", "contamination"),
    (r"\bcontaminaton\b", "contamination"),
    (r"\bexceedences\b", "exceedances"),
    (r"\bexceedence\b", "exceedance"),
    (r"\bexcedances\b", "exceedances"),
    (r"\bexcedance\b", "exceedance"),
    (r"\bexceedence\b", "exceedance"),
    (r"\bexc?eedings?\b", "exceeding"),
    (r"\barsnic\b", "arsenic"),
    (r"\barseninc\b", "arsenic"),
    (r"\barsenik\b", "arsenic"),
    (r"\bpfa'?s\b", "pfas"),
    (r"\bpfass\b", "pfas"),
    (r"\bpfos\b", "pfos"),
    (r"\bhydrocarbns\b", "hydrocarbons"),
    (r"\bhydrocabons\b", "hydrocarbons"),
    (r"\bhydrocarbn\b", "hydrocarbon"),
    (r"\bmetels\b", "metals"),
    (r"\bmetalic\b", "metallic"),
)

ANALYTE_ALIASES = {
    "as": ("arsenic",),
    "benzo(a)pyrene": ("benzo a pyrene", "bap"),
    "bap": ("benzo(a)pyrene", "benzo a pyrene"),
    "pfas": ("pfos", "pfoa"),
    "pfos": ("pfas",),
    "pfoa": ("pfas",),
    "trh": ("hydrocarbons", "total recoverable hydrocarbons"),
    "total recoverable hydrocarbons": ("trh", "hydrocarbons"),
    "hydrocarbons": ("trh", "total recoverable hydrocarbons"),
    "metals": ("heavy metals",),
}


def normalise_text(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip().lower()
    for pattern, replacement in TYPO_NORMALISATIONS:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text)


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
        if token not in ignored
        and (len(token) >= 3 or any(char.isdigit() for char in token))
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

    if len(candidate_key) <= 2:
        if re.search(rf"\b{re.escape(candidate_key)}\b", question_key):
            return True
    elif candidate_key in question_key:
        return True

    for alias in ANALYTE_ALIASES.get(candidate_key, ()):
        alias_key = normalise_text(alias)
        if len(alias_key) <= 2:
            if re.search(rf"\b{re.escape(alias_key)}\b", question_key):
                return True
        elif alias_key in question_key:
            return True

    compact_question = compact_text(question)
    compact_candidate = compact_text(candidate)
    if (
        compact_candidate
        and len(compact_candidate) > 2
        and compact_candidate in compact_question
    ):
        return True

    for alias in ANALYTE_ALIASES.get(candidate_key, ()):
        compact_alias = compact_text(alias)
        if (
            compact_alias
            and len(compact_alias) > 2
            and compact_alias in compact_question
        ):
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

        if question_mentions_candidate(
            question, candidate, ignored_tokens=ignored_tokens
        ):
            strong_matches.append(candidate)
            continue

        candidate_tokens = significant_tokens(candidate, ignored_tokens=ignored_tokens)
        if candidate_tokens and candidate_tokens[0] in question_tokens:
            weak_matches.append(candidate)

    return strong_matches or weak_matches
