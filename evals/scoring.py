"""Pure scoring logic for the golden eval harness.

No I/O here — evaluate_response/summarize/render_scorecard operate on plain
dicts so they are unit-testable without a live backend (see
tests/test_eval_scoring.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A locator counts as "exact" when it points at a table or page, rather than
# the regex-fallback values the server emits today ("source passage", chunk
# ids). This check is observational (severity: info) until roadmap Phase 4.
EXACT_LOCATOR_PATTERN = re.compile(r"(?i)\btable\s+\S+|\bp(?:age)?\.?\s*\d")
FALLBACK_LOCATOR = "source passage"

BLOCKING = "blocking"
INFO = "info"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class QuestionResult:
    question_id: str
    category: str
    severity: str
    checks: list[Check] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and all(check.passed for check in self.checks)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "severity": self.severity,
            "passed": self.passed,
            "error": self.error,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail}
                for c in self.checks
            ],
        }


def _citations(response: dict) -> list[dict]:
    citations = response.get("citations")
    return citations if isinstance(citations, list) else []


def _check_route(expect: dict, response: dict, checks: list[Check]) -> None:
    allowed = expect.get("route_used")
    if not allowed:
        return
    actual = response.get("route_used")
    checks.append(
        Check(
            "route_used",
            actual in allowed,
            f"expected one of {allowed}, got {actual!r}",
        )
    )


def _check_answer_content(expect: dict, response: dict, checks: list[Check]) -> None:
    answer = str(response.get("answer") or "")
    lowered = answer.lower()
    for needle in expect.get("must_include", []):
        checks.append(
            Check(
                f"must_include:{needle}",
                str(needle).lower() in lowered,
                f"answer must contain {needle!r}",
            )
        )
    for needle in expect.get("must_not_include", []):
        checks.append(
            Check(
                f"must_not_include:{needle}",
                str(needle).lower() not in lowered,
                f"answer must not contain {needle!r}",
            )
        )


def _check_citations(expect: dict, response: dict, checks: list[Check]) -> None:
    citations = _citations(response)

    min_citations = expect.get("min_citations")
    if min_citations is not None:
        checks.append(
            Check(
                "min_citations",
                len(citations) >= min_citations,
                f"expected >= {min_citations} citations, got {len(citations)}",
            )
        )

    max_citations = expect.get("max_citations")
    if max_citations is not None:
        checks.append(
            Check(
                "max_citations",
                len(citations) <= max_citations,
                f"expected <= {max_citations} citations, got {len(citations)}",
            )
        )

    source_pattern = expect.get("citation_source_pattern")
    if source_pattern:
        pattern = re.compile(source_pattern)
        matched = any(
            pattern.search(str(c.get("source") or ""))
            or pattern.search(str(c.get("title") or ""))
            for c in citations
        )
        checks.append(
            Check(
                "citation_source_pattern",
                matched,
                f"no citation source/title matches {source_pattern!r} "
                f"(sources: {[c.get('source') for c in citations]})",
            )
        )

    locator_forbidden = expect.get("citation_locator_must_not_match")
    if locator_forbidden:
        pattern = re.compile(locator_forbidden)
        offenders = [
            c.get("locator")
            for c in citations
            if pattern.search(str(c.get("locator") or ""))
        ]
        checks.append(
            Check(
                "citation_locator_must_not_match",
                not offenders,
                f"locators matching {locator_forbidden!r}: {offenders}",
            )
        )

    if expect.get("exact_locator"):
        inexact = [
            c.get("locator")
            for c in citations
            if str(c.get("locator") or "").strip().lower() == FALLBACK_LOCATOR
            or not EXACT_LOCATOR_PATTERN.search(str(c.get("locator") or ""))
        ]
        checks.append(
            Check(
                "exact_locator",
                bool(citations) and not inexact,
                f"citations without a table/page locator: {inexact}"
                if citations
                else "no citations to check",
            )
        )


def _check_grounded(expect: dict, response: dict, checks: list[Check]) -> None:
    expected = expect.get("grounded")
    if expected is None:
        return
    actual = response.get("grounded")
    checks.append(
        Check(
            "grounded",
            actual is expected,
            f"expected grounded={expected}, got {actual!r}",
        )
    )


def evaluate_response(
    question: dict,
    response: dict | None,
    error: str | None = None,
) -> QuestionResult:
    """Score one backend response against a question's expectations."""
    result = QuestionResult(
        question_id=question["id"],
        category=question.get("category", "uncategorized"),
        severity=question.get("severity", BLOCKING),
    )
    if error is not None:
        result.error = error
        return result
    if not isinstance(response, dict):
        result.error = "no response payload"
        return result

    expect = question.get("expect", {}) or {}
    _check_route(expect, response, result.checks)
    _check_answer_content(expect, response, result.checks)
    _check_citations(expect, response, result.checks)
    _check_grounded(expect, response, result.checks)
    return result


def summarize(results: list[QuestionResult]) -> dict:
    """Aggregate results: overall + per-category pass rates over blocking
    questions; info questions reported separately."""
    blocking = [r for r in results if r.severity == BLOCKING]
    info = [r for r in results if r.severity != BLOCKING]

    def _rate(items: list[QuestionResult]) -> dict:
        passed = sum(1 for r in items if r.passed)
        return {
            "total": len(items),
            "passed": passed,
            "pass_rate": round(passed / len(items), 4) if items else None,
        }

    categories: dict[str, list[QuestionResult]] = {}
    for r in blocking:
        categories.setdefault(r.category, []).append(r)

    return {
        "overall": _rate(blocking),
        "categories": {
            name: _rate(items) for name, items in sorted(categories.items())
        },
        "info": _rate(info),
        "errors": [r.question_id for r in results if r.error],
    }


def render_scorecard(summary: dict, results: list[QuestionResult]) -> str:
    """Markdown scorecard for committing alongside the JSON report."""
    lines = ["# Golden eval scorecard", ""]
    overall = summary["overall"]
    lines.append(
        f"**Overall (blocking): {overall['passed']}/{overall['total']} passed"
        + (
            f" ({overall['pass_rate']:.0%})**"
            if overall["pass_rate"] is not None
            else "**"
        )
    )
    lines.extend(["", "| Category | Passed | Total |", "|---|---|---|"])
    for name, stats in summary["categories"].items():
        lines.append(f"| {name} | {stats['passed']} | {stats['total']} |")
    info = summary["info"]
    if info["total"]:
        lines.append(f"| _info (non-blocking)_ | {info['passed']} | {info['total']} |")

    failures = [r for r in results if not r.passed]
    if failures:
        lines.extend(["", "## Failures", ""])
        for r in failures:
            marker = "" if r.severity == BLOCKING else " _(info)_"
            lines.append(f"### {r.question_id}{marker}")
            if r.error:
                lines.append(f"- error: {r.error}")
            for check in r.checks:
                if not check.passed:
                    lines.append(f"- {check.name}: {check.detail}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
