import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from scoring import evaluate_response, render_scorecard, summarize  # noqa: E402


def _question(expect: dict, severity: str = "blocking") -> dict:
    return {
        "id": "q1",
        "category": "project_evidence",
        "severity": severity,
        "expect": expect,
    }


def _response(**overrides) -> dict:
    base = {
        "answer": "Arsenic exceeds NEPM 2013 HIL-A at BH20 = 870 mg/kg (22 total exceedances).",
        "route_used": "project_only",
        "grounded": False,
        "citations": [],
    }
    base.update(overrides)
    return base


def test_passing_project_only_response():
    result = evaluate_response(
        _question(
            {
                "route_used": ["project_only"],
                "must_include": ["Arsenic", "22"],
                "must_not_include": ["no exceedances"],
                "max_citations": 0,
            }
        ),
        _response(),
    )
    assert result.passed
    assert all(check.passed for check in result.checks)


def test_must_include_is_case_insensitive():
    result = evaluate_response(_question({"must_include": ["arsenic"]}), _response())
    assert result.passed


def test_wrong_route_fails():
    result = evaluate_response(
        _question({"route_used": ["hybrid"]}), _response(route_used="project_only")
    )
    assert not result.passed
    failed = [c for c in result.checks if not c.passed]
    assert failed[0].name == "route_used"


def test_forbidden_content_fails():
    result = evaluate_response(
        _question({"must_not_include": ["there are no exceedances"]}),
        _response(answer="Good news, there are no exceedances here."),
    )
    assert not result.passed


def test_citation_expectations():
    citations = [
        {
            "source": "NEPM_2013.pdf",
            "title": "NEPM 2013",
            "locator": "Table 1A(1), p. 14",
        },
    ]
    result = evaluate_response(
        _question(
            {
                "min_citations": 1,
                "citation_source_pattern": "(?i)nepm",
                "exact_locator": True,
                "grounded": True,
            }
        ),
        _response(citations=citations, grounded=True),
    )
    assert result.passed


def test_fallback_locator_fails_exact_locator():
    citations = [{"source": "NEPM_2013.pdf", "locator": "source passage"}]
    result = evaluate_response(
        _question({"exact_locator": True}), _response(citations=citations)
    )
    assert not result.passed


def test_exact_locator_with_no_citations_fails():
    result = evaluate_response(
        _question({"exact_locator": True}), _response(citations=[])
    )
    assert not result.passed


def test_citation_locator_must_not_match():
    citations = [{"source": "NEPM_2013.pdf", "locator": "Table 99Z"}]
    result = evaluate_response(
        _question({"citation_locator_must_not_match": "99Z"}),
        _response(citations=citations),
    )
    assert not result.passed


def test_max_citations_exceeded_fails():
    result = evaluate_response(
        _question({"max_citations": 0}),
        _response(citations=[{"source": "NEPM_2013.pdf", "locator": "p. 3"}]),
    )
    assert not result.passed


def test_transport_error_is_reported_not_scored():
    result = evaluate_response(
        _question({"must_include": ["Arsenic"]}), None, error="HTTP 503"
    )
    assert not result.passed
    assert result.error == "HTTP 503"
    assert result.checks == []


def test_summarize_separates_blocking_and_info():
    passing = evaluate_response(_question({"must_include": ["Arsenic"]}), _response())
    failing_info = evaluate_response(
        _question({"must_include": ["missing"]}, severity="info"), _response()
    )
    summary = summarize([passing, failing_info])
    assert summary["overall"] == {"total": 1, "passed": 1, "pass_rate": 1.0}
    assert summary["info"]["total"] == 1
    assert summary["info"]["passed"] == 0


def test_scorecard_lists_failures():
    failing = evaluate_response(
        _question({"must_include": ["missing text"]}), _response()
    )
    summary = summarize([failing])
    scorecard = render_scorecard(summary, [failing])
    assert "0/1 passed" in scorecard
    assert "must_include:missing text" in scorecard
