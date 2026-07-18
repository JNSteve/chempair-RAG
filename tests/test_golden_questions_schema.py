"""Schema checks for the golden question set — keeps the eval data honest
without needing a live backend."""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

from run_eval import OVERRIDABLE_CONTEXT_FIELDS, build_context  # noqa: E402

VALID_CATEGORIES = {
    "project_evidence",
    "threshold_lookup",
    "criteria_explanation",
    "source_pathway",
    "jurisdiction",
    "follow_up",
    "injection",
    "guardrail",
    "table_lookup",
    "practice_guidance",
    "map_spatial",
    "saqp",
}
VALID_EXPECT_FIELDS = {
    "route_used",
    "must_include",
    "must_not_include",
    "must_match",
    "must_not_assert",
    "min_citations",
    "max_citations",
    "citation_source_pattern",
    "citation_locator_must_not_match",
    "exact_locator",
    "grounded",
}
PACKAGE_4_GOLDEN_IDS = {
    "contaminants_project",
    "exceedances_project",
    "arsenic_sources",
    "injection_original",
    "criteria_applied",
}


def load_spec() -> dict:
    with open(ROOT / "evals" / "golden_questions.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_profiles() -> dict:
    with open(ROOT / "evals" / "context_profiles.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["profiles"]


def test_question_ids_are_unique():
    questions = load_spec()["questions"]
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_original_package4_golden_questions_are_preserved():
    ids = {q["id"] for q in load_spec()["questions"]}
    assert PACKAGE_4_GOLDEN_IDS <= ids


def test_every_question_is_well_formed():
    questions = load_spec()["questions"]
    profiles = load_profiles()
    seen: set[str] = set()
    for question in questions:
        assert question.get("question", "").strip(), f"{question['id']}: empty question"
        assert question.get("category") in VALID_CATEGORIES, (
            f"{question['id']}: bad category"
        )
        assert question.get("severity", "blocking") in {"blocking", "info"}

        profile = question.get("profile", "none")
        assert profile == "none" or profile in profiles, (
            f"{question['id']}: unknown profile"
        )

        parent = question.get("follow_up_of")
        if parent:
            assert parent in seen, (
                f"{question['id']}: follow_up_of must reference an earlier question"
            )
        seen.add(question["id"])

        expect = question.get("expect") or {}
        unknown = set(expect) - VALID_EXPECT_FIELDS
        assert not unknown, f"{question['id']}: unknown expect fields {sorted(unknown)}"
        assert expect, f"{question['id']}: no expectations declared"

        overrides = question.get("overrides") or {}
        assert set(overrides) <= OVERRIDABLE_CONTEXT_FIELDS, (
            f"{question['id']}: unsupported overrides"
        )


def test_expectation_regexes_compile():
    for question in load_spec()["questions"]:
        expect = question.get("expect") or {}
        for fieldname in ("citation_source_pattern", "citation_locator_must_not_match"):
            pattern = expect.get(fieldname)
            if pattern:
                re.compile(pattern)


def test_contexts_build_for_every_question():
    profiles = load_profiles()
    for question in load_spec()["questions"]:
        context = build_context(profiles, question)
        if question.get("profile", "none") == "none":
            assert context is None
        else:
            assert context["schemaVersion"] in (4, 5)
            assert context["projectState"]["project"]["projectName"] == "Ducat"
            for key, value in (question.get("overrides") or {}).items():
                assert context[key] == value


def test_ducat_profile_matches_frontend_schema_v4_shape():
    profile = load_profiles()["ducat_soil"]
    assert (
        profile["projectEvidenceSummary"]["exceedanceSummary"]["totalExceedances"] == 22
    )
    top = profile["projectEvidenceSummary"]["topExceedancesByMagnitude"]
    assert top[0] == {
        "analyte": "Arsenic",
        "sampleCode": "BH20",
        "criterion": "NEPM 2013 HIL-A residential",
        "value": 870,
        "criterionValue": 100,
        "exceedanceFactor": 8.7,
        "isHotspot": True,
        "unit": "mg/kg",
    }
    assert profile["retrievalContext"] == {
        "matchedAnalytes": [],
        "matchedSampleCodes": [],
        "questionTokens": [],
        "retrievedRows": [],
    }
