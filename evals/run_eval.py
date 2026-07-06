"""Run the golden eval question set against a live Chempair RAG backend.

Usage:
    python evals/run_eval.py --base-url https://rag.example.com \
        --api-key $RAG_API_KEY --out evals/baseline/

Sends each question in golden_questions.yaml to POST /query with its context
profile, scores the response (evals/scoring.py), and writes report.json +
scorecard.md to --out. Exit code is nonzero when the blocking pass rate falls
below --min-score (default 0.0, i.e. record-only — set a threshold in CI).

Each question costs one backend/LLM call; use --filter to run a subset while
iterating.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring import evaluate_response, render_scorecard, summarize  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent
# Context fields the runner may override per question, mirroring what the
# enviro-sage frontend would set for that question.
OVERRIDABLE_CONTEXT_FIELDS = {
    "questionIntent",
    "preferredAnswerShape",
    "targetAnalytes",
}


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_context(profiles: dict, question: dict) -> dict | None:
    profile_name = question.get("profile", "none")
    if profile_name in (None, "none"):
        return None
    if profile_name not in profiles:
        raise KeyError(f"question {question['id']!r}: unknown profile {profile_name!r}")
    context = copy.deepcopy(profiles[profile_name])
    overrides = question.get("overrides") or {}
    unknown = set(overrides) - OVERRIDABLE_CONTEXT_FIELDS
    if unknown:
        raise KeyError(
            f"question {question['id']!r}: unsupported overrides {sorted(unknown)}"
        )
    context.update(overrides)
    return context


def post_query(
    base_url: str,
    api_key: str | None,
    payload: dict,
    timeout: float,
) -> dict:
    url = base_url.rstrip("/") + "/query"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run(args: argparse.Namespace) -> int:
    spec = load_yaml(args.questions)
    profiles = load_yaml(args.profiles)["profiles"]
    questions = spec["questions"]

    if args.filter:
        needle = args.filter.lower()
        selected_ids = {
            q["id"]
            for q in questions
            if needle in q["id"].lower() or needle == q.get("category")
        }
        # Keep parents of selected follow-ups so sessions can be threaded.
        for q in questions:
            if q["id"] in selected_ids and q.get("follow_up_of"):
                selected_ids.add(q["follow_up_of"])
        questions = [q for q in questions if q["id"] in selected_ids]
        if not questions:
            print(f"No questions match filter {args.filter!r}", file=sys.stderr)
            return 2

    api_key = args.api_key or os.environ.get("RAG_API_KEY")
    session_ids: dict[str, str | None] = {}
    results = []
    responses: dict[str, dict] = {}

    for question in questions:
        qid = question["id"]
        payload: dict = {"question": question["question"], "mode": args.mode}
        parent = question.get("follow_up_of")
        if parent:
            payload["session_id"] = session_ids.get(parent)
        context = build_context(profiles, question)
        if context is not None:
            payload["context"] = context

        response: dict | None = None
        error: str | None = None
        try:
            response = post_query(args.base_url, api_key, payload, args.timeout)
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"

        if response is not None:
            session_ids[qid] = response.get("session_id")
            responses[qid] = response

        result = evaluate_response(question, response, error)
        results.append(result)
        status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
        print(f"[{status}] {qid}")

    summary = summarize(results)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "base_url": args.base_url,
        "mode": args.mode,
        "summary": summary,
        "results": [r.to_dict() for r in results],
        "responses": responses if args.include_responses else None,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "scorecard.md").write_text(
        render_scorecard(summary, results), encoding="utf-8"
    )

    overall = summary["overall"]
    rate = overall["pass_rate"] if overall["pass_rate"] is not None else 1.0
    print(
        f"\nBlocking: {overall['passed']}/{overall['total']} passed."
        f" Report written to {out_dir}/"
    )
    return 0 if rate >= args.min_score else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True, help="Backend base URL")
    parser.add_argument("--api-key", help="Bearer key (default: $RAG_API_KEY)")
    parser.add_argument(
        "--questions", type=Path, default=EVALS_DIR / "golden_questions.yaml"
    )
    parser.add_argument(
        "--profiles", type=Path, default=EVALS_DIR / "context_profiles.yaml"
    )
    parser.add_argument(
        "--out", default=str(EVALS_DIR / "out"), help="Report output directory"
    )
    parser.add_argument("--mode", default="hybrid", help="Query mode (default: hybrid)")
    parser.add_argument(
        "--filter", help="Only run questions whose id contains / category equals this"
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0, help="Per-request timeout seconds"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Fail (exit 1) if blocking pass rate is below this (default 0.0: record-only)",
    )
    parser.add_argument(
        "--include-responses",
        action="store_true",
        help="Embed full backend responses in report.json (for debugging)",
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
