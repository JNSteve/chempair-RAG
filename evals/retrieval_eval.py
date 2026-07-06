"""Embedding bake-off: score retrieval quality of candidate embedding models
against the gold set in retrieval_gold.yaml, using the real KB chunks.

    python evals/retrieval_eval.py --rag-storage ./rag_storage \
        --models all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5

For each model: embed every KB text chunk and every gold query, rank chunks
by cosine similarity, and score hit@5 / hit@10 / MRR@10 (a "hit" = a top-k
chunk whose source filename matches the query's expect_files pattern).
Writes report.json + comparison.md to --out.

Notes:
- Chunks come from kv_store_text_chunks.json (works on both the legacy KB
  and Phase 2 re-ingested KBs; legacy `tables_` prefixes are stripped for
  matching).
- This scores RETRIEVAL only — no LLM, no API key needed. Run locally where
  HuggingFace is reachable; models download on first use.
- Picking the winner here feeds roadmap Phase 3; changing the serving
  embedding model requires a full re-ingest (vector dims change).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

EVALS_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = "all-MiniLM-L6-v2,BAAI/bge-small-en-v1.5,BAAI/bge-base-en-v1.5"
# Instruction prefixes some models expect on the QUERY side only.
QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
}
TOP_K = 10


def load_chunks(rag_storage: Path) -> list[dict]:
    """Return [{id, file, text}] from a LightRAG storage dir."""
    path = rag_storage / "kv_store_text_chunks.json"
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    chunks = []
    for chunk_id, value in raw.items():
        if not isinstance(value, dict):
            continue
        content = value.get("content")
        if not content:
            continue
        filename = str(value.get("file_path", "")).replace("\\", "/").split("/")[-1]
        chunks.append(
            {
                "id": chunk_id,
                "file": filename.removeprefix("tables_"),
                "text": content,
            }
        )
    return chunks


def query_prefix_for(model_name: str) -> str:
    lowered = model_name.lower()
    for marker, prefix in QUERY_PREFIXES.items():
        if marker in lowered:
            return prefix
    return ""


def first_hit_rank(ranked_files: list[str], expect_pattern: str) -> int | None:
    """1-based rank of the first file matching the pattern, else None."""
    pattern = re.compile(expect_pattern)
    for index, filename in enumerate(ranked_files, 1):
        if pattern.search(filename):
            return index
    return None


def score_query(ranked_files: list[str], expect_pattern: str) -> dict:
    rank = first_hit_rank(ranked_files[:TOP_K], expect_pattern)
    return {
        "first_hit_rank": rank,
        "hit_at_5": rank is not None and rank <= 5,
        "hit_at_10": rank is not None and rank <= 10,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
    }


def summarize_model(per_query: dict[str, dict]) -> dict:
    total = len(per_query)
    return {
        "queries": total,
        "hit_at_5": round(sum(q["hit_at_5"] for q in per_query.values()) / total, 4),
        "hit_at_10": round(sum(q["hit_at_10"] for q in per_query.values()) / total, 4),
        "mrr_at_10": round(
            sum(q["reciprocal_rank"] for q in per_query.values()) / total, 4
        ),
    }


def render_comparison(results: dict[str, dict], per_query: dict[str, dict]) -> str:
    lines = [
        "# Embedding bake-off",
        "",
        "| Model | hit@5 | hit@10 | MRR@10 |",
        "|---|---|---|---|",
    ]
    for model, summary in results.items():
        lines.append(
            f"| {model} | {summary['hit_at_5']:.0%} | {summary['hit_at_10']:.0%} "
            f"| {summary['mrr_at_10']:.3f} |"
        )
    lines.extend(["", "## Misses (no hit in top 10)", ""])
    for model, queries in per_query.items():
        misses = [qid for qid, score in queries.items() if not score["hit_at_10"]]
        lines.append(f"- **{model}**: {', '.join(misses) if misses else 'none'}")
    return "\n".join(lines) + "\n"


def evaluate_model(model_name: str, chunks: list[dict], queries: list[dict]) -> dict:
    """Embed corpus + queries with one model and score every gold query."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    chunk_vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True,
    )
    prefix = query_prefix_for(model_name)
    query_vectors = model.encode(
        [prefix + q["query"] for q in queries], normalize_embeddings=True
    )

    scores = np.asarray(query_vectors) @ np.asarray(chunk_vectors).T
    per_query = {}
    for row, query in zip(scores, queries):
        top_indices = row.argsort()[::-1][:TOP_K]
        ranked_files = [chunks[i]["file"] for i in top_indices]
        per_query[query["id"]] = score_query(ranked_files, query["expect_files"])
    return per_query


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rag-storage", type=Path, default=Path("rag_storage"))
    parser.add_argument("--gold", type=Path, default=EVALS_DIR / "retrieval_gold.yaml")
    parser.add_argument(
        "--models", default=DEFAULT_MODELS, help="comma-separated model names"
    )
    parser.add_argument("--out", type=Path, default=EVALS_DIR / "retrieval_out")
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8") as handle:
        queries = yaml.safe_load(handle)["queries"]
    chunks = load_chunks(args.rag_storage)
    if not chunks:
        print(f"no chunks found in {args.rag_storage}", file=sys.stderr)
        return 1
    print(f"{len(chunks)} chunks, {len(queries)} gold queries")

    results: dict[str, dict] = {}
    per_query: dict[str, dict] = {}
    for model_name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {model_name} ===")
        per_query[model_name] = evaluate_model(model_name, chunks, queries)
        results[model_name] = summarize_model(per_query[model_name])
        print(results[model_name])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps({"summary": results, "per_query": per_query}, indent=2),
        encoding="utf-8",
    )
    (args.out / "comparison.md").write_text(
        render_comparison(results, per_query), encoding="utf-8"
    )
    print(f"\nReports written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
