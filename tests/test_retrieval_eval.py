"""Tests for the retrieval bake-off scoring + gold set (no embeddings needed:
sentence-transformers is only imported inside evaluate_model)."""

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals"))

import retrieval_eval as re_eval  # noqa: E402


def test_first_hit_rank_and_scoring():
    ranked = ["Other.pdf", "NEPM_2013.pdf", "Another.pdf"]
    assert re_eval.first_hit_rank(ranked, "(?i)nepm") == 2
    assert re_eval.first_hit_rank(ranked, "(?i)pfas") is None

    score = re_eval.score_query(ranked, "(?i)nepm")
    assert score == {
        "first_hit_rank": 2,
        "hit_at_5": True,
        "hit_at_10": True,
        "reciprocal_rank": 0.5,
    }
    miss = re_eval.score_query(ranked, "(?i)pfas")
    assert miss["first_hit_rank"] is None
    assert miss["reciprocal_rank"] == 0.0


def test_rank_outside_top_k_is_a_miss():
    ranked = ["x.pdf"] * 10 + ["NEPM_2013.pdf"]
    assert re_eval.score_query(ranked, "(?i)nepm")["hit_at_10"] is False


def test_summarize_model_averages():
    per_query = {
        "a": {"hit_at_5": True, "hit_at_10": True, "reciprocal_rank": 1.0},
        "b": {"hit_at_5": False, "hit_at_10": True, "reciprocal_rank": 0.1},
    }
    summary = re_eval.summarize_model(per_query)
    assert summary == {
        "queries": 2,
        "hit_at_5": 0.5,
        "hit_at_10": 1.0,
        "mrr_at_10": 0.55,
    }


def test_query_prefix_only_for_bge_models():
    assert re_eval.query_prefix_for("BAAI/bge-small-en-v1.5").startswith("Represent")
    assert re_eval.query_prefix_for("all-MiniLM-L6-v2") == ""


def test_load_chunks_strips_legacy_tables_prefix(tmp_path):
    store = {
        "chunk-1": {"content": "text a", "file_path": "my_pdfs/NEPM_2013.pdf"},
        "chunk-2": {"content": "table text", "file_path": "tables_NEPM_2013.pdf"},
        "chunk-3": {"content": "", "file_path": "empty.pdf"},
        "bad": "not-a-dict",
    }
    (tmp_path / "kv_store_text_chunks.json").write_text(
        json.dumps(store), encoding="utf-8"
    )
    chunks = re_eval.load_chunks(tmp_path)
    assert [(c["id"], c["file"]) for c in chunks] == [
        ("chunk-1", "NEPM_2013.pdf"),
        ("chunk-2", "NEPM_2013.pdf"),
    ]


def test_gold_set_is_well_formed():
    gold = yaml.safe_load(
        (ROOT / "evals" / "retrieval_gold.yaml").read_text(encoding="utf-8")
    )
    queries = gold["queries"]
    assert len(queries) >= 15
    ids = [q["id"] for q in queries]
    assert len(set(ids)) == len(ids)
    for query in queries:
        assert query["query"].strip()
        re.compile(query["expect_files"])  # every pattern must compile


def test_render_comparison_lists_misses():
    results = {"model-a": {"hit_at_5": 0.5, "hit_at_10": 1.0, "mrr_at_10": 0.75}}
    per_query = {
        "model-a": {
            "q1": {"hit_at_10": True},
            "q2": {"hit_at_10": False},
        }
    }
    markdown = re_eval.render_comparison(results, per_query)
    assert "| model-a | 50% | 100% | 0.750 |" in markdown
    assert "**model-a**: q2" in markdown
