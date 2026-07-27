"""
Proposal generation + fail-closed validation for /query `proposals[]`.

Mirrors the Enviro-Sage frontend validator
(src/lib/alfieProposals/validate.ts @ main df5fe84f, PR #612 + #614)
exactly: same six operations, caps, ranges, enum sets, exact-key rules,
and the ">=1 field" rule on both update operations. The frontend silently
drops anything malformed, so this module is the only error signal —
every rejection is logged with a reason.

The model controls ONLY operation, payload, rationale, citations. The
server generates `id`, derives `kind` from the operation prefix, and
echoes `baseline` verbatim from the request's proposalContext — the model
can never fabricate staleness anchors.
"""

import json
import logging
import math
import uuid
from dataclasses import dataclass

from context_models import ProposalContext

logger = logging.getLogger("chempair.proposals")

# Brief says emit 0-3 in practice; the frontend consumes at most 5.
MAX_PROPOSALS_PER_ANSWER = 3

MAX_RATIONALE_CHARS = 600
MAX_CITATIONS = 6
MAX_CITATION_CHARS = 200
MAX_ID_CHARS = 128
MAX_NAME_CHARS = 120
MAX_NOTES_CHARS = 600
MAX_NARRATIVE_CHARS = 4000
MAX_LIST_ITEMS = 10
MAX_LIST_ITEM_CHARS = 300
MIN_OFFSET_M = 0.0
MAX_OFFSET_M = 500.0
MAX_DEPTH_M = 100.0
MIN_GRID_SIZE_M = 1
MAX_GRID_SIZE_M = 500

SAQP_OPERATIONS = frozenset(
    {
        "saqp.set_grid_parameters",
        "saqp.add_targeted_point",
        "saqp.update_point_attributes",
    }
)
CSM_OPERATIONS = frozenset(
    {"csm.add_linkage", "csm.update_linkage", "csm.update_narrative"}
)
OPERATIONS = SAQP_OPERATIONS | CSM_OPERATIONS

ANCHOR_TYPES = frozenset({"sample", "saqp_point"})
PRIORITIES = frozenset({"low", "medium", "high"})
RISK_LEVELS = frozenset({"high", "moderate", "low", "incomplete"})
NARRATIVE_SECTIONS = frozenset(
    {"csmSummary", "keyFindings", "recommendations", "exposureJustification"}
)


class ProposalRejected(Exception):
    """One candidate failed validation. Always caught by the generator."""


def _fail(reason: str):
    raise ProposalRejected(reason)


# ---- validation primitives (mirror validate.ts helpers) ----


def _as_record(value, label: str) -> dict:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    return value


def _exact_keys(record: dict, allowed: frozenset, label: str) -> None:
    for key in record:
        if key not in allowed:
            _fail(f"{label} has unexpected field")


def _capped_str(value, label: str, max_len: int) -> str:
    if not isinstance(value, str):
        _fail(f"{label} must be a string")
    trimmed = value.strip()
    if not trimmed:
        _fail(f"{label} must not be empty")
    if len(trimmed) > max_len:
        _fail(f"{label} exceeds {max_len} characters")
    return trimmed


def _opt_capped_str(value, label: str, max_len: int):
    # JSON null and absent both mean "not provided" (TS undefined).
    return None if value is None else _capped_str(value, label, max_len)


def _as_bool(value, label: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{label} must be a boolean")
    return value


def _ranged_num(value, label: str, lo: float, hi: float) -> float:
    # bool is an int subclass in Python — reject it explicitly.
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        _fail(f"{label} must be a finite number")
    if value < lo or value > hi:
        _fail(f"{label} must be between {lo} and {hi}")
    return float(value)


def _as_enum(value, label: str, allowed: frozenset) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(f"{label} is not a recognised value")
    return value


def _known_id(value, label: str, valid_ids: frozenset) -> str:
    candidate = _capped_str(value, label, MAX_ID_CHARS)
    if candidate not in valid_ids:
        _fail(f"{label} references an unknown id")
    return candidate


def _string_list(value, label: str, max_items: int, max_item_chars: int) -> list:
    if not isinstance(value, list):
        _fail(f"{label} must be an array")
    if not value:
        _fail(f"{label} must not be empty")
    if len(value) > max_items:
        _fail(f"{label} exceeds {max_items} items")
    return [
        _capped_str(item, f"{label}[{index}]", max_item_chars)
        for index, item in enumerate(value)
    ]


# ---- artifact extraction (baseline + valid-id sets) ----


@dataclass(frozen=True)
class SaqpArtifact:
    plan_id: str
    updated_at: str
    point_ids: frozenset
    sample_ids: frozenset


@dataclass(frozen=True)
class CsmArtifact:
    csm_id: str
    updated_at: str
    source_ids: frozenset
    pathway_ids: frozenset
    receptor_ids: frozenset
    linkage_ids: frozenset
    media: frozenset


def _ref_ids(refs) -> frozenset:
    if not refs:
        return frozenset()
    return frozenset(
        ref.id.strip()
        for ref in refs
        if isinstance(ref.id, str) and ref.id.strip()
    )


def extract_artifacts(
    ctx: ProposalContext | None,
) -> tuple[SaqpArtifact | None, CsmArtifact | None]:
    """An artifact qualifies only with a non-empty id AND updatedAt — both
    are needed for the baseline the frontend's staleness check requires."""
    saqp = None
    csm = None
    if ctx is None:
        return None, None
    raw_saqp = ctx.saqp
    if (
        raw_saqp
        and isinstance(raw_saqp.planId, str)
        and raw_saqp.planId.strip()
        and isinstance(raw_saqp.updatedAt, str)
        and raw_saqp.updatedAt.strip()
    ):
        saqp = SaqpArtifact(
            plan_id=raw_saqp.planId.strip(),
            updated_at=raw_saqp.updatedAt.strip(),
            point_ids=_ref_ids(raw_saqp.points),
            sample_ids=_ref_ids(raw_saqp.samples),
        )
    raw_csm = ctx.csm
    if (
        raw_csm
        and isinstance(raw_csm.id, str)
        and raw_csm.id.strip()
        and isinstance(raw_csm.updatedAt, str)
        and raw_csm.updatedAt.strip()
    ):
        csm = CsmArtifact(
            csm_id=raw_csm.id.strip(),
            updated_at=raw_csm.updatedAt.strip(),
            source_ids=_ref_ids(raw_csm.sources),
            pathway_ids=_ref_ids(raw_csm.pathways),
            receptor_ids=_ref_ids(raw_csm.receptors),
            linkage_ids=_ref_ids(raw_csm.linkages),
            media=frozenset(
                medium.strip()
                for medium in (raw_csm.media or [])
                if isinstance(medium, str) and medium.strip()
            ),
        )
    return saqp, csm


# ---- SAQP payload validators ----


def _validate_set_grid_parameters(payload: dict) -> dict:
    _exact_keys(payload, frozenset({"gridEnabled", "gridSizeM"}), "payload")
    grid_enabled = _as_bool(payload.get("gridEnabled"), "payload.gridEnabled")
    grid_size = _ranged_num(
        payload.get("gridSizeM"), "payload.gridSizeM", MIN_GRID_SIZE_M, MAX_GRID_SIZE_M
    )
    if grid_size != int(grid_size):
        _fail("payload.gridSizeM must be an integer")
    return {"gridEnabled": grid_enabled, "gridSizeM": int(grid_size)}


def _validate_anchor(value, saqp: SaqpArtifact) -> dict:
    record = _as_record(value, "payload.anchor")
    _exact_keys(record, frozenset({"type", "id"}), "payload.anchor")
    anchor_type = _as_enum(record.get("type"), "payload.anchor.type", ANCHOR_TYPES)
    valid_ids = saqp.sample_ids if anchor_type == "sample" else saqp.point_ids
    anchor_id = _known_id(record.get("id"), "payload.anchor.id", valid_ids)
    return {"type": anchor_type, "id": anchor_id}


def _validate_add_targeted_point(payload: dict, saqp: SaqpArtifact) -> dict:
    _exact_keys(
        payload,
        frozenset(
            {
                "anchor",
                "offsetM",
                "bearingDeg",
                "sampleName",
                "depthFromM",
                "depthToM",
                "matrix",
                "priority",
            }
        ),
        "payload",
    )
    anchor = _validate_anchor(payload.get("anchor"), saqp)
    offset_m = _ranged_num(
        payload.get("offsetM"), "payload.offsetM", MIN_OFFSET_M, MAX_OFFSET_M
    )
    bearing_deg = _ranged_num(payload.get("bearingDeg"), "payload.bearingDeg", 0, 360)
    sample_name = _capped_str(
        payload.get("sampleName"), "payload.sampleName", MAX_NAME_CHARS
    )
    depth_from = _ranged_num(
        payload.get("depthFromM"), "payload.depthFromM", 0, MAX_DEPTH_M
    )
    depth_to = _ranged_num(payload.get("depthToM"), "payload.depthToM", 0, MAX_DEPTH_M)
    if depth_from > depth_to:
        _fail("payload.depthFromM must be <= payload.depthToM")
    matrix = _capped_str(payload.get("matrix"), "payload.matrix", MAX_NAME_CHARS)
    priority = _as_enum(payload.get("priority"), "payload.priority", PRIORITIES)
    return {
        "anchor": anchor,
        "offsetM": offset_m,
        "bearingDeg": bearing_deg,
        "sampleName": sample_name,
        "depthFromM": depth_from,
        "depthToM": depth_to,
        "matrix": matrix,
        "priority": priority,
    }


def _validate_update_point_attributes(payload: dict, saqp: SaqpArtifact) -> dict:
    _exact_keys(
        payload,
        frozenset(
            {"pointId", "depthFromM", "depthToM", "matrix", "priority", "notes"}
        ),
        "payload",
    )
    result = {
        "pointId": _known_id(payload.get("pointId"), "payload.pointId", saqp.point_ids)
    }
    if payload.get("depthFromM") is not None:
        result["depthFromM"] = _ranged_num(
            payload.get("depthFromM"), "payload.depthFromM", 0, MAX_DEPTH_M
        )
    if payload.get("depthToM") is not None:
        result["depthToM"] = _ranged_num(
            payload.get("depthToM"), "payload.depthToM", 0, MAX_DEPTH_M
        )
    if (
        "depthFromM" in result
        and "depthToM" in result
        and result["depthFromM"] > result["depthToM"]
    ):
        _fail("payload.depthFromM must be <= payload.depthToM")
    matrix = _opt_capped_str(payload.get("matrix"), "payload.matrix", MAX_NAME_CHARS)
    if matrix is not None:
        result["matrix"] = matrix
    if payload.get("priority") is not None:
        result["priority"] = _as_enum(
            payload.get("priority"), "payload.priority", PRIORITIES
        )
    notes = _opt_capped_str(payload.get("notes"), "payload.notes", MAX_NOTES_CHARS)
    if notes is not None:
        result["notes"] = notes
    if len(result) == 1:
        _fail("No fields to update")
    return result


# ---- CSM payload validators ----


def _validate_add_linkage(payload: dict, csm: CsmArtifact) -> dict:
    _exact_keys(
        payload,
        frozenset(
            {
                "sourceId",
                "pathwayId",
                "receptorId",
                "riskLevel",
                "isComplete",
                "reasoning",
            }
        ),
        "payload",
    )
    return {
        "sourceId": _known_id(
            payload.get("sourceId"), "payload.sourceId", csm.source_ids
        ),
        "pathwayId": _known_id(
            payload.get("pathwayId"), "payload.pathwayId", csm.pathway_ids
        ),
        "receptorId": _known_id(
            payload.get("receptorId"), "payload.receptorId", csm.receptor_ids
        ),
        "riskLevel": _as_enum(
            payload.get("riskLevel"), "payload.riskLevel", RISK_LEVELS
        ),
        "isComplete": _as_bool(payload.get("isComplete"), "payload.isComplete"),
        "reasoning": _capped_str(
            payload.get("reasoning"), "payload.reasoning", MAX_NOTES_CHARS
        ),
    }


def _validate_update_linkage(payload: dict, csm: CsmArtifact) -> dict:
    _exact_keys(
        payload, frozenset({"linkageId", "riskLevel", "isComplete", "notes"}), "payload"
    )
    result = {
        "linkageId": _known_id(
            payload.get("linkageId"), "payload.linkageId", csm.linkage_ids
        )
    }
    if payload.get("riskLevel") is not None:
        result["riskLevel"] = _as_enum(
            payload.get("riskLevel"), "payload.riskLevel", RISK_LEVELS
        )
    if payload.get("isComplete") is not None:
        result["isComplete"] = _as_bool(
            payload.get("isComplete"), "payload.isComplete"
        )
    notes = _opt_capped_str(payload.get("notes"), "payload.notes", MAX_NOTES_CHARS)
    if notes is not None:
        result["notes"] = notes
    if len(result) == 1:
        # Mirrors enviro-sage PR #614 (validate.ts 'No fields to update').
        _fail("No fields to update")
    return result


def _validate_update_narrative(payload: dict, csm: CsmArtifact) -> dict:
    _exact_keys(payload, frozenset({"section", "medium", "text", "items"}), "payload")
    section = _as_enum(payload.get("section"), "payload.section", NARRATIVE_SECTIONS)
    if section == "csmSummary":
        return {
            "section": section,
            "text": _capped_str(
                payload.get("text"), "payload.text", MAX_NARRATIVE_CHARS
            ),
        }
    if section in ("keyFindings", "recommendations"):
        return {
            "section": section,
            "items": _string_list(
                payload.get("items"),
                "payload.items",
                MAX_LIST_ITEMS,
                MAX_LIST_ITEM_CHARS,
            ),
        }
    medium = _capped_str(payload.get("medium"), "payload.medium", MAX_NAME_CHARS)
    if medium not in csm.media:
        _fail("payload.medium is not an affected medium")
    return {
        "section": section,
        "medium": medium,
        "text": _capped_str(payload.get("text"), "payload.text", MAX_NARRATIVE_CHARS),
    }
