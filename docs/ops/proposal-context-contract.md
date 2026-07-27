# `proposalContext` — request contract for /query proposals

The backend emits `proposals[]` only when the frontend sends this optional
block inside `context`. Field names here are the coordination contract with
Enviro-Sage (the handoff brief's §4 "known gap" — the frontend follow-up must
send exactly these):

```json
"proposalContext": {
  "saqp": {
    "planId": "<saqp_plans.id>",
    "updatedAt": "<plan updated_at, echoed verbatim into baseline>",
    "points":  [{ "id": "<saqp_points.id>", "label": "SP01" }],
    "samples": [{ "id": "<sample id>", "label": "BH01_0.5" }]
  },
  "csm": {
    "id": "<CSM id>",
    "updatedAt": "<CSM updatedAt, echoed verbatim into baseline>",
    "sources":   [{ "id": "s1", "label": "Former UST" }],
    "pathways":  [{ "id": "p1", "label": "Leaching" }],
    "receptors": [{ "id": "r1", "label": "Groundwater users" }],
    "linkages":  [{ "id": "l1", "label": "UST to GW", "origin": "generated" }],
    "media": ["Soil", "Groundwater"]
  }
}
```

Rules:

- An artifact (saqp / csm) only participates if BOTH its id and updatedAt
  are non-empty strings — they become the proposal `baseline` verbatim.
- `label` is display-only context for the model; `origin` on linkages is
  "generated" or "consultant" (the model avoids consultant-authored ones).
- Ids are the only values the model may reference; anything else is
  rejected server-side (proposals.py mirrors the frontend validator at
  enviro-sage main df5fe84f, PR #612 + #614).
- Server emits at most 3 proposals; the frontend accepts up to 5.
- Gated by RAG_ENABLE_PROPOSALS (default off).
