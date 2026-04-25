# Context Bot — Architecture Spec

> This doc defines the classification, handoff, and isolation rules for the Context Bot.
> It is the single source of truth for any agent (Claude Code, Codex, or human) implementing or modifying the Context Bot or its integration with the RAG AI.

---

## 1. Role

The Context Bot is a classifier and router. It sits between the user and the RAG AI. It has read access to the active project state (selected site, criteria set, depth range, soil type, analyte results, borehole data). It does NOT have access to the regulatory knowledge base — that belongs exclusively to the RAG AI.

---

## 2. Classification Logic

Every user question is classified into exactly one of three routes.

### Route A — Project-only

**Trigger:** The question can be fully answered using project data alone. No regulatory or guideline information is needed.

**Examples:**
- "What's the benzene concentration at BH01?"
- "Which boreholes exceeded the selected criteria?"
- "Show me the results for the 0.5–1.0m samples"

**Behaviour:** Context Bot answers directly. RAG AI is not invoked.

### Route B — Hybrid (project + regulatory)

**Trigger:** The question references or implies project context AND requires regulatory/guideline information from the knowledge base.

**Examples:**
- "What is the HSL value for benzene?" (user has a project loaded with HSL-A selected, 0–1m, silt)
- "Are my TPH results above the criteria?"
- "What's the screening level for lead in my soil type?"

**Behaviour:** Context Bot resolves the project-specific values, builds a handoff packet (see §3), and passes it to the RAG AI. The RAG AI queries the KB independently and assembles the final response.

### Route C — Regulatory-only

**Trigger:** The question asks about regulatory frameworks, guidelines, or criteria values with no reference to a specific project. OR the user has no active project loaded.

**Examples:**
- "What are the NEPM HSL values for benzene?"
- "Explain the difference between HSL-A and HSL-D"
- "What does NEPM say about vapour intrusion assessment?"

**Behaviour:** The original user question is passed directly to the RAG AI. Context Bot does not contribute to the response.

### Edge Cases

| Scenario | Route | Rationale |
|---|---|---|
| User asks a hybrid question but no project is loaded | C (Regulatory-only) | No project context to resolve — degrade gracefully, answer from KB only. Optionally note that results could be refined if a project were loaded. |
| User asks about an analyte not present in their project data | B (Hybrid) | Context Bot reports "no data for [analyte] in current project" as the relay. RAG AI still answers the regulatory question. |
| User asks a question unrelated to contaminated land | Reject / general response | Context Bot responds that the question is outside scope. RAG AI is not invoked. |
| Ambiguous — could be project-only or hybrid | B (Hybrid) | Default to hybrid. Over-fetching from KB is better than under-answering. |

---

## 3. Handoff Packet Schema

When the Context Bot classifies a question as **Route B (Hybrid)**, it constructs the following handoff packet and passes it to the RAG AI.

```typescript
interface HandoffPacket {
  /**
   * The user's original question, unmodified.
   * This is the ONLY field the RAG AI uses for vector search / KB retrieval.
   */
  original_question: string;

  /**
   * A rewritten query optimised for KB retrieval.
   * Strips project-specific jargon. Focuses on the regulatory construct.
   * Example: "HSL values for benzene in NEPM 2013 Table 1A"
   *
   * If null, RAG AI uses original_question for retrieval instead.
   */
  kb_query: string | null;

  /**
   * Pre-written text containing the resolved project-specific answer.
   * This is OPAQUE to the RAG AI — it must not be parsed, reasoned about,
   * or used as input to any retrieval or embedding operation.
   * It is prepended verbatim to the final response under the
   * "Site context" section.
   */
  relay_block: string;

  /**
   * Metadata for logging / debugging. Not used in response generation.
   */
  classification: {
    route: "hybrid";
    project_id: string;
    criteria_set: string;      // e.g. "HSL-A"
    depth_range: string;       // e.g. "0-1m"
    soil_type: string;         // e.g. "silt"
    resolved_analytes: string[]; // e.g. ["benzene"]
  };
}
```

### Example Handoff

User question: *"What is the HSL value for benzene?"*
Active project: HSL-A selected, 0–1m depth, silt soil type.

```json
{
  "original_question": "What is the HSL value for benzene?",
  "kb_query": "NEPM 2013 HSL values for benzene, include all land use scenarios and soil types from exceedance tables",
  "relay_block": "For your selected criteria (HSL-A, 0–1m, silt), the benzene screening level is 0.5 mg/kg.",
  "classification": {
    "route": "hybrid",
    "project_id": "proj_abc123",
    "criteria_set": "HSL-A",
    "depth_range": "0-1m",
    "soil_type": "silt",
    "resolved_analytes": ["benzene"]
  }
}
```

---

## 4. Isolation Contract

These rules are non-negotiable. They exist to prevent the Context Bot's resolved values from contaminating the RAG AI's independent KB retrieval.

### Rule 1 — Relay block is opaque

The `relay_block` string MUST NOT be:
- Included in any vector search query
- Used as input to any embedding operation
- Parsed or reasoned about by the RAG AI
- Used to validate, cross-check, or filter KB results

The RAG AI treats `relay_block` as a sealed envelope. It prepends the contents to the final response. That is all.

### Rule 2 — KB query is the only retrieval input

The RAG AI's vector search and chunk retrieval MUST use only:
- `kb_query` (preferred, if provided), OR
- `original_question` (fallback)

No other field from the handoff packet may influence retrieval.

### Rule 3 — Response assembly order

The final response to the user is assembled in this fixed order:

1. **Site context** — the `relay_block` content, verbatim. Presented under a clear heading (e.g. "Your project criteria").
2. **Additional regulatory context** — the RAG AI's independent answer sourced entirely from the KB. Presented under a separate heading (e.g. "Regulatory information from NEPM 2013"). This section prioritises numeric values from exceedance tables where applicable.

These two sections must be visually and structurally separated in the response. The RAG AI must not blend, merge, or interleave content from the relay block with KB-sourced content.

### Rule 4 — RAG system prompt enforcement

The RAG AI's system prompt must include an explicit instruction along these lines:

```text
You will receive a structured handoff with two inputs:

RELAY_BLOCK: Pre-written project-specific context. Prepend this verbatim to your
response under a "Site context" heading. Do NOT use this text for retrieval.
Do NOT reason about the values in it. Do NOT let it influence your KB search
or your regulatory answer in any way.

KB_QUERY: Use this — and ONLY this — as your search query against the knowledge base.
Prioritise numeric values from exceedance tables. Provide complete table data where
relevant rather than summarising.

Your regulatory answer must be independently derived from the knowledge base as if
the relay block did not exist.
```

---

## 5. Context Bot Decision Pseudocode

```text
function classify(question: string, project: Project | null): Route {
  // No project loaded — can only do regulatory
  if (!project) return Route.REGULATORY_ONLY;

  // Check if question is about contaminated land / environmental
  if (!isEnvironmentalQuery(question)) return Route.REJECT;

  const refsProjectData = referencesProjectSpecificData(question, project);
  const needsRegulatory = requiresRegulatoryKnowledge(question);

  if (refsProjectData && !needsRegulatory) return Route.PROJECT_ONLY;
  if (refsProjectData && needsRegulatory)  return Route.HYBRID;
  if (!refsProjectData && needsRegulatory)  return Route.REGULATORY_ONLY;

  // Ambiguous — default to hybrid if project is loaded
  return Route.HYBRID;
}
```

---

## 6. Testing Scenarios

Use these to validate the implementation:

| # | Question | Project state | Expected route | Expected relay_block (summary) | Expected KB query focus |
|---|---|---|---|---|---|
| 1 | "What's the benzene value at BH01?" | Loaded, BH01 has benzene data | A (Project-only) | N/A | N/A |
| 2 | "What is the HSL for benzene?" | Loaded, HSL-A / 0–1m / silt | B (Hybrid) | "HSL-A, 0–1m, silt = 0.5 mg/kg" | "NEPM HSL values benzene all scenarios" |
| 3 | "What are the NEPM HSL values for benzene?" | Loaded (any) | C (Regulatory-only) | N/A | Original question |
| 4 | "What is the HSL for benzene?" | No project loaded | C (Regulatory-only) | N/A | Original question |
| 5 | "Are my lead results above criteria?" | Loaded, no lead data | B (Hybrid) | "No lead data in current project" | "NEPM screening levels for lead" |
| 6 | "What's the weather today?" | Any | Reject | N/A | N/A |
| 7 | "Explain HSL-A vs HSL-D" | Loaded (any) | C (Regulatory-only) | N/A | Original question |
| 8 | "Do my results exceed the HIL-A for arsenic?" | Loaded, arsenic data present | B (Hybrid) | "Arsenic at [boreholes]: [values]. HIL-A = [resolved value]" | "NEPM HIL-A values arsenic" |

---

## 7. `/query` API Contract

The backend owns routing, context filtering, KB query planning, and final response sectioning.
The frontend must send current workspace state but must not decide regulatory logic.

### Request

```typescript
interface QueryRequest {
  question: string;
  mode?: "hybrid" | "local" | "global" | "naive";
  session_id?: string | null;
  context?: WorkspaceContext | null;
}
```

`context` is optional for regulatory-only use. When present, it should include the current project state, selected criteria, criteria details, exceedances, project result rows, retrieval context, and recent conversation state where available.

### Response

```typescript
interface QueryResponse {
  answer: string;
  mode: string;
  session_id: string;
  session_reset: boolean;
  context_used: boolean;
  route_used: "project_only" | "hybrid" | "regulatory_only";
  grounded: boolean;
  citations: Citation[];
  sections: {
    site_context: string | null;
    regulatory_context: string | null;
    application: string | null;
  };
  debug: {
    effective_question: string | null;
    kb_query: string | null;
    route_reason: string | null;
    used_project_fields: string[];
    retrieval_mode: string | null;
    citation_count: number;
    citation_sources: string[];
  };
}
```

### Response Semantics

- `answer` is the backwards-compatible display string.
- `sections.site_context` is project/job context resolved from the supplied workspace only.
- `sections.regulatory_context` is the independent KB answer.
- `sections.application` is reserved for a future controlled synthesis layer.
- `debug.kb_query` is the exact query sent to the knowledge base.
- `debug.used_project_fields` lists the project context fields used to build the site context.
- `debug.retrieval_mode`, `debug.citation_count`, and `debug.citation_sources` expose retrieval observability for QA and frontend diagnostics.
- `citations` apply to `sections.regulatory_context`, not to project context.

For Route B (Hybrid), the backend assembles `answer` in this order:

```text
Site context
[sections.site_context]

Regulatory context
[sections.regulatory_context]
```

For Route A (Project-only), `sections.site_context` contains the project-only answer and `sections.regulatory_context` is `null`.

For Route C (Regulatory-only), `sections.regulatory_context` contains the KB answer and `sections.site_context` is `null`.
