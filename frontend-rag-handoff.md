# Chempair RAG Frontend Handoff Packet

## Purpose

This packet defines how the Chempair frontend should integrate with the RAG backend after the Phase 1 and Phase 2 backend contract fixes.

The backend owns:
- routing between project-only, hybrid, and regulatory-only answers
- filtering project context
- building the knowledge-base query
- retrieving regulatory context
- returning structured answer sections and citations

The frontend owns:
- sending the active workspace/project state accurately
- rendering structured response sections
- showing citations for regulatory content
- optionally exposing debug metadata in developer/admin views

The frontend must not implement regulatory routing or decide which criteria/regulations apply.

---

## Endpoint

`POST /query`

### Request Schema

```ts
type QueryMode = "hybrid" | "local" | "global" | "naive";

interface QueryRequest {
  question: string;
  mode?: QueryMode;
  session_id?: string | null;
  context?: WorkspaceContext | null;
}
```

Use `mode: "hybrid"` unless there is a specific backend reason to use another LightRAG mode.

Send `session_id` from the previous response to support follow-up questions.

Send `context` whenever a project is loaded. Send `context: null` or omit `context` when no project is loaded.

---

## Workspace Context

The frontend should send the active project state in this shape.

```ts
interface WorkspaceContext {
  schemaVersion?: number;
  generatedAtIso?: string;
  projectState?: ProjectState;
  retrievalContext?: RetrievalContext;
  conversation?: ConversationMessage[];
}

interface ProjectState {
  project?: ProjectInfo;
  selectedCriteria?: SelectedCriteria;
  criteriaDetails?: CriteriaDetail[];
  exceedanceSummary?: ExceedanceSummary;
  exceedances?: Exceedance[];
  projectResults?: ProjectResultRow[];
  fieldSummary?: FieldSummary;
}

interface ProjectInfo {
  projectName?: string;
  projectId?: string;
  siteName?: string;
  address?: string;
  labReportNumber?: string;
  projectType?: string;
  sourceFile?: string;
  totalSamples?: number;
  totalAnalytes?: number;
}

interface SelectedCriteria {
  applicableCriteria?: string;
  regulations?: string[];
  landUse?: string;
  state?: string;
  criteriaNames?: string[];
  criteriaCount?: number;
}

interface CriteriaDetail {
  name?: string;
  thresholds?: CriterionThreshold[];
}

interface CriterionThreshold {
  analyte?: string;
  value?: number | string;
  unit?: string;
}

interface ExceedanceSummary {
  totalExceedances?: number;
  affectedSamples?: string[];
  affectedAnalytes?: string[];
  exceededCriteria?: string[];
  hotspotCount?: number;
}

interface Exceedance {
  analyte?: string;
  sampleCode?: string;
  criterion?: string;
  value?: number | string;
  criterionValue?: number | string;
  exceedanceFactor?: number;
  isHotspot?: boolean;
  unit?: string;
  date?: string;
}

interface ProjectResultRow {
  sampleCode?: string;
  depth?: string;
  collectionDate?: string;
  sampleType?: string;
  sampleRound?: string;
  labName?: string;
  labReportNumber?: string;
  coordinates?: { lat?: number; lng?: number };
  analyteValues?: AnalyteValue[];
}

interface AnalyteValue {
  analyte?: string;
  value?: number | string;
  unit?: string;
}

interface FieldSummary {
  hasFieldData?: boolean;
  sessionCount?: number;
  boreholeCount?: number;
  fieldSampleCount?: number;
  lithologyLogCount?: number;
  latestSessionDate?: string;
  sampleTypes?: string[];
  depthRange?: string;
  hasGpsData?: boolean;
}

interface RetrievalContext {
  matchedAnalytes?: string[];
  matchedSampleCodes?: string[];
  questionTokens?: string[];
  retrievedRows?: ProjectResultRow[];
}

interface ConversationMessage {
  role?: string;
  content?: string;
}
```

### Required When Available

For good answers, send these fields whenever the frontend has them:
- `projectState.project.projectName`
- `projectState.project.siteName`
- `projectState.project.address`
- `projectState.project.labReportNumber`
- `projectState.selectedCriteria.applicableCriteria`
- `projectState.selectedCriteria.regulations`
- `projectState.selectedCriteria.landUse`
- `projectState.selectedCriteria.state`
- `projectState.criteriaDetails[].thresholds[]`
- `projectState.exceedances[]`
- `projectState.projectResults[]`
- `retrievalContext.matchedAnalytes`
- `retrievalContext.matchedSampleCodes`
- `retrievalContext.retrievedRows`

The backend caps oversized context payloads. Keep `projectResults` relevant where possible, but do not strip criteria, exceedances, or selected project metadata.

---

## Response Schema

```ts
interface QueryResponse {
  answer: string;
  mode: string;
  session_id: string;
  session_reset: boolean;
  context_used: boolean;
  route_used: "project_only" | "hybrid" | "regulatory_only";
  grounded: boolean;
  citations: Citation[];
  sections: ResponseSections;
  debug: DebugMetadata;
}

interface Citation {
  source: string;
  title: string;
  locator: string;
  snippet: string;
}

interface ResponseSections {
  site_context: string | null;
  regulatory_context: string | null;
  application: string | null;
}

interface DebugMetadata {
  effective_question: string | null;
  kb_query: string | null;
  route_reason: string | null;
  used_project_fields: string[];
  retrieval_mode: string | null;
  citation_count: number;
  citation_sources: string[];
}
```

---

## Route Semantics

### `project_only`

The backend answered using only supplied project context.

Examples:
- “What criteria have I selected?”
- “What are the main exceedances?”
- “What is the lead result at BH-01?”

Render:
- show `sections.site_context`
- do not show citation UI unless `citations` is non-empty
- `sections.regulatory_context` should normally be `null`

### `hybrid`

The backend used project context and an independent KB query.

Examples:
- “What is the HSL for benzene?”
- “What does the NEPM say about my TRH exceedance?”
- “Is this result above the selected criterion?”

Render:
- show `sections.site_context` under a heading such as `Site context`
- show `sections.regulatory_context` under a heading such as `Regulatory context`
- show citations under the regulatory section only
- do not merge citations into the site context section

### `regulatory_only`

The backend answered from the KB only.

Examples:
- “What does the NEPM say about asbestos?”
- “Explain HSL-A vs HSL-D”
- “What are the soil sampling guidelines?”

Render:
- show `sections.regulatory_context`
- show citations if present
- do not show an empty site context panel

---

## Rendering Rules

Prefer structured sections over parsing `answer`.

Recommended display logic:

```ts
function renderResponse(response: QueryResponse) {
  if (response.sections.site_context) {
    renderSection("Site context", response.sections.site_context);
  }

  if (response.sections.regulatory_context) {
    renderSection("Regulatory context", response.sections.regulatory_context);
  }

  if (response.sections.application) {
    renderSection("Application", response.sections.application);
  }

  if (response.citations.length > 0) {
    renderCitations(response.citations);
  }
}
```

`answer` remains backwards-compatible, but new UI should prefer `sections`.

Citation display:
- `title`: human-readable source title
- `locator`: page/table/chunk locator
- `snippet`: short evidence extract
- `source`: source file name

Example citation label:

```text
NEPM 2013, Table 1B(7), p. 123
```

---

## Debug Display

In development/admin builds, expose:
- `route_used`
- `debug.route_reason`
- `debug.effective_question`
- `debug.kb_query`
- `debug.used_project_fields`
- `debug.retrieval_mode`
- `debug.citation_count`
- `debug.citation_sources`

This is useful for diagnosing:
- missing project context
- wrong analyte matching
- wrong route
- bad KB query
- missing citations

Do not expose debug metadata in ordinary client-facing report output.

---

## Sample Requests And Responses

### 1. Project-Only

Request:

```json
{
  "question": "What criteria have I selected?",
  "mode": "hybrid",
  "context": {
    "schemaVersion": 3,
    "generatedAtIso": "2026-04-26T10:00:00.000Z",
    "projectState": {
      "project": {
        "projectName": "Project One",
        "siteName": "Ducat",
        "labReportNumber": "LAB-001"
      },
      "selectedCriteria": {
        "applicableCriteria": "NEPM 2013 HIL-A",
        "regulations": ["NEPM 2013"],
        "landUse": "Residential",
        "state": "QLD",
        "criteriaNames": ["NEPM 2013 HIL-A"]
      }
    }
  }
}
```

Response shape:

```json
{
  "answer": "The selected criteria are NEPM 2013 HIL-A.",
  "route_used": "project_only",
  "context_used": true,
  "grounded": false,
  "citations": [],
  "sections": {
    "site_context": "The selected criteria are NEPM 2013 HIL-A.",
    "regulatory_context": null,
    "application": null
  },
  "debug": {
    "effective_question": "What criteria have I selected?",
    "kb_query": "What criteria have I selected?",
    "route_reason": "deterministic_project_fact",
    "used_project_fields": [],
    "retrieval_mode": "hybrid",
    "citation_count": 0,
    "citation_sources": []
  }
}
```

### 2. Hybrid

Request:

```json
{
  "question": "What is the HSL for benzene?",
  "mode": "hybrid",
  "context": {
    "schemaVersion": 3,
    "generatedAtIso": "2026-04-26T10:00:00.000Z",
    "projectState": {
      "project": {
        "projectName": "Project Benzene"
      },
      "selectedCriteria": {
        "applicableCriteria": "EPM 2013 HSL-A Low Density Residential Sand (0m to <1m)",
        "regulations": ["NEPM 2013"],
        "landUse": "Low Density Residential",
        "state": "QLD",
        "criteriaNames": ["EPM 2013 HSL-A Low Density Residential Sand (0m to <1m)"]
      },
      "criteriaDetails": [
        {
          "name": "EPM 2013 HSL-A Low Density Residential Sand (0m to <1m)",
          "thresholds": [
            { "analyte": "Benzene", "value": 0.5, "unit": "mg/kg" }
          ]
        }
      ]
    },
    "retrievalContext": {
      "matchedAnalytes": ["Benzene"],
      "questionTokens": ["benzene", "hsl-a"]
    }
  }
}
```

Response shape:

```json
{
  "route_used": "hybrid",
  "context_used": true,
  "grounded": true,
  "sections": {
    "site_context": "Project: Project Benzene.\nSelected criteria: EPM 2013 HSL-A Low Density Residential Sand (0m to <1m); land use Low Density Residential; state QLD; regulations NEPM 2013.\nApplied project criterion: Benzene = 0.5 mg/kg under EPM 2013 HSL-A Low Density Residential Sand (0m to <1m).",
    "regulatory_context": "[backend KB answer]",
    "application": null
  },
  "debug": {
    "kb_query": "NEPM 2013 HSL criteria values for Benzene; selected criterion: EPM 2013 HSL-A Low Density Residential Sand (0m to <1m); include table values and units",
    "used_project_fields": [
      "project.projectName",
      "selectedCriteria.applicableCriteria",
      "selectedCriteria.landUse",
      "selectedCriteria.state",
      "selectedCriteria.regulations",
      "criteria.criteriaDetails.thresholds"
    ],
    "retrieval_mode": "hybrid",
    "citation_count": 1,
    "citation_sources": ["NEPM_2013.pdf"]
  },
  "citations": [
    {
      "source": "NEPM_2013.pdf",
      "title": "NEPM 2013",
      "locator": "Table 1B(7), p. 123",
      "snippet": "Short evidence extract..."
    }
  ]
}
```

### 3. Regulatory-Only

Request:

```json
{
  "question": "Explain HSL-A vs HSL-D",
  "mode": "hybrid",
  "context": null
}
```

Response shape:

```json
{
  "route_used": "regulatory_only",
  "context_used": false,
  "grounded": true,
  "sections": {
    "site_context": null,
    "regulatory_context": "[backend KB answer]",
    "application": null
  },
  "citations": [
    {
      "source": "NEPM_2013.pdf",
      "title": "NEPM 2013",
      "locator": "p. 45",
      "snippet": "Short evidence extract..."
    }
  ]
}
```

---

## Frontend Acceptance Checklist

- [ ] Sends `context` whenever a project is loaded.
- [ ] Sends selected criteria, criteria details, exceedances, and project results where available.
- [ ] Persists and resends `session_id` for follow-up questions.
- [ ] Renders `sections.site_context` and `sections.regulatory_context` separately.
- [ ] Shows citations only for regulatory/KB content.
- [ ] Does not infer route or regulatory scope client-side.
- [ ] Provides a developer/admin debug view for `debug.kb_query` and `debug.route_reason`.
- [ ] Handles `session_reset: true` by storing the new response but starting the next question without the old session.
- [ ] Handles `context_used: false` by not displaying a site-context panel.
- [ ] Handles empty `citations` without implying the response is uncited project data.

---

## Notes For Frontend Engineer

The backend has already been updated so hybrid responses are sectioned and the KB query is rewritten before retrieval.

The most important frontend fix is to stop treating the response as one unstructured blob. Render the `sections` object directly.

The second most important fix is to send enough project context. If `criteriaDetails`, `projectResults`, or `exceedances` are missing, the backend can still answer regulatory questions, but it cannot reliably explain the current job conditions.
