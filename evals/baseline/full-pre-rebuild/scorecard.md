# Golden eval scorecard

**Overall (blocking): 22/25 passed (88%)**

| Category | Passed | Total |
|---|---|---|
| criteria_explanation | 3 | 4 |
| follow_up | 2 | 2 |
| injection | 4 | 4 |
| jurisdiction | 3 | 3 |
| project_evidence | 6 | 6 |
| source_pathway | 2 | 2 |
| threshold_lookup | 2 | 4 |
| _info (non-blocking)_ | 2 | 5 |

## Failures

### worst_depth _(info)_
- must_include:0-0.5: answer must contain '0-0.5'

### threshold_kb_lead_hil_b
- citation_source_pattern: no citation source/title matches '(?i)nepm' (sources: ['F2013C00288VOL08.pdf', 'F2013C00288VOL18.pdf', 'F2013C00288VOL02.pdf', 'F2013C00288VOL18.pdf'])

### threshold_kb_copper_anzecc
- citation_source_pattern: no citation source/title matches '(?i)anzecc' (sources: ['F2013C00288VOL08.pdf', 'F2013C00288VOL07.pdf', 'F2013C00288VOL06.pdf', 'pfas-nemp-3.pdf'])

### groundwater_trigger
- citation_source_pattern: no citation source/title matches '(?i)nepm' (sources: ['F2013C00288VOL03.pdf', 'F2013C00288VOL02.pdf', 'pfas-nemp-3.pdf', 'F2013C00288VOL05.pdf'])

### table_lookup_hil_metals _(info)_
- exact_locator: citations without a table/page locator: ['chunk-468a8e7c86285064ec9e7bba45afad3d']

### table_lookup_esl_benzene _(info)_
- exact_locator: citations without a table/page locator: ['chunk-fd665233d7367a3b2979cf5020416c16', 'chunk-75a5131bf1790c33fe2beb2a101afd05']
