# Embedding bake-off

| Model | hit@5 | hit@10 | MRR@10 |
|---|---|---|---|
| all-MiniLM-L6-v2 | 94% | 94% | 0.823 |
| BAAI/bge-small-en-v1.5 | 94% | 100% | 0.944 |
| BAAI/bge-base-en-v1.5 | 94% | 94% | 0.938 |

## Misses (no hit in top 10)

- **all-MiniLM-L6-v2**: groundwater_investigation_trigger
- **BAAI/bge-small-en-v1.5**: none
- **BAAI/bge-base-en-v1.5**: groundwater_investigation_trigger
