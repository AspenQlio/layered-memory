# layered-memory

**Layered memory for AI agents: capture raw evidence, distill reusable insights,
and measure retrieval quality.**

An agent that stores every interaction in one flat history eventually loses the
difference between what happened and what it learned. `layered-memory` makes
that distinction explicit, preserves lineage between layers, and provides a
reproducible retrieval benchmark.

> [Versión en español](README.es.md)

## The model

```text
   literal capture          reviewed insight          final artifact
   (raw)                    (insight)                  (artifact)
      │  distill                 │  produce                 ▲
      └──────────────────────────┴──────────────────────────┘
                           parent_id
```

| Layer | Purpose | Indexed | Retrieved by default |
|---|---|---|---|
| `raw` | Literal observation or conversation | Yes | Yes |
| `insight` | Reviewed, reusable lesson | Yes | Yes |
| `artifact` | Runbook, report, or synthesized answer | Optional | No |

Promotion creates a new row linked to its source. It never overwrites the raw
evidence. `GET /memories/{id}/lineage` returns the complete chain.

## Measured baseline

The repository includes 27 queries against a 26-document fictional corpus.
Cases are grouped by intent so a single average cannot hide the hard cases.

| Group | What it measures | hit@1 | hit@3 | hit@10 | MRR |
|---|---|---:|---:|---:|---:|
| `lexical` (12) | Query and answer share vocabulary | **1.00** | 1.00 | 1.00 | 1.00 |
| `paraphrase` (9) | Same need, different words | **0.22** | 0.33 | 0.67 | 0.33 |
| `promotion` (6) | The useful answer is the distilled insight | **0.17** | 0.33 | 0.67 | 0.26 |

The default backend is deterministic and network-free. Its hit@1 drop from
1.00 to 0.22 exposes the exact limitation of lexical hashing: it does not
understand paraphrases. The benchmark is a baseline, not a claim of semantic
quality.

The same 27 cases were run against BGE-M3 through Ollama. Overall hit@1 rose
from 0.56 to 0.89, and paraphrase hit@1 rose from 0.22 to 0.78. The complete
report records the model digest. A hybrid backend keeps the semantic top five
and applies weighted RRF to the remaining results. It keeps hit@1 at 0.89 and
raises hit@10 from 0.96 to 1.00.

See [`docs/evaluation.md`](docs/evaluation.md) for the complete metrics,
limitations, and semantic-embedder instructions.

## Quick start

```bash
git clone https://github.com/AspenQlio/layered-memory.git
cd layered-memory

uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate

layered-memory --database-url sqlite:///memory.db capture \
  "The checkout gateway certificate expired without an alert." \
  --title "Certificate incident" --tag incident --tag payments

layered-memory --database-url sqlite:///memory.db stats
layered-memory --database-url sqlite:///memory.db search "expired certificate"
layered-memory serve --port 8000  # Open http://127.0.0.1:8000/docs
```

Run the complete flow in memory:

```bash
python scripts/demo.py
```

Run the deterministic retrieval benchmark:

```bash
layered-memory-eval \
  --corpus docs/demo-corpus.json \
  --dataset docs/eval-dataset.json \
  --markdown docs/eval-table.md
```

## API example

```bash
# Capture raw evidence
curl -sX POST localhost:8000/memories/capture \
  -H 'content-type: application/json' \
  -d '{"content":"Carrier webhooks lost events during bursts.",
       "title":"Dropped webhooks","tags":["incident","logistics"]}'

# Distill a reviewed insight without changing the source
curl -sX POST localhost:8000/memories/<id>/distill \
  -H 'content-type: application/json' \
  -d '{"title":"Synchronous delivery loses events during outages",
       "content":"A 500 discarded the notification, while retries could duplicate it."}'

# Build a character-budgeted prompt context
curl -sX POST localhost:8000/memories/context \
  -H 'content-type: application/json' \
  -d '{"query":"delivery notifications do not arrive","k":3,"max_chars":1200}'

# Escalate a low-confidence decision to a human
curl -sX POST localhost:8000/handoffs \
  -H 'content-type: application/json' \
  -d '{"reason":"discount outside policy","session_id":"s-42"}'
```

## Python example

```python
from layered_memory.memory.service import MemoryService
from layered_memory.retrieval.embeddings import HashingEmbedder
from layered_memory.retrieval.index import VectorIndex
from layered_memory.store.db import (
    build_engine,
    create_session_factory,
    init_db,
    session_scope,
)

engine = build_engine("sqlite:///memory.db")
init_db(engine)
factory = create_session_factory(engine)

embedder = HashingEmbedder(dim=512)
service = MemoryService(embedder)
index = VectorIndex(embedder)

with session_scope(factory) as session:
    raw = service.capture(session, "literal evidence", tags=["note"])
    service.distill(session, raw.id, title="lesson", content="reviewed insight")
    for hit in index.search(session, "what did I learn?", k=3):
        print(f"{hit.score:.3f}  {hit.record.title}")
```

## Main routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | Active embedding backend, dimension, and database |
| `POST` | `/memories/capture` | Store literal content in `raw` |
| `POST` | `/memories/{id}/distill` | Create an `insight` child |
| `POST` | `/memories/{id}/produce` | Create an `artifact` child |
| `GET` | `/memories/{id}/lineage` | Return the chain from its raw source |
| `GET` | `/memories/stats` | Counts, coverage, and pending distillation |
| `POST` | `/memories/search` | Similarity search with filters |
| `POST` | `/memories/context` | Search constrained by a character budget |
| `POST` | `/memories/reindex` | Re-embed after changing the backend |
| `POST` | `/handoffs` | Queue a human escalation |
| `POST` | `/handoffs/{id}/claim` | Atomically claim queued work |
| `POST` | `/handoffs/{id}/resolve` | Close an escalation with its resolution |

## Design decisions

- **Promotion never rewrites its source.** Incorrect distillation creates a new
  child; the original evidence remains available.
- **Deletion unlinks instead of cascading.** An insight survives its raw source
  with `parent_id = NULL`.
- **Handoff claims are atomic.** `UPDATE ... WHERE status='queued'` makes the
  affected-row count decide the winner. A second worker receives HTTP 409.
- **Prompt context uses a character budget.** Model context, not document count,
  is the actual constraint.
- **Embeddings use `LargeBinary`.** The same schema works on SQLite and
  PostgreSQL. The current NumPy search is intended for roughly 50k rows per
  namespace; the scale path is `pgvector` behind the same `VectorIndex` API.
- **Dimension mismatch fails loudly.** Changing embedders without reindexing
  raises an error instead of returning meaningless rankings.

The default `HashingEmbedder` keeps tests and CI offline. Any server implementing
OpenAI-compatible `POST /v1/embeddings` can provide semantic vectors:

```bash
export LAYERED_MEMORY_EMBEDDING_BACKEND=openai
export LAYERED_MEMORY_EMBEDDING_MODEL=bge-m3
export LAYERED_MEMORY_EMBEDDING_BASE_URL=http://localhost:11434/v1
layered-memory reindex
```

Set the backend to `hybrid` to keep the semantic top five and rerank the tail
with lexical hashing. The lexical signal includes titles, content, and tags.

```bash
export LAYERED_MEMORY_EMBEDDING_BACKEND=hybrid
export LAYERED_MEMORY_HYBRID_SEMANTIC_HEAD=5
export LAYERED_MEMORY_HYBRID_SEMANTIC_WEIGHT=3
layered-memory reindex
```

See [`docs/architecture.md`](docs/architecture.md) for the dependency and data
flow details.

## Development

```bash
ruff check .
pytest --cov=layered_memory --cov-report=term-missing  # 74 tests
```

The suite covers promotion lineage, retrieval filters and ranking, competing
handoff claims, HTTP error translation, and evaluation metrics.

## License

[MIT](LICENSE)
