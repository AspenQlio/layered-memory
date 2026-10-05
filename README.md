# Layered Memory

> Capture raw evidence, distill reusable insights, and measure retrieval quality for AI agents.

An agent that stores every interaction in one flat history eventually loses the difference between what happened and what it learned. **Layered Memory** makes that distinction explicit, preserves lineage between layers, and provides a reproducible retrieval benchmark.

> [Versión en español](README.es.md)

## Features

- **Lineage Preservation:** Promotion creates a new row linked to its source without overwriting raw evidence.
- **Three-Tier Architecture:** Explicit distinction between `raw` evidence, distilled `insight`, and synthesized `artifact`.
- **Reproducible Benchmarks:** Built-in evaluation dataset with 27 queries against a fictional corpus to measure hit@k and MRR.
- **Hybrid Retrieval Backend:** Combines semantic search (e.g., via Ollama) and lexical hashing with weighted RRF.
- **Atomic Handoffs:** Safe, conflict-free queueing system for human escalation.

## The Model

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

## Tech Stack

- **Language:** Python
- **Database:** SQLite (Default) / PostgreSQL ready (`LargeBinary` for vectors)
- **API Framework:** FastAPI
- **Package Management:** uv
- **Testing & Linting:** pytest, ruff

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) installed.

### Installation

```bash
git clone https://github.com/AspenQlio/layered-memory.git
cd layered-memory

uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate
```

## Usage

### Quick Start CLI

```bash
# Capture an observation
layered-memory --database-url sqlite:///memory.db capture \
  "The checkout gateway certificate expired without an alert." \
  --title "Certificate incident" --tag incident --tag payments

# Retrieve stats and search
layered-memory --database-url sqlite:///memory.db stats
layered-memory --database-url sqlite:///memory.db search "expired certificate"

# Start the API server
layered-memory serve --port 8000  # Open http://127.0.0.1:8000/docs
```

Run the complete flow in memory:
```bash
python scripts/demo.py
```

### API Example

```bash
# Capture raw evidence
curl -sX POST localhost:8000/memories/capture \
  -H 'content-type: application/json' \
  -d '{"content":"Carrier webhooks lost events.", "title":"Dropped webhooks","tags":["incident"]}'

# Distill a reviewed insight
curl -sX POST localhost:8000/memories/1/distill \
  -H 'content-type: application/json' \
  -d '{"title":"Sync delivery loses events", "content":"A 500 discarded the notification."}'

# Build a character-budgeted prompt context
curl -sX POST localhost:8000/memories/context \
  -H 'content-type: application/json' \
  -d '{"query":"delivery notifications do not arrive","k":3,"max_chars":1200}'
```

*(See the full README or `docs/` for Python Client usage and the complete API route table).*

## Semantic Backend & Evaluation

The default backend is deterministic and network-free. It uses lexical hashing, meaning it does not understand paraphrases natively. You can connect it to an OpenAI-compatible embedder (like Ollama) for high-quality semantic retrieval:

```bash
export LAYERED_MEMORY_EMBEDDING_BACKEND=openai
export LAYERED_MEMORY_EMBEDDING_MODEL=bge-m3
export LAYERED_MEMORY_EMBEDDING_BASE_URL=http://localhost:11434/v1
layered-memory reindex
```

Run the deterministic retrieval benchmark to see metrics (hit@1, hit@3, MRR):
```bash
layered-memory-eval \
  --corpus docs/demo-corpus.json \
  --dataset docs/eval-dataset.json \
  --markdown docs/eval-table.md
```

## Design Decisions

- **Promotion never rewrites its source:** The original evidence remains available.
- **Deletion unlinks instead of cascading:** An insight survives its raw source (`parent_id = NULL`).
- **Handoff claims are atomic:** `UPDATE ... WHERE status='queued'` prevents race conditions.
- **Prompt context uses a character budget:** Model context, not document count, is the true constraint.
- **Dimension mismatch fails loudly:** Changing embedders without reindexing raises an error.

## Development

```bash
ruff check .
pytest --cov=layered_memory --cov-report=term-missing
```
The suite covers promotion lineage, retrieval filters, ranking, competing handoff claims, HTTP error translation, and evaluation metrics.

## License

This project is licensed under the MIT License.
