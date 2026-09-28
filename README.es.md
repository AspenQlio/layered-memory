# layered-memory

**Memoria por capas para agentes de IA: captura, destilación y retrieval con
evaluación reproducible.**

Un agente que acumula todo en un solo plano se ahoga en su propio historial y no
distingue lo que vivió de lo que concluyó. Este proyecto separa esos momentos
en tres capas explícitas, mide si el retrieval funciona y expone todo por HTTP.

> [English version](README.md)

---

## El problema

Dos fallas aparecen tarde y cuestan caro:

1. **El contexto crece sin criterio.** Cada turno se llena con más historial. La
   respuesta empeora y el costo sube.
2. **La observación y la conclusión pesan igual.** Una nota suelta y la lección
   que se extrajo de ella terminan compitiendo en el mismo ranking.

La respuesta es estructural: la memoria tiene capas y **subir de capa es un acto
explícito**, no un efecto secundario de escribir.

```
   captura literal          destilado revisable         producto final
   (raw)                    (insight)                    (artifact)
      │  distill                 │  produce                   ▲
      └──────────────────────────┴────────────────────────────┘
                           parent_id
```

| Capa | Qué es | Se indexa | Se busca |
|---|---|---|---|
| `raw` | Lo que se dijo o se observó, textual | Sí | Por omisión |
| `insight` | La lección destilada, reescrita | Sí | Por omisión |
| `artifact` | Runbook, informe, respuesta | Opcional | A pedido |

El destilado es una fila nueva que apunta a su origen, nunca una reescritura: el
texto literal siempre sigue ahí, y `GET /memories/{id}/lineage` devuelve la
cadena completa.

---

## Resultados de la evaluación

27 consultas contra un corpus de 26 documentos, agrupadas por naturaleza:

| Grupo | Qué mide | hit@1 | hit@3 | hit@10 | MRR |
|---|---|---|---|---|---|
| `lexical` (12) | La consulta comparte vocabulario | **1.00** | 1.00 | 1.00 | 1.00 |
| `paraphrase` (9) | La misma necesidad, otras palabras | **0.22** | 0.33 | 0.67 | 0.33 |
| `promotion` (6) | La respuesta útil es el destilado | **0.17** | 0.33 | 0.67 | 0.26 |

El backend por defecto es una representación de bolsa de palabras determinista y
sin red. `hit@1` cae de 1.00 a 0.22 entre consultas literales y parafraseadas: es
la firma exacta de un índice que no entiende sinónimos, y por eso el número está
a la vista en vez de escondido detrás de un promedio.

Los mismos 27 casos se ejecutaron con BGE-M3 mediante Ollama. El `hit@1` global
subió de 0.56 a 0.89, y el grupo `paraphrase` subió de 0.22 a 0.78. El reporte
completo registra el digest del modelo y el único fallo restante.

Análisis completo, cambio de embedder y limitaciones conocidas en
[`docs/evaluation.md`](docs/evaluation.md).

---

## Arranque rápido

```bash
git clone https://github.com/AspenQlio/layered-memory.git
cd layered-memory
uv venv
uv pip install -e ".[dev]"
source .venv/bin/activate

layered-memory --database-url sqlite:///memoria.db capture \
  "El certificado TLS del gateway venció y nadie lo notó." \
  --title "Incidente de certificados" --tag incidente --tag pagos

layered-memory --database-url sqlite:///memoria.db search "certificado vencido"
layered-memory serve --port 8000             # http://127.0.0.1:8000/docs

layered-memory-eval \
  --corpus docs/demo-corpus.json \
  --dataset docs/eval-dataset.json
```

El demo end-to-end usa una base en memoria y no deja archivos locales:

```bash
python scripts/demo.py
```

---

## Decisiones de diseño

**Promover marca el origen y no lo reescribe.** El texto literal es evidencia.

**El borrado desvincula en vez de cascadear.** Un `insight` sobrevive a la captura
que lo originó con `parent_id = NULL`.

**`claim` es atómico.** `UPDATE ... WHERE status='queued'` y se mira el
`rowcount`: el que actualiza gana, el otro recibe `409`.

**El bloque de contexto recorta por caracteres, no por número de documentos.**

**El embedding va como `LargeBinary`, no `ARRAY(Float)`,** para que un solo
esquema funcione en SQLite y PostgreSQL sin dialectos. El camino de escala es
`pgvector`, sin cambiar la API de `VectorIndex.search`.

**Cambiar de embedder rompe la dimensión a propósito:** `search` verifica que
coincida y lanza un error en vez de devolver ruido.

Un embedder semántico es una variable de entorno:

```bash
LAYERED_MEMORY_EMBEDDING_BACKEND=openai
LAYERED_MEMORY_EMBEDDING_MODEL=bge-m3
LAYERED_MEMORY_EMBEDDING_BASE_URL=http://localhost:11434/v1
```

Detalle de arquitectura en [`docs/architecture.md`](docs/architecture.md); API
completa en `/docs` al levantar el servidor.

---

## Desarrollo

```bash
pytest          # 63 tests
ruff check .
```

MIT.
