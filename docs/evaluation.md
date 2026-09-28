# Evaluacion

La pregunta que responde este documento es una sola: **sirve este indice para
recuperar lo que un agente necesita?** Sin medicion, cualquier ajuste de
embeddings, de chunking o de filtros es un cambio de sabor.

## Como se corre

```bash
layered-memory-eval \
  --corpus docs/demo-corpus.json \
  --dataset docs/eval-dataset.json \
  --json docs/eval-results.json \
  --markdown docs/eval-table.md
```

Es determinista: mismo corpus, mismo backend, mismos numeros. El corpus y el
set de casos estan versionados en JSON precisamente para que una regresion se
vea en el diff y no en la memoria del equipo.

## El corpus

26 documentos de una empresa de comercio ficticia, repartidos en las tres capas:
13 capturas, 11 destilados y 2 artefactos. Los documentos son deliberadamente
heterogeneos: incidentes, pagos, recursos humanos, calidad de datos,
inventario, licencias, seguridad, busqueda de catalogo, email transaccional,
operaciones, costes y accesibilidad.

## Los tres grupos de casos

Un promedio unico sobre 27 consultas no dice nada, porque mezcla dos problemas
distintos. Los casos estan agrupados:

| Grupo | Que mide | Casos |
|---|---|---|
| `lexical` | La consulta comparte vocabulario con el documento | 12 |
| `paraphrase` | La misma necesidad, con otras palabras | 9 |
| `promotion` | La respuesta util es el destilado, no la captura | 6 |

## Resultados medidos

Backend `hash` (`HashingEmbedder`, dim 512), sin red.

| grupo | casos | hit@1 | hit@3 | hit@5 | hit@10 | mrr | ndcg@5 |
|---|---|---|---|---|---|---|---|
| lexical | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| paraphrase | 9 | 0.22 | 0.33 | 0.33 | 0.67 | 0.33 | 0.29 |
| promotion | 6 | 0.17 | 0.33 | 0.33 | 0.67 | 0.26 | 0.15 |
| **promedio** | 27 | **0.56** | **0.63** | **0.63** | **0.81** | **0.61** | **0.58** |

## Que dicen estos numeros

**El indice es perfecto cuando las palabras coinciden y se rompe cuando no.**
`hit@1` cae de 1.00 a 0.22 entre `lexical` y `paraphrase`. Esa es la firma exacta
de una representacion de bolsa de palabras: el mismo tema con otra redaccion no
se parece nada. No es un bug del indice, es lo que el indice puede hacer.

**Ampliar `k` recupera recall a costa de precision.** De `k=1` a `k=10` el
`hit@10` sube a 0.67 en los grupos duros, mientras `precision@10` cae a 0.08:
diez documentos, cero o uno relevante. Por eso `build_context_block` recorta por
presupuesto de caracteres y no por numero de documentos.

**El grupo `promotion` es el mas dificil, y eso aporta informacion.** Hit@1 de 0.17
significa que cuando la pregunta es "¿que habria que cambiar?", el agente
recupera la captura literal antes que el destilado que ya respondio eso. El
destilado esta disponible; el ranking lo deja fuera. Destilar mejora la
respuesta solo si el retrieval lo prioriza.

**Cinco de 27 casos no aparecen en el top-10.** Son las fugas del indice, y son
la lista de trabajo: o se agrega un sinonimo, o se cambia el embedder, o la
consulta no tiene respuesta en el corpus. Los tres son decisiones distintas.

## Cambiar el embedder

El backend por defecto es determinista y local a proposito: los tests, la CI y
este demo tienen que correr sin red y sin GPU. Para medir un embedder semantico
se apunta a cualquier endpoint compatible con la especificacion de OpenAI:

```bash
# Ollama
ollama serve
ollama pull bge-m3

layered-memory-eval \
  --corpus docs/demo-corpus.json --dataset docs/eval-dataset.json \
  --backend openai --embedding-model bge-m3 \
  --embedding-base-url http://localhost:11434/v1
```

El resto de las metricas se recalcula igual. El numero que hay que mirar es la
columna `paraphrase`: si no sube de 0.22, el embedder semantico no esta
aportando sobre este corpus y el problema esta en otra parte.

En produccion, cambiar de embedder exige `layered-memory reindex`: la dimension
guardada deja de coincidir y `VectorIndex.search` lo rechaza a proposito en vez
de devolver ruido.

## Metricas implementadas

Todas en `eval/metrics.py`, con relevancia binaria y posicion 1 como la mejor:

- `hit@k` — al menos un relevante en el top-k
- `recall@k` — fraccion de relevantes en el top-k
- `precision@k` — fraccion de relevantes dentro del top-k, penaliza el relleno
- `mrr` — 1/rank del primer relevante
- `ndcg@k` — ganancia normalizada con descuento logaritmico

`EvalReport.aggregate_by_group()` es la razon de que el modulo exista: un promedio
unico sobre consultas de dos naturalezas distintas es un numero que no lleva a
ninguna decision.

## Limitaciones conocidas

- **El corpus es pequeno** (26 documentos). Los hit@k estan optimistas respecto de
  un corpus real, donde la competencia por el ranking es mayor.
- **La anotacion es de un solo documento.** Un caso como `q-pro-03` marca solo el
  destilado, pero la captura literal tambien es aceptable; el ranking la pone
  primero y el caso cuenta como fallo. Con relevancia graduada o multiples
  aceptables, ese grupo mejoraria.
- **No hay evaluacion de la calidad de la respuesta.** Aqui se mide que se recupera
  el documento correcto, no que el agente lo use bien. Eso exigiria un juez con
  modelo y su propio conjunto de metricas.
- **El backend por omision no es semantico.** Todo lo anterior es valido como
  linea base, no como techo.
