# Arquitectura

## El problema

Un agente conversacional que solo acumula texto crudo tiene dos problemas que
se pagan tarde:

1. **Se ahoga en su propio historial.** Cuanto mas conversa, mas contexto, peor
   la respuesta y mas caro el turno.
2. **No distingue lo que vivio de lo que concluyo.** Una observacion aislada y la
   leccion que se extrajo de ella terminan con el mismo peso en la busqueda.

`layered-memory` resuelve los dos con una decision estructural: la memoria tiene
capas, y subir de capa es un acto explicito que alguien (o algo) tiene que
hacer.

## Las tres capas

```
   captura literal          destilado revisable        producto final
   (raw)                    (insight)                   (artifact)
      │                         │                           ▲
      │  distill                │                           │ produce
      └────────────────────────►┴───────────────────────────┘
                              parent_id
```

| Capa | Que es | Cuanto crece | Se indexa | Se busca |
|---|---|---|---|---|
| `raw` | Lo que se dijo o se observo, textual | Mucho, automatico | Si | Por omision |
| `insight` | La leccion destilada, reescrita | Poco, curado | Si | Por omision |
| `artifact` | Runbook, informe, respuesta sintetica | Muy poco | Opcional | A pedido |

Las reglas que sostienen el modelo:

- **Solo `raw` acepta contenido sin destilar.** Nada mas nace de otra cosa.
- **Promover marca el origen.** Una captura queda `distilled` y sale de la cola de
  pendientes.
- **Ninguna promocion reescribe el original.** Si el destilado se equivoca, se
  escribe otro hijo. El texto literal siempre sigue ahi.
- **El borrado desvincula, no cascada.** Si se elimina una captura, sus hijos
  sobreviven con `parent_id = NULL` en vez de desaparecer con ella.

## Flujo de escritura

```
capture()      -> INSERT en memories (layer=raw, status=pending) + vector
distill()      -> INSERT en memories (layer=insight, parent_id=raw) + vector
                   UPDATE memories SET status='distilled' WHERE id=raw
produce()      -> INSERT en memories (layer=artifact, parent_id=insight)
```

Cada paso escribe tambien una fila en `events`. Esa bitacora existe para dos
cosas concretas: depurar por que un documento no aparece en una busqueda, y
medir el lazo de aprendizaje (que capturas se destilan, que consultas quedan sin
cobertura).

## Flujo de lectura

```
consulta -> embedder -> vector normalizado
                        |
   SELECT ... WHERE namespace = ? AND layer IN (...) AND embedding IS NOT NULL
                        |
            filtro por tags (exhaustivo, en Python)
                        |
          coseno por numpy: matriz @ vector
                        |
             top-k por puntaje, con min_score
```

Dos decisiones que conviene explicar:

**El embedding va como `LargeBinary`, no como `ARRAY(Float)`.** Un unico esquema
funciona en SQLite y PostgreSQL sin dialectos. El costo es que la busqueda
vectorial se resuelve en numpy en vez de en el motor.

**El filtro por namespace y capa va en SQL; el de tags en Python.** SQL usa los
indices y acota el trabajo. Python recibe un subconjunto ya acotado. Cuando el
filtro de tags sea el dominante, esta division se revisa.

El limite actual es `~50k filas por namespace`: se descargan todos los vectores
del subconjunto. El camino de escala es mover la columna a `pgvector` y dejar
`ORDER BY embedding <=> :query LIMIT k`. La API de `VectorIndex.search` no
cambia; cambia su interior. Ese es el motivo de que la clase exista.

## Escalado a humano

```
        +-----------+   confianza baja    +---------------+
  agente |           | -----------------> | handoff.queued|
        +-----------+                    +---------------+
                                                | claim (atomico)
                                                v
                                        +-----------------+
                                        |  claimed        |
                                        +-----------------+
                                                | resolve
                                                v
                                        +-----------------+
                                        |  resolved       |
                                        +-----------------+
```

`claim` es un `UPDATE ... WHERE status='queued'` cuyo `rowcount` decide el
ganador. Dos consumidores compitiendo no pueden ganar los dos, y el perdedor
recibe `HandoffConflict` en vez de procesar la misma peticion dos veces.

El par `(reason, resolution)` de cada handoff resuelto es la semilla del lazo de
aprendizaje: es exactamente el ejemplo de "el agente no debio decidir esto solo".

## Representacion vectorial y persistencia

Los vectores se guardan como `float32` little-endian, L2-normalizados al
escribir. El producto punto entre dos vectores normalizados es el coseno, asi que
comparar es una multiplicacion de matrices y nada mas.

Un cambio de embedder cambia la dimension. `VectorIndex.search` verifica que la
dimension guardada coincida con la del embedder activo y lanza un error en vez
de devolver ruido; `layered-memory reindex` rehace los vectores.

## Estructura del codigo

```
src/layered_memory/
├── domain.py                 enumeraciones compartidas, sin dependencias
├── config.py                 ajustes por entorno o .env
├── store/
│   ├── models.py             tablas SQLAlchemy 2.0
│   └── db.py                 motor, sesiones, init
├── memory/
│   ├── record.py             forma de dominio, independiente del ORM
│   └── service.py            capture / distill / produce / reindex
├── retrieval/
│   ├── embeddings.py         HashingEmbedder y OpenAICompatEmbedder
│   └── index.py              busqueda coseno + armado de contexto
├── agents/
│   └── handoff.py            cola de escalado con claim atomico
├── api/                      FastAPI: esquemas + rutas
├── eval/                     metricas, dataset y corrida reproducible
└── cli.py                    operacion sin HTTP
```

Las dependencias apuntan hacia adentro: `domain` no importa nada del proyecto,
`store` y `retrieval` no conocen la API, y la API es la unica capa que traduce
excepciones de dominio a codigos HTTP.
