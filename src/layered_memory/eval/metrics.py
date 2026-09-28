"""Metricas de recuperacion.

Implementa hit@k, recall@k, MRR y nDCG@k con la convencion estandar: la
relevancia es binaria y se toma de la posicion 1 como la mejor. Todas las
funciones toleran que ``ranked`` tenga menos elementos que ``k``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import log2

from layered_memory.eval.errors import EvaluationDataError
from layered_memory.types import JsonObject


def hit_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> bool:
    """True si al menos un documento esperado aparece en el top-k."""
    return bool(set(ranked[:k]) & set(expected))


def recall_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Fraccion de documentos esperados que aparece en el top-k."""
    if not expected:
        return 0.0
    found = len(set(ranked[:k]) & set(expected))
    return found / len(set(expected))


def reciprocal_rank(ranked: Sequence[str], expected: Sequence[str]) -> float:
    """1/rango del primer documento esperado. 0 si no aparece."""
    wanted = set(expected)
    for position, doc_id in enumerate(ranked, start=1):
        if doc_id in wanted:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Ganancia normalizada por descuentos, con relevancia binaria."""
    if not expected:
        return 0.0
    wanted = set(expected)
    dcg = sum(
        1.0 / log2(position + 1) for position, doc in enumerate(ranked[:k], 1) if doc in wanted
    )
    ideal_hits = min(len(wanted), k)
    idcg = sum(1.0 / log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def precision_at_k(ranked: Sequence[str], expected: Sequence[str], k: int) -> float:
    """Fraccion de relevantes dentro del top-k. Se penaliza el relleno."""
    if k <= 0:
        return 0.0
    relevant = len(set(ranked[:k]) & set(expected))
    return relevant / min(k, max(len(ranked), 1))


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Una consulta con sus documentos relevantes conocidos."""

    id: str
    query: str
    expected: tuple[str, ...]
    note: str = ""
    group: str = "default"

    @classmethod
    def from_dict(cls, data: JsonObject) -> EvalCase:
        missing = {"id", "query", "expected"} - set(data)
        if missing:
            raise EvaluationDataError(f"el caso {data.get('id', '?')} no tiene {sorted(missing)}")
        expected = data["expected"]
        if not isinstance(expected, list) or not expected:
            raise EvaluationDataError(
                f"el caso {data['id']} necesita 'expected' como lista no vacia"
            )
        return cls(
            id=str(data["id"]),
            query=str(data["query"]),
            expected=tuple(str(item) for item in expected),
            note=str(data.get("note", "")),
            group=str(data.get("group", "default")),
        )


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Resultado de evaluar un caso, con el ranking completo para depurar."""

    case_id: str
    query: str
    expected: tuple[str, ...]
    ranked: tuple[str, ...]
    scores: tuple[float, ...]
    group: str = "default"

    def metrics(self, ks: Sequence[int]) -> dict[str, float]:
        values: dict[str, float] = {"mrr": reciprocal_rank(self.ranked, self.expected)}
        for k in ks:
            values[f"hit@{k}"] = 1.0 if hit_at_k(self.ranked, self.expected, k) else 0.0
            values[f"recall@{k}"] = recall_at_k(self.ranked, self.expected, k)
            values[f"ndcg@{k}"] = ndcg_at_k(self.ranked, self.expected, k)
            values[f"precision@{k}"] = precision_at_k(self.ranked, self.expected, k)
        return values

    @property
    def first_relevant_rank(self) -> int | None:
        wanted = set(self.expected)
        for position, doc in enumerate(self.ranked, start=1):
            if doc in wanted:
                return position
        return None

    def as_dict(self) -> JsonObject:
        return {
            "case_id": self.case_id,
            "group": self.group,
            "query": self.query,
            "expected": list(self.expected),
            "ranked": list(self.ranked),
            "scores": [round(s, 6) for s in self.scores],
            "first_relevant_rank": self.first_relevant_rank,
            "metrics": {k: round(v, 4) for k, v in self.metrics((1, 3, 5, 10)).items()},
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Agregado de la corrida, con los casos individuales para inspeccionar."""

    backend: str
    model: str
    dim: int
    ks: tuple[int, ...]
    results: tuple[CaseResult, ...]
    corpus_size: int
    misses: tuple[str, ...] = field(default=())

    def aggregate(self) -> dict[str, float]:
        if not self.results:
            return {}
        keys = self.results[0].metrics(self.ks).keys()
        totals = {key: 0.0 for key in keys}
        for result in self.results:
            for key, value in result.metrics(self.ks).items():
                totals[key] += value
        return {key: round(value / len(self.results), 4) for key, value in totals.items()}

    def aggregate_by_group(self) -> dict[str, dict[str, float]]:
        """Metricas separadas por grupo de casos.

        Comparar "misma palabra" contra "misma idea, otras palabras" es lo que
        separa un indice lexical de uno semantico, y un promedio unico lo
        esconde.
        """
        groups: dict[str, list[CaseResult]] = {}
        for result in self.results:
            groups.setdefault(result.group, []).append(result)
        return {
            name: _mean([r.metrics(self.ks) for r in rows]) for name, rows in sorted(groups.items())
        }

    def as_dict(self) -> JsonObject:
        return {
            "backend": self.backend,
            "model": self.model,
            "dim": self.dim,
            "corpus_size": self.corpus_size,
            "cases": len(self.results),
            "ks": list(self.ks),
            "aggregate": self.aggregate(),
            "aggregate_by_group": self.aggregate_by_group(),
            "misses": list(self.misses),
            "results": [r.as_dict() for r in self.results],
        }

    def to_markdown(self) -> str:
        """Tabla por caso, con el rango del primer relevante para ver falladas."""
        agg = self.aggregate()
        cuts = " | ".join(str(k) for k in self.ks)
        head = f"| caso | consulta | primer relevante | {cuts} | mrr |"
        sep = "|---" * (len(self.ks) + 4) + "|"
        rows = [head, sep]
        for result in self.results:
            metrics = result.metrics(self.ks)
            rank = result.first_relevant_rank
            cells = [result.case_id, f"{result.query[:52]}", str(rank) if rank else "-"]
            cells += [f"{metrics[f'hit@{k}']:.0f}" for k in self.ks]
            cells.append(f"{metrics['mrr']:.2f}")
            rows.append("| " + " | ".join(cells) + " |")
        by_group = self.aggregate_by_group()
        if len(by_group) > 1:
            rows.append("")
            rows.append("por grupo:")
            rows.append("")
            rows.append(
                "| grupo | casos | " + " | ".join(f"hit@{k}" for k in self.ks) + " | mrr | ndcg@5 |"
            )
            rows.append("|---" * (len(self.ks) + 4) + "|")
            for name, values in by_group.items():
                count = sum(1 for r in self.results if r.group == name)
                cells = [name, str(count)]
                cells += [f"{values.get(f'hit@{k}', 0.0):.2f}" for k in self.ks]
                cells.append(f"{values.get('mrr', 0.0):.2f}")
                cells.append(f"{values.get('ndcg@5', 0.0):.2f}")
                rows.append("| " + " | ".join(cells) + " |")
        summary = (
            "| **promedio** | | | "
            + " | ".join(f"**{agg[f'hit@{k}']:.2f}**" for k in self.ks)
            + f" | **{agg['mrr']:.2f}** |"
        )
        rows.append(summary)
        return "\n".join(rows)

    def to_text(self) -> str:
        agg = self.aggregate()
        lines = [
            f"backend={self.backend} model={self.model} dim={self.dim}",
            f"corpus={self.corpus_size} casos={len(self.results)}",
            "",
        ]
        lines += [f"  {key:<14} {value:.4f}" for key, value in agg.items()]
        by_group = self.aggregate_by_group()
        if len(by_group) > 1:
            lines += ["", "por grupo:"]
            for name, values in by_group.items():
                lines.append(f"  [{name}]")
                lines += [f"    {key:<12} {value:.4f}" for key, value in values.items()]
        if self.misses:
            lines += ["", "casos sin acierto en el top-10:"]
            lines += [f"  - {miss}" for miss in self.misses]
        return "\n".join(lines)


def _mean(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: round(sum(row.get(key, 0.0) for row in rows) / len(rows), 4) for key in rows[0]}


__all__ = [
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "hit_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
