"""Metricas de retrieval y validacion del set de evaluacion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import event

from layered_memory.config import Settings
from layered_memory.eval.dataset import load_cases, load_corpus, seed, validate_cases
from layered_memory.eval.metrics import (
    CaseResult,
    EvalReport,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from layered_memory.eval.run_eval import build_parser, run_evaluation
from layered_memory.memory.service import MemoryService
from layered_memory.retrieval.embeddings import HashingEmbedder
from layered_memory.store.db import build_engine


def test_hit_and_recall_at_k() -> None:
    ranked = ["a", "b", "c", "d"]

    assert hit_at_k(ranked, ["c"], 3) is True
    assert hit_at_k(ranked, ["c"], 2) is False
    assert recall_at_k(ranked, ["a", "z"], 4) == 0.5
    assert recall_at_k(ranked, [], 4) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_ndcg_rewards_earlier_hits() -> None:
    early = ndcg_at_k(["a", "b", "c", "d"], ["a"], 4)
    late = ndcg_at_k(["d", "c", "b", "a"], ["a"], 4)

    assert early > late
    assert early == pytest.approx(1.0)
    assert ndcg_at_k(["a"], [], 4) == 0.0


def test_precision_penalises_padding() -> None:
    assert precision_at_k(["a", "x", "y", "z"], ["a"], 4) == pytest.approx(0.25)
    assert precision_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)


def test_case_result_metrics_shape() -> None:
    result = CaseResult(
        case_id="q1",
        query="consulta",
        expected=("x",),
        ranked=("x", "y"),
        scores=(0.9, 0.2),
    )
    metrics = result.metrics((1, 3))

    assert metrics["hit@1"] == 1.0
    assert metrics["mrr"] == pytest.approx(1.0)
    assert result.first_relevant_rank == 1
    assert result.as_dict()["metrics"]["hit@1"] == 1.0


def test_report_aggregates_and_lists_misses() -> None:
    results = (
        CaseResult(case_id="q1", query="a", expected=("x",), ranked=("x",), scores=(0.5,)),
        CaseResult(case_id="q2", query="b", expected=("y",), ranked=("z",), scores=(0.5,)),
    )
    report = EvalReport(
        backend="hash",
        model="hashing-v1",
        dim=64,
        ks=(1,),
        results=results,
        corpus_size=10,
        misses=("q2: b",),
    )

    aggregate = report.aggregate()

    assert aggregate["hit@1"] == 0.5
    assert aggregate["mrr"] == 0.5
    assert "| q1 |" in report.to_markdown()
    assert "promedio" in report.to_markdown()
    assert "q2: b" in report.to_text()


def _write(
    tmp_path: Path, corpus: dict[str, Any], cases: list[dict[str, Any]]
) -> tuple[Path, Path]:
    corpus_path = tmp_path / "corpus.json"
    dataset_path = tmp_path / "dataset.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    dataset_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    return corpus_path, dataset_path


def test_corpus_rejects_duplicate_keys(tmp_path: Path, corpus: dict[str, Any]) -> None:
    corpus["documents"].append(dict(corpus["documents"][0]))
    corpus_path, _ = _write(tmp_path, corpus, [])

    with pytest.raises(ValueError, match="duplicadas"):
        load_corpus(corpus_path)


def test_corpus_rejects_a_dangling_parent(tmp_path: Path, corpus: dict[str, Any]) -> None:
    corpus["documents"][1]["parent"] = "no-existe"
    corpus_path, _ = _write(tmp_path, corpus, [])

    with pytest.raises(ValueError, match="padre inexistente"):
        load_corpus(corpus_path)


def test_corpus_rejects_an_unknown_layer(tmp_path: Path, corpus: dict[str, Any]) -> None:
    corpus["documents"][0]["layer"] = "conocimiento"
    corpus_path, _ = _write(tmp_path, corpus, [])

    with pytest.raises(ValueError, match="capa invalida"):
        load_corpus(corpus_path)


def test_cases_reject_an_unknown_expected_key(tmp_path: Path, corpus: dict[str, Any]) -> None:
    corpus_path, dataset_path = _write(
        tmp_path, corpus, [{"id": "q1", "query": "x", "expected": ["fantasma"]}]
    )

    with pytest.raises(ValueError, match="ausentes del corpus"):
        validate_cases(load_corpus(corpus_path), load_cases(dataset_path))


def test_seed_maps_keys_to_ids_in_promotion_order(
    tmp_path: Path, corpus: dict[str, Any], service: MemoryService, session: Any
) -> None:
    corpus_path, _ = _write(tmp_path, corpus, [])
    loaded = load_corpus(corpus_path)

    key_to_id = seed(service, session, loaded)

    assert len(key_to_id) == loaded.size
    assert key_to_id["oncall"] != key_to_id["oncall-insight"]


def test_run_evaluation_measures_the_demo_corpus(tmp_path: Path, corpus: dict[str, Any]) -> None:
    corpus_path, dataset_path = _write(
        tmp_path,
        corpus,
        [
            {
                "id": "q1",
                "query": "certificado del gateway de pagos vencido",
                "expected": ["oncall"],
            },
            {
                "id": "q2",
                "query": "por que el alta de vendedor demora seis dias",
                "expected": ["rrhh"],
            },
            {
                "id": "q3",
                "query": "runbook para renovar certificados",
                "expected": ["oncall-artifact"],
            },
        ],
    )
    settings = Settings(database_url="sqlite:///:memory:", embedding_dim=128)

    report = run_evaluation(
        corpus_path=corpus_path, dataset_path=dataset_path, settings=settings, ks=(1, 3)
    )

    assert report.corpus_size == 5
    assert len(report.results) == 3
    assert 0.0 <= report.aggregate()["hit@3"] <= 1.0
    assert report.aggregate()["hit@3"] >= 0.5


def test_run_evaluation_disposes_its_temporary_engine(
    tmp_path: Path,
    corpus: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from layered_memory.eval import run_eval

    corpus_path, dataset_path = _write(
        tmp_path,
        corpus,
        [{"id": "q1", "query": "certificado vencido", "expected": ["oncall"]}],
    )
    engine = build_engine("sqlite:///:memory:")
    closed_connections = 0

    @event.listens_for(engine, "close")
    def count_closed_connection(*_args: object) -> None:
        nonlocal closed_connections
        closed_connections += 1

    monkeypatch.setattr(run_eval, "build_engine", lambda _url: engine)

    run_evaluation(
        corpus_path=corpus_path,
        dataset_path=dataset_path,
        settings=Settings(database_url="sqlite:///:memory:", embedding_dim=128),
        ks=(1,),
    )

    assert closed_connections == 1, "run_evaluation dejo abierta su conexion SQLite"


def test_run_evaluation_closes_its_embedder(
    tmp_path: Path,
    corpus: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from layered_memory.eval import run_eval

    class TrackingEmbedder(HashingEmbedder):
        closed = False

        def close(self) -> None:
            self.closed = True

    corpus_path, dataset_path = _write(
        tmp_path,
        corpus,
        [{"id": "q1", "query": "certificado vencido", "expected": ["oncall"]}],
    )
    embedder = TrackingEmbedder(dim=128)
    monkeypatch.setattr(run_eval, "build_embedder", lambda _settings: embedder)

    run_evaluation(
        corpus_path=corpus_path,
        dataset_path=dataset_path,
        settings=Settings(database_url="sqlite:///:memory:", embedding_dim=128),
        ks=(1,),
    )

    assert embedder.closed, "run_evaluation dejo abierto su cliente de embeddings"


def test_eval_cli_reports_bad_ks(capsys: pytest.CaptureFixture[str]) -> None:
    from layered_memory.eval.run_eval import main

    code = main(["--corpus", "x", "--dataset", "y", "--ks", "a,b"])

    assert code == 2
    assert "ks" in capsys.readouterr().err


def test_eval_cli_parser_defaults() -> None:
    args = build_parser().parse_args(["--corpus", "c.json", "--dataset", "d.json"])

    assert args.ks == "1,3,5,10"
    assert args.database_url == "sqlite:///:memory:"
