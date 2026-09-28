"""Superficie HTTP: ciclo completo, orden de rutas y traduccion de errores."""

from __future__ import annotations

from fastapi.testclient import TestClient

from layered_memory import __version__


def test_health_reports_the_active_embedding_backend(client: TestClient) -> None:
    body = client.get("/health").json()

    assert body == {
        "status": "ok",
        "version": __version__,
        "embedding_backend": "hash",
        "embedding_model": "hashing-v1",
        "embedding_dim": 128,
        "database": "sqlite",
    }


def test_full_promotion_cycle_over_http(client: TestClient) -> None:
    captured = client.post(
        "/memories/capture",
        json={
            "content": "El job nocturno de respaldos fallaba en silencio desde hace dos semanas.",
            "title": "Respaldos",
            "tags": ["infra", "Backups"],
        },
    )
    assert captured.status_code == 201
    raw = captured.json()
    assert raw["layer"] == "raw"
    assert raw["tags"] == ["infra", "backups"]

    distilled = client.post(
        f"/memories/{raw['id']}/distill",
        json={
            "title": "El job de respaldos no reporta errores",
            "content": "El exit code se descartaba, asi que el fallo no llegaba a nadie.",
        },
    )
    assert distilled.status_code == 201
    insight = distilled.json()
    assert insight["layer"] == "insight"
    assert insight["parent_id"] == raw["id"]

    produced = client.post(
        f"/memories/{insight['id']}/produce",
        json={
            "title": "Checklist de jobs programados",
            "content": "1. Exit code. 2. Alerta. 3. Reintento.",
        },
    )
    assert produced.status_code == 201
    artifact = produced.json()
    assert artifact["layer"] == "artifact"

    lineage = client.get(f"/memories/{artifact['id']}/lineage").json()
    assert [row["layer"] for row in lineage] == ["raw", "insight", "artifact"]

    children = client.get(f"/memories/{raw['id']}/children").json()
    assert [row["id"] for row in children] == [insight["id"]]


def test_stats_route_is_not_shadowed_by_the_memory_id_route(client: TestClient) -> None:
    client.post("/memories/capture", json={"content": "contenido"})

    response = client.get("/memories/stats")

    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_search_and_context_endpoints(client: TestClient) -> None:
    client.post(
        "/memories/capture",
        json={
            "content": "El servidor de base de datos se queda sin memoria a las 3 AM.",
            "title": "Memoria agotada",
        },
    )
    client.post(
        "/memories/capture",
        json={
            "content": "La campana de ventas del trimestre cierra el 30 de junio.",
            "title": "Cierre",
        },
    )

    found = client.post(
        "/memories/search", json={"query": "memoria agotada del servidor de base de datos", "k": 1}
    )
    assert found.status_code == 200
    assert found.json()["count"] == 1
    assert found.json()["hits"][0]["title"] == "Memoria agotada"

    context = client.post(
        "/memories/context", json={"query": "campana de ventas", "k": 2, "max_chars": 200}
    )
    assert context.status_code == 200
    assert context.json()["chars"] <= 200
    assert context.json()["ids"]


def test_handoff_lifecycle_over_http(client: TestClient) -> None:
    created = client.post(
        "/handoffs", json={"reason": "descuento fuera de politica", "session_id": "s1"}
    )
    assert created.status_code == 201
    handoff_id = created.json()["id"]
    assert client.get("/handoffs/depth").json() == {"queued": 1}

    claimed = client.post(f"/handoffs/{handoff_id}/claim", json={"claimed_by": "robot-1"})
    assert claimed.json()["status"] == "claimed"
    assert claimed.json()["claimed_by"] == "robot-1"

    conflict = client.post(f"/handoffs/{handoff_id}/claim", json={"claimed_by": "robot-2"})
    assert conflict.status_code == 409

    resolved = client.post(
        f"/handoffs/{handoff_id}/resolve",
        json={"resolution": "aprobado al 10%", "resolved_by": "jefe"},
    )
    assert resolved.json()["status"] == "resolved"
    assert client.get("/handoffs/depth").json() == {"queued": 0}

    listed = client.get("/handoffs", params={"status": "resolved"}).json()
    assert [row["id"] for row in listed] == [handoff_id]


def test_unknown_memory_returns_404(client: TestClient) -> None:
    assert client.get("/memories/no-existe").status_code == 404
    assert client.get("/memories/no-existe/children").status_code == 404


def test_missing_or_invalid_payloads_return_422(client: TestClient) -> None:
    assert client.post("/memories/capture", json={}).status_code == 422
    assert client.post("/memories/capture", json={"content": "   "}).status_code == 422

    captured = client.post("/memories/capture", json={"content": "algo valido"})
    blank = client.post(
        f"/memories/{captured.json()['id']}/distill",
        json={"title": "t", "content": "   "},
    )
    assert blank.status_code == 422


def test_delete_removes_the_memory_and_its_lineage_endpoint_404s(
    client: TestClient,
) -> None:
    captured = client.post("/memories/capture", json={"content": "contenido descartable"})
    memory_id = captured.json()["id"]

    assert client.delete(f"/memories/{memory_id}").status_code == 204
    assert client.get(f"/memories/{memory_id}").status_code == 404


def test_reindex_endpoint_reports_the_count(client: TestClient) -> None:
    client.post("/memories/capture", json={"content": "contenido para reindexar"})

    assert client.post("/memories/reindex").json() == {"reindexed": 1}


def test_openapi_schema_is_generated(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()

    assert "/memories/search" in schema["paths"]
    assert schema["info"]["title"] == "layered-memory"
