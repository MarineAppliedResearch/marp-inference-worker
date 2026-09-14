"""Coordinator URL resolution for authenticated model delivery."""

from marp_inference_worker.jobs.coordinator_client import CoordinatorClient
from marp_inference_worker.models import model_cache
from marp_inference_worker.models.model_spec import ModelSpec


def test_relative_artifact_uses_coordinator_origin_and_service_token() -> None:
    client = CoordinatorClient("https://api.example.test", "secret")
    try:
        url, headers = client.resolve_artifact_url("/api/v2/model/91/artifact")
    finally:
        client.close()

    assert url == "https://api.example.test/api/v2/model/91/artifact"
    assert headers == {"Authorization": "Bearer secret"}


def test_service_token_is_not_sent_to_another_origin() -> None:
    client = CoordinatorClient("https://api.example.test", "secret")
    try:
        url, headers = client.resolve_artifact_url("https://files.example.test/model.pt")
    finally:
        client.close()

    assert url == "https://files.example.test/model.pt"
    assert headers is None


def test_extensionless_api_route_keeps_the_declared_model_format(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(model_cache, "_CACHE_ROOT", tmp_path)
    spec = ModelSpec(
        model_id="registered-model",
        engine="ultralytics",
        model_arch="yolo",
        task="detect",
        artifact={
            "url": "https://api.example.test/api/v2/model/91/artifact",
            "format": "pt",
            "sha256": "a" * 64,
        },
    )

    assert model_cache.get_artifact_cache_path(spec).name == "artifact.pt"
