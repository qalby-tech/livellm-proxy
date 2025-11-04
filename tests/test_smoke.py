"""
Simple smoke tests to verify basic endpoint functionality.
These tests ensure the app starts and endpoints respond correctly.
"""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def smoke_client():
    """Create a test client for smoke tests without dependency overrides."""
    with TestClient(app) as test_client:
        yield test_client


def test_app_starts(smoke_client):
    """Test that the FastAPI app starts successfully."""
    assert smoke_client is not None
    assert app is not None


def test_ping_endpoint(smoke_client):
    """Test that the ping endpoint responds."""
    response = smoke_client.get("/livellm/ping")
    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "ok"}


def test_get_configs_endpoint_exists(smoke_client):
    """Test that the get configs endpoint is accessible."""
    response = smoke_client.get("/livellm/providers/configs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_add_config_endpoint_exists(smoke_client):
    """Test that the add config endpoint responds (even if with validation error)."""
    response = smoke_client.post("/livellm/providers/config", json={})
    # Should fail validation but endpoint should exist
    assert response.status_code in [400, 422]


def test_delete_config_endpoint_exists(smoke_client):
    """Test that the delete config endpoint is accessible."""
    response = smoke_client.delete("/livellm/providers/config/nonexistent-uid")
    # Should fail but endpoint should exist
    assert response.status_code in [400, 404]


@pytest.mark.agent
def test_agent_run_endpoint_exists(smoke_client):
    """Test that the agent run endpoint is accessible."""
    response = smoke_client.post("/livellm/agent/run", json={})
    # Should fail validation but endpoint should exist
    assert response.status_code in [400, 422]


@pytest.mark.agent
def test_agent_run_stream_endpoint_exists(smoke_client):
    """Test that the agent run stream endpoint is accessible."""
    response = smoke_client.post("/livellm/agent/run_stream", json={})
    # Should fail validation but endpoint should exist
    assert response.status_code in [400, 422]


@pytest.mark.audio
def test_audio_speak_endpoint_exists(smoke_client):
    """Test that the audio speak endpoint is accessible."""
    response = smoke_client.post("/livellm/audio/speak", json={})
    # Should fail validation but endpoint should exist
    assert response.status_code in [400, 422]


@pytest.mark.audio
def test_audio_speak_stream_endpoint_exists(smoke_client):
    """Test that the audio speak stream endpoint is accessible."""
    response = smoke_client.post("/livellm/audio/speak_stream", json={})
    # Should fail validation but endpoint should exist
    assert response.status_code in [400, 422]


@pytest.mark.audio
def test_audio_transcribe_endpoint_exists(smoke_client):
    """Test that the audio transcribe endpoint is accessible."""
    # Form data endpoint - should fail without proper multipart data
    response = smoke_client.post("/livellm/audio/transcribe")
    assert response.status_code in [400, 422]


@pytest.mark.audio
def test_audio_transcribe_json_endpoint_exists(smoke_client):
    """Test that the audio transcribe json endpoint is accessible."""
    response = smoke_client.post("/livellm/audio/transcribe_json", json={})
    # Should fail validation but endpoint should exist
    assert response.status_code in [400, 422]


def test_openapi_docs_available(smoke_client):
    """Test that OpenAPI documentation is available."""
    response = smoke_client.get("/livellm/openapi.json")
    assert response.status_code == 200
    assert "openapi" in response.json()

