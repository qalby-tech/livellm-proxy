"""Tests for main application endpoints."""

import pytest


def test_ping_endpoint(client):
    """Test the /ping endpoint returns ok status."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_app_has_agent_router(client):
    """Test that agent router is included in the app."""
    # Check that agent endpoints exist in the OpenAPI schema
    openapi_schema = client.app.openapi()
    assert "/agent/run" in openapi_schema["paths"]
    assert "/agent/run_stream" in openapi_schema["paths"]


def test_app_has_audio_router(client):
    """Test that audio router is included in the app."""
    # Check that audio endpoints exist in the OpenAPI schema
    openapi_schema = client.app.openapi()
    assert "/audio/speak" in openapi_schema["paths"]
    assert "/audio/speak_stream" in openapi_schema["paths"]
    assert "/audio/transcribe" in openapi_schema["paths"]

