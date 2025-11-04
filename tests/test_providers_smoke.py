"""
Smoke tests for provider configuration management.
These tests verify basic CRUD operations for provider configs.
"""

import pytest


def test_get_empty_configs(client):
    """Test getting configs when none are configured."""
    response = client.get("/livellm/providers/configs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_add_valid_config(client):
    """Test adding a valid provider configuration."""
    config = {
        "uid": "test-openai",
        "provider": "openai",
        "api_key": "sk-test-key-123"
    }
    response = client.post("/livellm/providers/config", json=config)
    assert response.status_code == 201
    assert response.json()["success"] is True


def test_add_config_with_base_url(client):
    """Test adding a provider config with custom base URL."""
    config = {
        "uid": "test-openai-custom",
        "provider": "openai",
        "api_key": "sk-test-key-456",
        "base_url": "https://custom.openai.com/v1"
    }
    response = client.post("/livellm/providers/config", json=config)
    assert response.status_code == 201
    assert response.json()["success"] is True


def test_add_config_missing_required_fields(client):
    """Test that adding config without required fields fails."""
    config = {
        "uid": "test-incomplete"
        # missing provider and api_key
    }
    response = client.post("/livellm/providers/config", json=config)
    assert response.status_code == 422  # Validation error


def test_add_config_invalid_provider(client):
    """Test that adding config with invalid provider fails."""
    config = {
        "uid": "test-invalid",
        "provider": "invalid-provider-name",
        "api_key": "sk-test-key"
    }
    response = client.post("/livellm/providers/config", json=config)
    assert response.status_code == 422  # Validation error


def test_delete_nonexistent_config(client):
    """Test deleting a config that doesn't exist."""
    response = client.delete("/livellm/providers/config/nonexistent-uid")
    assert response.status_code == 400


def test_get_configs_returns_list(client):
    """Test that get configs always returns a list."""
    response = client.get("/livellm/providers/configs")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

