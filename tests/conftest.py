"""Pytest configuration and fixtures for testing the FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from main import app
from managers.agent import AgentManager
from managers.audio import AudioManager
from routers.agent import get_agent_manager
from routers.audio import get_audio_manager


@pytest.fixture
def mock_agent_manager():
    """Create a mock AgentManager with mocked run method."""
    manager = MagicMock(spec=AgentManager)
    manager.run = AsyncMock()
    return manager


@pytest.fixture
def mock_audio_manager():
    """Create a mock AudioManager with mocked speak and transcribe methods."""
    manager = MagicMock(spec=AudioManager)
    manager.speak = AsyncMock()
    manager.transcribe = AsyncMock()
    return manager


@pytest.fixture
def client(mock_agent_manager, mock_audio_manager):
    """Create a test client for the FastAPI application with dependency overrides."""
    app.dependency_overrides[get_agent_manager] = lambda: mock_agent_manager
    app.dependency_overrides[get_audio_manager] = lambda: mock_audio_manager
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up after test
    app.dependency_overrides.clear()


@pytest.fixture
def agent_request_payload():
    """Sample agent request payload."""
    return {
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "tools": []
    }


@pytest.fixture
def agent_headers():
    """Sample agent request headers."""
    return {
        "X-Api-Key": "test-api-key",
        "X-Provider": "openai"
    }


@pytest.fixture
def audio_speak_payload():
    """Sample audio speak request payload."""
    return {
        "model": "tts-1",
        "text": "Hello world",
        "voice": "alloy",
        "output_format": "mp3"
    }


@pytest.fixture
def audio_headers():
    """Sample audio request headers."""
    return {
        "X-Api-Key": "test-api-key",
        "X-Provider": "openai"
    }
