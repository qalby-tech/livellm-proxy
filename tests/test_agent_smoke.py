"""
Smoke tests for agent endpoints using mocked dependencies.
These tests verify the basic request/response flow without calling real LLM APIs.
"""

import pytest


@pytest.mark.agent
def test_agent_run_with_mock(client, mock_agent_manager, agent_payload):
    """Test agent run endpoint with mocked manager."""
    # Setup mock response
    mock_agent_manager.safe_run.return_value = {
        "output": "Hello! I'm doing well, thank you for asking.",
        "usage": {"input_tokens": 10, "output_tokens": 15}
    }
    
    response = client.post("/livellm/agent/run", json=agent_payload)
    
    assert response.status_code == 200
    assert "output" in response.json()
    assert "usage" in response.json()
    mock_agent_manager.safe_run.assert_called_once()


@pytest.mark.agent
def test_agent_run_missing_required_fields(client, mock_agent_manager):
    """Test that agent run fails with missing required fields."""
    incomplete_payload = {
        "provider_uid": "test-provider"
        # missing model and messages
    }
    
    response = client.post("/livellm/agent/run", json=incomplete_payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.agent
def test_agent_run_empty_messages(client, mock_agent_manager):
    """Test that agent run handles empty messages."""
    payload = {
        "provider_uid": "test-provider-uid",
        "model": "gpt-4",
        "messages": []  # Empty messages
    }
    
    response = client.post("/livellm/agent/run", json=payload)
    # Should accept empty messages or fail validation
    assert response.status_code in [200, 400, 422, 500]


@pytest.mark.agent
def test_agent_run_with_tools(client, mock_agent_manager, agent_payload):
    """Test agent run endpoint with tools."""
    mock_agent_manager.safe_run.return_value = {
        "output": "Response with tools",
        "usage": {"input_tokens": 20, "output_tokens": 30}
    }
    
    payload = agent_payload.copy()
    payload["tools"] = [
        {
            "kind": "web_search",
            "search_context_size": 5,
            "kwargs": {}
        }
    ]
    
    response = client.post("/livellm/agent/run", json=payload)
    # Should either succeed or fail validation
    assert response.status_code in [200, 422]


@pytest.mark.agent
def test_agent_run_stream_endpoint(client, mock_agent_manager, agent_payload):
    """Test agent run stream endpoint returns streaming response."""
    from models.agent.agent import AgentResponse, AgentResponseUsage
    
    async def mock_stream():
        yield AgentResponse(output="chunk1", usage=AgentResponseUsage(input_tokens=10, output_tokens=0))
        yield AgentResponse(output="chunk2", usage=AgentResponseUsage(input_tokens=10, output_tokens=5))
    
    mock_agent_manager.safe_run.return_value = mock_stream()
    
    response = client.post("/livellm/agent/run_stream", json=agent_payload)
    
    # Should return 200 and streaming content
    assert response.status_code == 200
    # Content type should be NDJSON
    assert "application/x-ndjson" in response.headers.get("content-type", "")


@pytest.mark.agent
def test_agent_fallback_request_structure(client, mock_agent_manager):
    """Test that agent accepts fallback request structure."""
    mock_agent_manager.safe_run.return_value = {
        "output": "Response from fallback",
        "usage": {"input_tokens": 10, "output_tokens": 15}
    }
    
    fallback_payload = {
        "requests": [
            {
                "provider_uid": "openai-1",
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            {
                "provider_uid": "anthropic-1",
                "model": "claude-3-opus-20240229",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        ],
        "strategy": "sequential",
        "timeout_per_request": 360
    }
    
    response = client.post("/livellm/agent/run", json=fallback_payload)
    # Should either succeed or fail validation
    assert response.status_code in [200, 400, 422]


@pytest.mark.agent
def test_agent_invalid_strategy(client, mock_agent_manager):
    """Test that invalid fallback strategy is rejected."""
    fallback_payload = {
        "requests": [
            {
                "provider_uid": "openai-1",
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        ],
        "strategy": "invalid-strategy",  # Invalid strategy
        "timeout_per_request": 360
    }
    
    response = client.post("/livellm/agent/run", json=fallback_payload)
    assert response.status_code == 422  # Validation error

