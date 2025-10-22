"""Tests for agent endpoints."""

import pytest
from models.agent.run import AgentResponse, AgentResponseUsage


class TestAgentRun:
    """Tests for the /agent/run endpoint."""

    def test_agent_run_success(self, client, mock_agent_manager, agent_request_payload, agent_headers):
        """Test successful agent run."""
        # Setup mock
        mock_agent_manager.run.return_value = AgentResponse(
            output="Hello! I'm doing well, thank you for asking.",
            usage=AgentResponseUsage(input_tokens=10, output_tokens=15)
        )

        # Make request
        response = client.post(
            "/agent/run",
            json=agent_request_payload,
            headers=agent_headers
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert "usage" in data
        assert data["output"] == "Hello! I'm doing well, thank you for asking."
        assert data["usage"]["input_tokens"] == 10
        assert data["usage"]["output_tokens"] == 15
        
        # Verify manager was called
        mock_agent_manager.run.assert_called_once()

    def test_agent_run_with_base_url(self, client, mock_agent_manager, agent_request_payload, agent_headers):
        """Test agent run with custom base URL."""
        mock_agent_manager.run.return_value = AgentResponse(
            output="Response",
            usage=AgentResponseUsage(input_tokens=5, output_tokens=10)
        )

        headers = {**agent_headers, "X-Base-Url": "https://custom.api.com"}
        response = client.post(
            "/agent/run",
            json=agent_request_payload,
            headers=headers
        )

        assert response.status_code == 200
        mock_agent_manager.run.assert_called_once()

    def test_agent_run_missing_api_key(self, client, agent_request_payload):
        """Test agent run without API key header."""
        headers = {"X-Provider": "openai"}
        response = client.post(
            "/agent/run",
            json=agent_request_payload,
            headers=headers
        )
        
        assert response.status_code == 422  # Unprocessable Entity

    def test_agent_run_missing_provider(self, client, agent_request_payload):
        """Test agent run without provider header."""
        headers = {"X-Api-Key": "test-key"}
        response = client.post(
            "/agent/run",
            json=agent_request_payload,
            headers=headers
        )
        
        assert response.status_code == 422  # Unprocessable Entity

    def test_agent_run_invalid_provider(self, client, mock_agent_manager, agent_request_payload):
        """Test agent run with invalid provider."""
        mock_agent_manager.run.side_effect = ValueError("Invalid provider")

        headers = {"X-Api-Key": "test-key", "X-Provider": "invalid_provider"}
        response = client.post(
            "/agent/run",
            json=agent_request_payload,
            headers=headers
        )
        
        assert response.status_code == 400
        assert "Invalid provider" in response.json()["detail"]

    def test_agent_run_with_different_providers(self, client, mock_agent_manager, agent_request_payload):
        """Test agent run with different provider types."""
        mock_agent_manager.run.return_value = AgentResponse(
            output="Response",
            usage=AgentResponseUsage(input_tokens=5, output_tokens=10)
        )

        for provider in ["openai", "google", "anthropic", "groq"]:
            headers = {"X-Api-Key": "test-key", "X-Provider": provider}
            response = client.post(
                "/agent/run",
                json=agent_request_payload,
                headers=headers
            )
            assert response.status_code == 200

    def test_agent_run_invalid_payload(self, client, agent_headers):
        """Test agent run with invalid payload."""
        invalid_payload = {
            "model": "gpt-4"
            # Missing required fields
        }
        
        response = client.post(
            "/agent/run",
            json=invalid_payload,
            headers=agent_headers
        )
        
        assert response.status_code == 422


class TestAgentRunStream:
    """Tests for the /agent/run_stream endpoint."""

    def test_agent_run_stream_success(self, client, mock_agent_manager, agent_request_payload, agent_headers):
        """Test successful streaming agent run."""
        # Setup mock async generator
        async def mock_stream():
            yield '{"output": "Hello", "usage": {"input_tokens": 10, "output_tokens": 5}}\n'
            yield '{"output": " World", "usage": {"input_tokens": 10, "output_tokens": 10}}\n'
        
        mock_agent_manager.run.return_value = mock_stream()

        # Make request
        response = client.post(
            "/agent/run_stream",
            json=agent_request_payload,
            headers=agent_headers
        )

        # Assertions
        assert response.status_code == 200
        assert "application/x-ndjson" in response.headers["content-type"]
        
        # Verify manager was called with stream=True
        mock_agent_manager.run.assert_called_once()
        call_kwargs = mock_agent_manager.run.call_args[1]
        assert call_kwargs["stream"] is True

    def test_agent_run_stream_with_base_url(self, client, mock_agent_manager, agent_request_payload, agent_headers):
        """Test streaming with custom base URL."""
        async def mock_stream():
            yield '{"output": "test", "usage": {"input_tokens": 1, "output_tokens": 1}}\n'
        
        mock_agent_manager.run.return_value = mock_stream()

        headers = {**agent_headers, "X-Base-Url": "https://custom.api.com"}
        response = client.post(
            "/agent/run_stream",
            json=agent_request_payload,
            headers=headers
        )

        assert response.status_code == 200

    def test_agent_run_stream_missing_headers(self, client, agent_request_payload):
        """Test streaming without required headers."""
        response = client.post(
            "/agent/run_stream",
            json=agent_request_payload
        )
        
        assert response.status_code == 422

    def test_agent_run_stream_invalid_provider(self, client, mock_agent_manager, agent_request_payload):
        """Test streaming with invalid provider."""
        mock_agent_manager.run.side_effect = ValueError("Invalid provider")

        headers = {"X-Api-Key": "test-key", "X-Provider": "invalid"}
        response = client.post(
            "/agent/run_stream",
            json=agent_request_payload,
            headers=headers
        )
        
        assert response.status_code == 400


class TestAgentWithTools:
    """Tests for agent endpoints with tools."""

    def test_agent_run_with_web_search_tool(self, client, mock_agent_manager, agent_headers):
        """Test agent run with web search tool."""
        mock_agent_manager.run.return_value = AgentResponse(
            output="Search result",
            usage=AgentResponseUsage(input_tokens=10, output_tokens=20)
        )

        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Search for Python tutorials"}],
            "tools": [
                {
                    "kind": "web_search",
                    "search_context_size": "medium"
                }
            ]
        }

        response = client.post(
            "/agent/run",
            json=payload,
            headers=agent_headers
        )

        assert response.status_code == 200

    def test_agent_run_with_mcp_tool(self, client, mock_agent_manager, agent_headers):
        """Test agent run with MCP tool."""
        mock_agent_manager.run.return_value = AgentResponse(
            output="MCP result",
            usage=AgentResponseUsage(input_tokens=10, output_tokens=20)
        )

        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Use MCP tool"}],
            "tools": [
                {
                    "kind": "mcp_streamable_server",
                    "url": "https://mcp.example.com",
                    "prefix": "mcp_"
                }
            ]
        }

        response = client.post(
            "/agent/run",
            json=payload,
            headers=agent_headers
        )

        assert response.status_code == 200

    def test_agent_run_with_multiple_messages(self, client, mock_agent_manager, agent_headers):
        """Test agent run with conversation history."""
        mock_agent_manager.run.return_value = AgentResponse(
            output="Response to history",
            usage=AgentResponseUsage(input_tokens=30, output_tokens=20)
        )

        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "model", "content": "First response"},
                {"role": "user", "content": "Second message"}
            ],
            "tools": []
        }

        response = client.post(
            "/agent/run",
            json=payload,
            headers=agent_headers
        )

        assert response.status_code == 200

