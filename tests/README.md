# Tests

This directory contains pytest tests for the FastAPI application.

## Installation

First, install the testing dependencies:

```bash
pip install pytest pytest-asyncio httpx
```

Or if using uv:

```bash
uv pip install pytest pytest-asyncio httpx
```

## Running Tests

Run all tests:
```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

Run specific test file:
```bash
pytest tests/test_agent.py
pytest tests/test_audio.py
pytest tests/test_main.py
```

Run specific test class:
```bash
pytest tests/test_agent.py::TestAgentRun
```

Run specific test function:
```bash
pytest tests/test_agent.py::TestAgentRun::test_agent_run_success
```

Run with coverage (requires pytest-cov):
```bash
pip install pytest-cov
pytest --cov=. --cov-report=html
```

## Test Structure

- `conftest.py` - Pytest fixtures and configuration
- `test_main.py` - Tests for main application endpoints
- `test_agent.py` - Tests for agent endpoints (`/agent/run`, `/agent/run_stream`)
- `test_audio.py` - Tests for audio endpoints (`/audio/speak`, `/audio/speak_stream`, `/audio/transcribe`)

## Mocking

All tests mock the manager classes (`AgentManager`, `AudioManager`) to avoid making actual API calls to external services. This ensures:

1. Tests run quickly
2. Tests don't require API keys
3. Tests are deterministic and reliable
4. No external dependencies

## Test Coverage

The test suite covers:

- ✅ All endpoints (GET, POST)
- ✅ Request validation (missing headers, invalid payloads)
- ✅ Success responses
- ✅ Error handling (invalid providers, internal errors)
- ✅ Streaming endpoints
- ✅ Multiple provider types
- ✅ Custom base URLs
- ✅ File upload (transcription)
- ✅ Different message types and tools

