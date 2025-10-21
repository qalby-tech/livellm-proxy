# Qalby Proxy

A unified FastAPI proxy service for multiple AI providers (OpenAI, Anthropic, Google, Groq, ElevenLabs) that provides a consistent interface for agent-based chat and audio processing.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Streaming Endpoints](#streaming-endpoints)
- [Client Implementation Guide](#client-implementation-guide)
- [Error Handling](#error-handling)
- [Supported Providers](#supported-providers)
- [Examples](#examples)

---

## Overview

Qalby Proxy provides a unified API to interact with multiple AI service providers. It uses header-based authentication and provider selection, making it easy to switch between providers without changing your request payloads.

### Key Benefits

- **Unified Interface**: Same API for OpenAI, Anthropic, Google, Groq, and ElevenLabs
- **Header-Based Auth**: Clean separation of authentication from business logic
- **Streaming Support**: Full support for streaming responses (agent text and audio)
- **Provider Caching**: Efficient provider instance caching for better performance
- **Custom Endpoints**: Support for custom base URLs and proxy servers

---

## Features

### Agent Services
- Text-based conversational AI
- Tool/function calling support
- Streaming and non-streaming responses
- Support for OpenAI, Anthropic, Google, and Groq

### Audio Services
- Text-to-Speech (TTS) with streaming
- Speech-to-Text (STT) transcription
- Support for OpenAI and ElevenLabs

---

## Installation

### Prerequisites

- Python 3.12+
- uv (recommended) or pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd qalby-proxy
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
export LOGFIRE_WRITE_TOKEN=your-token  # For monitoring
export OTEL_EXPORTER_OTLP_ENDPOINT=your-endpoint  # For OpenTelemetry
export HOST=0.0.0.0  # Default: 0.0.0.0
export PORT=8000     # Default: 8000
```

4. Run the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

---

## Quick Start

### Test the Server

```bash
# Check server health
curl http://localhost:8000/ping
```

### Simple Agent Request

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-openai-api-key" \
  -H "X-Provider: openai" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "tools": []
  }'
```

### Simple Audio Request

```bash
curl -X POST http://localhost:8000/audio/speak \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-openai-api-key" \
  -H "X-Provider: openai" \
  -d '{
    "model": "tts-1",
    "text": "Hello, world!",
    "voice": "alloy",
    "output_format": "mp3"
  }' \
  --output speech.mp3
```

---

## API Endpoints

### Health Check

**GET** `/ping`

Returns server status.

**Response:**
```json
{"status": "ok"}
```

### Agent Endpoints

#### POST `/agent/run`

Run an agent with a single, complete response.

**Headers:**
- `X-Api-Key` (required): API key for the provider
- `X-Provider` (required): Provider name (`openai`, `anthropic`, `google`, `groq`)
- `X-Base-Url` (optional): Custom base URL for the provider

**Request Body:**
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "Your message"}
  ],
  "tools": [],
  "gen_config": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

**Response:**
```json
{
  "output": "Agent's complete response",
  "usage": {
    "input_tokens": 10,
    "output_tokens": 25
  }
}
```

#### POST `/agent/run_stream`

Stream agent responses as newline-delimited JSON (NDJSON).

**Headers:** Same as `/agent/run`

**Request Body:** Same as `/agent/run`

**Response:** Stream of NDJSON objects
```
{"output": "First", "usage": {}}\n
{"output": " chunk", "usage": {}}\n
{"output": "", "usage": {"input_tokens": 10, "output_tokens": 25}}\n
```

### Audio Endpoints

#### POST `/audio/speak`

Convert text to speech (non-streaming).

**Headers:**
- `X-Api-Key` (required): API key for the provider
- `X-Provider` (required): Provider name (`openai`, `elevenlabs`)
- `X-Base-Url` (optional): Custom base URL for the provider

**Request Body:**
```json
{
  "model": "tts-1",
  "text": "Text to convert to speech",
  "voice": "alloy",
  "output_format": "mp3",
  "gen_config": {
    "speed": 1.0
  }
}
```

**Response:** Binary audio data with headers:
- `Content-Type`: Audio MIME type (e.g., `audio/mpeg`)
- `X-Sample-Rate`: Sample rate in Hz

#### POST `/audio/speak_stream`

Convert text to speech with streaming response.

**Headers:** Same as `/audio/speak`

**Request Body:** Same as `/audio/speak`

**Response:** Streaming binary audio data with same headers

#### POST `/audio/transcribe`

Transcribe audio to text.

**Headers:** Same as other audio endpoints

**Form Data:**
- `model` (required): Model name (e.g., `whisper-1`)
- `file` (required): Audio file to transcribe
- `language` (optional): Language code (e.g., `en`)
- `gen_config` (optional): JSON string with additional config

**Response:**
```json
{
  "text": "Transcribed text",
  "language": "en",
  "usage": {
    "input_tokens": 0,
    "output_tokens": 10
  }
}
```

---

## Streaming Endpoints

### How Streaming Works

Streaming endpoints (`/agent/run_stream` and `/audio/speak_stream`) return data incrementally as it becomes available from the provider. This provides a better user experience for long-running operations.

### Agent Streaming Format

Agent streaming uses **newline-delimited JSON (NDJSON)**. Each line is a complete JSON object:

```
{"output": "Hello", "usage": {}}\n
{"output": " world", "usage": {}}\n
{"output": "!", "usage": {}}\n
{"output": "", "usage": {"input_tokens": 5, "output_tokens": 3}}\n
```

- Each chunk contains an `output` field with incremental text
- The final chunk typically has empty `output` and contains `usage` statistics
- Lines are separated by `\n` (newline)

### Audio Streaming Format

Audio streaming returns **raw binary audio chunks**:

- Content is streamed as it's generated
- `Content-Type` header indicates format (e.g., `audio/mpeg`)
- `X-Sample-Rate` header provides sample rate
- Write chunks directly to a file or audio player

---

## Client Implementation Guide

### Critical: Error Handling in Streaming Requests

**The most common mistake** when implementing streaming clients is incorrect error handling. Here's why:

When you make a streaming request, the HTTP connection opens immediately, but the response body is read incrementally. If an error occurs:

1. The response status code is available immediately
2. The response body is **NOT** available until you read it
3. Attempting to access `response.text` or `response.content` directly will fail

### ✅ Correct Way to Handle Streaming Errors

#### Using httpx (Python)

```python
async with httpx.AsyncClient() as client:
    async with client.stream("POST", url, headers=headers, json=payload) as response:
        # Check status BEFORE consuming the stream
        if response.status_code != 200:
            # Read the error body
            error_body = await response.aread()
            print(f"Error: {response.status_code}")
            try:
                error_data = json.loads(error_body.decode('utf-8'))
                print(f"Detail: {error_data.get('detail')}")
            except:
                print(f"Response: {error_body.decode('utf-8')}")
            return
        
        # Now safe to consume the stream
        async for chunk in response.aiter_bytes():
            process(chunk)
```

#### Using requests (Python)

```python
import requests

response = requests.post(url, headers=headers, json=payload, stream=True)

# Check status before consuming
if response.status_code != 200:
    # Read the error body
    error_body = response.content
    print(f"Error: {response.status_code}")
    try:
        error_data = response.json()
        print(f"Detail: {error_data.get('detail')}")
    except:
        print(f"Response: {error_body.decode('utf-8')}")
    return

# Now safe to iterate
for chunk in response.iter_content(chunk_size=8192):
    process(chunk)
```

#### Using fetch (JavaScript)

```javascript
const response = await fetch(url, {
  method: 'POST',
  headers: headers,
  body: JSON.stringify(payload)
});

// Check status before consuming stream
if (!response.ok) {
  const errorText = await response.text();
  try {
    const errorData = JSON.parse(errorText);
    console.error(`Error ${response.status}: ${errorData.detail}`);
  } catch {
    console.error(`Error ${response.status}: ${errorText}`);
  }
  return;
}

// Now safe to consume stream
const reader = response.body.getReader();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  process(value);
}
```

### ❌ Wrong Way (Will Crash on Errors)

```python
# DON'T DO THIS - will crash if there's an error
async with client.stream("POST", url, ...) as response:
    response.raise_for_status()  # Can't access error body after this
    async for chunk in response.aiter_bytes():
        process(chunk)

# DON'T DO THIS - will crash on streaming responses
try:
    # ...make request...
except httpx.HTTPStatusError as e:
    print(e.response.text)  # Crashes: response not read yet!
```

### Non-Streaming Endpoints

Non-streaming endpoints (`/agent/run`, `/audio/speak`, `/audio/transcribe`) work normally:

```python
response = await client.post(url, headers=headers, json=payload)
response.raise_for_status()  # This works fine for non-streaming
data = response.json()
```

---

## Error Handling

### HTTP Status Codes

- `200 OK`: Success
- `400 Bad Request`: Invalid provider, malformed request, or validation error
- `500 Internal Server Error`: Server error or provider API error

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

1. **Invalid Provider**
   ```json
   {"detail": "Invalid provider: invalid_name"}
   ```
   **Solution**: Use one of the supported provider names

2. **Authentication Failed**
   ```json
   {"detail": "Authentication failed with provider"}
   ```
   **Solution**: Check your API key in `X-Api-Key` header

3. **Provider API Error**
   ```json
   {"detail": "Provider error details..."}
   ```
   **Solution**: Check provider-specific error message

### Best Practices

1. **Always check status code** before consuming streaming responses
2. **Handle errors gracefully** with try-catch blocks
3. **Read error body** using appropriate method for streaming vs non-streaming
4. **Log errors** for debugging
5. **Retry with exponential backoff** for transient errors

---

## Supported Providers

### Agent Providers

| Provider | Value | Models | Base URL |
|----------|-------|--------|----------|
| OpenAI | `openai` | GPT-4, GPT-3.5, etc. | `https://api.openai.com/v1` |
| Anthropic | `anthropic` | Claude 3.5, etc. | `https://api.anthropic.com` |
| Google | `google` | Gemini, etc. | `https://generativelanguage.googleapis.com` |
| Groq | `groq` | Llama, Mixtral, etc. | `https://api.groq.com/openai/v1` |

### Audio Providers

| Provider | Value | TTS Models | STT Models |
|----------|-------|------------|------------|
| OpenAI | `openai` | tts-1, tts-1-hd | whisper-1 |
| ElevenLabs | `elevenlabs` | eleven_turbo_v2_5, etc. | Custom models |

### Custom Base URLs

You can use custom base URLs for:
- **OpenAI-compatible APIs** (Azure OpenAI, OpenRouter, etc.)
- **Local LLM servers** (Ollama, vLLM, LiteLLM, etc.)
- **Proxy servers**

Example:
```bash
curl -X POST http://localhost:8000/agent/run \
  -H "X-Api-Key: your-key" \
  -H "X-Provider: openai" \
  -H "X-Base-Url: http://localhost:11434/v1" \
  -d '{"model": "llama2", "messages": [...]}'
```

---

## Examples

### Complete Python Client Example

See `test.py` for a complete working example that demonstrates:
- Agent streaming and non-streaming
- Audio streaming with OpenAI and ElevenLabs
- Proper error handling for streaming responses
- Health checks and timeout handling

Run the tests:
```bash
# Set your API keys
export OPENAI_API_KEY=your-key
export ELEVENLABS_API_KEY=your-key

# Run tests
python test.py
```

### JavaScript/TypeScript Example

```typescript
import fetch from 'node-fetch';

async function testAgentStreaming() {
  const response = await fetch('http://localhost:8000/agent/run_stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Api-Key': 'your-api-key',
      'X-Provider': 'openai'
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [{role: 'user', content: 'Hello!'}],
      tools: []
    })
  });

  // Check for errors
  if (!response.ok) {
    const error = await response.text();
    console.error(`Error: ${response.status} - ${error}`);
    return;
  }

  // Read NDJSON stream
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const {done, value} = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, {stream: true});
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.trim()) {
        const data = JSON.parse(line);
        if (data.output) {
          process.stdout.write(data.output);
        }
        if (data.usage) {
          console.log('\nUsage:', data.usage);
        }
      }
    }
  }
}
```

### cURL Examples

See `API_EXAMPLES.md` for comprehensive cURL examples for all endpoints.

---

## Development

### Running Tests

```bash
# Start the server
python main.py

# In another terminal, run tests
python test.py
```

### Project Structure

```
qalby-proxy/
├── main.py              # FastAPI application entry point
├── routers/             # API route handlers
│   ├── agent.py         # Agent endpoints
│   └── audio.py         # Audio endpoints
├── managers/            # Business logic managers
│   ├── agent.py         # Agent provider management
│   └── audio.py         # Audio provider management
├── audio_ai/            # Audio provider implementations
│   ├── base.py          # Base audio provider interface
│   ├── openai.py        # OpenAI audio implementation
│   └── elevenlabs.py    # ElevenLabs implementation
├── models/              # Pydantic models
│   ├── common.py        # Common models (Settings, etc.)
│   ├── agent/           # Agent request/response models
│   └── audio/           # Audio request/response models
├── test.py              # Test client with examples
├── README.md            # This file
└── API_EXAMPLES.md      # Detailed API examples

```

### Monitoring

The service includes built-in monitoring with Logfire and OpenTelemetry:

```bash
export LOGFIRE_WRITE_TOKEN=your-token
export OTEL_EXPORTER_OTLP_ENDPOINT=your-endpoint
python main.py
```

---

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]

## Support

For issues, questions, or contributions, please [open an issue](your-repo-url/issues).

