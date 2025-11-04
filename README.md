# Qalby Proxy

A unified FastAPI proxy server for AI services (LLMs and Audio) with built-in fallback support.

## Features

- 🤖 **Multi-Provider Agent Support**: OpenAI, Anthropic, Google, Groq
- 🔊 **Audio Services**: Text-to-Speech and Speech-to-Text
- 🔄 **Intelligent Fallback**: Automatic failover between providers
- 🛠️ **MCP Tools**: Support for web search and MCP streamable servers
- 📊 **Observability**: Built-in logging with Logfire
- 🚀 **Streaming Support**: Both streaming and non-streaming responses

## Installation

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the server
python main.py

# Or with custom host/port
HOST=0.0.0.0 PORT=8000 python main.py
```

## API Endpoints

### Provider Configuration Management

#### Get All Provider Configurations
```http
GET /providers/configs
```

Returns a list of all provider configurations with masked API keys.

#### Add Provider Configuration
```http
POST /providers/config
Content-Type: application/json

{
  "uid": "openai-1",
  "provider": "openai",
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1"  // optional
}
```

#### Delete Provider Configuration
```http
DELETE /providers/config/{uid}
```

### Agent Endpoints

#### Run Agent (Non-Streaming)
```http
POST /agent/run
Content-Type: application/json
```

**Single Request:**
```json
{
  "provider_uid": "openai-1",
  "model": "gpt-4",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "tools": [],
  "gen_config": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

**Fallback Request (Sequential):**
```json
{
  "requests": [
    {
      "provider_uid": "openai-1",
      "model": "gpt-4",
      "messages": [...]
    },
    {
      "provider_uid": "anthropic-1",
      "model": "claude-3-opus-20240229",
      "messages": [...]
    }
  ],
  "strategy": "sequential",
  "timeout_per_request": 360
}
```

**Fallback Request (Parallel):**
```json
{
  "requests": [...],
  "strategy": "parallel",
  "timeout_per_request": 360
}
```

#### Run Agent (Streaming)
```http
POST /agent/run_stream
Content-Type: application/json
```

Accepts the same payload formats as `/agent/run` but returns NDJSON stream.

### Audio Endpoints

#### Text-to-Speech (Non-Streaming)
```http
POST /audio/speak
Content-Type: application/json
```

**Single Request:**
```json
{
  "provider_uid": "openai-1",
  "model": "tts-1",
  "text": "Hello world",
  "voice": "alloy",
  "output_format": "mp3"
}
```

**Fallback Request:**
```json
{
  "requests": [
    {
      "provider_uid": "openai-1",
      "model": "tts-1",
      "text": "Hello world",
      "voice": "alloy"
    },
    {
      "provider_uid": "elevenlabs-1",
      "model": "eleven_multilingual_v2",
      "text": "Hello world",
      "voice": "Rachel"
    }
  ],
  "strategy": "sequential",
  "timeout_per_request": 360
}
```

#### Text-to-Speech (Streaming)
```http
POST /audio/speak_stream
Content-Type: application/json
```

Accepts the same payload formats as `/audio/speak` but returns streaming audio.

#### Speech-to-Text (Form-based)
```http
POST /audio/transcribe
Content-Type: multipart/form-data

provider_uid: openai-1
model: whisper-1
file: <audio file>
language: en (optional)
gen_config: {"temperature": 0} (optional)
```

**Note:** This endpoint only supports single requests, not fallback.

#### Speech-to-Text (JSON-based)
```http
POST /audio/transcribe_json
Content-Type: application/json
```

**Single Request:**
```json
{
  "provider_uid": "openai-1",
  "model": "whisper-1",
  "file": [
    "audio.mp3",
    "<base64-encoded-audio>",
    "audio/mpeg"
  ],
  "language": "en"
}
```

**Fallback Request:**
```json
{
  "requests": [
    {
      "provider_uid": "openai-1",
      "model": "whisper-1",
      "file": ["audio.mp3", "<base64>", "audio/mpeg"]
    },
    {
      "provider_uid": "groq-1",
      "model": "whisper-large-v3",
      "file": ["audio.mp3", "<base64>", "audio/mpeg"]
    }
  ],
  "strategy": "sequential",
  "timeout_per_request": 360
}
```

## Fallback Strategies

### Sequential
Tries each provider one by one until one succeeds.

**Use case:** When you have a preferred provider but want guaranteed backup.

```json
{
  "strategy": "sequential",
  "requests": [
    { "provider_uid": "preferred-provider", ... },
    { "provider_uid": "backup-provider", ... }
  ]
}
```

### Parallel
Tries all providers simultaneously and returns the first successful response.

**Use case:** When you need the fastest response and don't care which provider serves it.

```json
{
  "strategy": "parallel",
  "requests": [
    { "provider_uid": "provider-1", ... },
    { "provider_uid": "provider-2", ... },
    { "provider_uid": "provider-3", ... }
  ]
}
```

## Supported Providers

### Agent Providers
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude 3 family)
- Google (Gemini family)
- Groq (Mixtral, Llama, etc.)

### Audio Providers
- OpenAI (TTS + Whisper)
- ElevenLabs (TTS)

## Tools Support

### Web Search Tool
```json
{
  "kind": "web_search",
  "search_context_size": 5,
  "kwargs": {}
}
```

### MCP Streamable Server
```json
{
  "kind": "mcp_streamable",
  "url": "https://your-mcp-server.com",
  "prefix": "tool_",
  "timeout": 30,
  "kwargs": {}
}
```

## Message Types

### Text Message
```json
{
  "role": "user",  // or "model" or "system"
  "content": "Your message here"
}
```

### Binary Message (Images, etc.)
```json
{
  "role": "user",
  "content": "<base64-encoded-content>",
  "mime_type": "image/jpeg",
  "caption": "Optional caption"
}
```

## Environment Variables

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000

# Observability (optional)
LOGFIRE_WRITE_TOKEN=your-token
OTEL_EXPORTER_OTLP_ENDPOINT=your-endpoint
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              FastAPI App                     │
├─────────────────────────────────────────────┤
│  Routers                                     │
│  ├── Providers Router (/providers/*)        │
│  ├── Agent Router (/agent/*)                │
│  └── Audio Router (/audio/*)                │
├─────────────────────────────────────────────┤
│  Managers                                    │
│  ├── ConfigManager (Provider configs)       │
│  ├── FallbackManager (Failover logic)       │
│  ├── AgentManager (LLM operations)          │
│  └── AudioManager (TTS/STT operations)      │
├─────────────────────────────────────────────┤
│  AI Services                                 │
│  ├── OpenAI (Agent + Audio)                 │
│  ├── Anthropic (Agent)                      │
│  ├── Google (Agent)                         │
│  ├── Groq (Agent)                           │
│  └── ElevenLabs (Audio)                     │
└─────────────────────────────────────────────┘
```

## Key Features Explained

### Fallback Manager
The fallback manager provides intelligent failover:

- **Automatic retry**: If a provider fails, automatically tries the next one
- **Configurable timeout**: Set per-request timeouts
- **Two strategies**: Sequential (one-by-one) or Parallel (race)
- **Transparent**: Works with all endpoints (streaming and non-streaming)

### Safe Methods
All managers expose "safe" methods that accept both single requests and fallback requests:

- `AgentManager.safe_run()` - Agent with fallback
- `AudioManager.safe_speak()` - TTS with fallback
- `AudioManager.safe_transcribe()` - STT with fallback

### Provider Caching
ConfigManager caches provider clients for performance:

- Providers are initialized once per UID
- Automatic reuse across requests
- Thread-safe operations

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200` - Success
- `400` - Bad request (validation errors)
- `500` - Internal server error

Error responses include detailed messages:
```json
{
  "detail": "Error description here"
}
```

## Development

### Running Tests
```bash
pytest
```

### Linting
```bash
ruff check .
```

### Docker Deployment
```bash
docker build -t qalby-proxy .
docker run -p 8000:8000 qalby-proxy
```

### Kubernetes Deployment
Helm chart is available in the `chart/` directory:

```bash
helm install qalby-proxy ./chart
```

## License

[Your License Here]

## Contributing

[Your Contributing Guidelines Here]

