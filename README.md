# Qalby Proxy

A unified API proxy server that consolidates OpenAI, Google Gemini, and ElevenLabs APIs under a single, consistent HTTP interface. Built with FastAPI.

## Overview

Qalby Proxy provides a standardized way to interact with multiple AI providers through three core endpoints:
- **`/chat`** - Chat completions with support for text, images, and tool calling
- **`/speak`** - Text-to-speech (TTS)
- **`/transcribe`** - Speech-to-text (STT)

All endpoints use a unified authentication system and consistent request/response formats across providers.

## Features

- **Multi-Provider Support**: OpenAI, Google Gemini, and ElevenLabs
- **Unified API**: Consistent interface across different providers
- **Tool Support**: Web search and MCP (Model Context Protocol) tools for Gemini
- **Media Support**: Handle text, images, audio across compatible providers
- **Bearer Authentication**: Single master API key for all requests
- **CORS Enabled**: Ready for browser-based clients
- **Docker Support**: Containerized deployment available

## Provider Capabilities

| Provider | `/chat` | `/speak` | `/transcribe` |
|----------|---------|----------|---------------|
| OpenAI | ✅ | ✅ | ✅ |
| Google Gemini | ✅ | ❌ | ❌ |
| ElevenLabs | ❌ | ✅ | ✅ |

## Quick Start

### Prerequisites

- Python 3.12 or higher
- API keys for the providers you want to use (at least one required)

### Installation

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```ini
# Required: Master API key for authentication
master_api_key=your-secure-master-key

# Optional: Server configuration
host=0.0.0.0
port=8000

# Provider API keys (at least one required)
openai_api_key=sk-...
# openai_base_url=https://api.openai.com/v1

google_api_key=AIza...
# google_base_url=https://generativelanguage.googleapis.com

elevenlabs_api_key=el_...
# elevenlabs_base_url=https://api.elevenlabs.io
```

### Running the Server

**Direct Python:**
```bash
python main.py
```

**Using uvicorn:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Using Docker:**
```bash
docker build -t qalby-proxy .
docker run -p 8000:8000 --env-file .env qalby-proxy
```

Once running, access the interactive API documentation at: http://localhost:8000/docs

## Model Reference Format

All requests use the format `provider/model`:

**Examples:**
- `openai/gpt-4o-mini`
- `openai/gpt-4o`
- `openai/whisper-1`
- `google/gemini-1.5-flash`
- `google/gemini-1.5-pro`
- `elevenlabs/eleven_multilingual_v2`

## API Reference

### Authentication

All endpoints require Bearer token authentication using your `master_api_key`:

```
Authorization: Bearer <your-master-api-key>
```

### POST `/chat`

Chat completions endpoint supporting text and media messages with optional tool calling.

**Request Body:**
```json
{
  "model": "openai/gpt-4o-mini",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is the weather like?"
    }
  ],
  "tools": [
    {
      "web_search": {
        "exclude_domains": ["example.com"]
      }
    }
  ],
  "tool_choice": "auto"
}
```

**Message Types:**

*Text Message:*
```json
{
  "role": "user",
  "content": "Hello!"
}
```

*Media Message (Image):*
```json
{
  "role": "user",
  "content": "<base64-encoded-image-data>",
  "mime_type": "image/png"
}
```

**Tools:**

*Web Search (Google Gemini only):*
```json
{
  "web_search": {
    "exclude_domains": ["reddit.com", "twitter.com"]
  }
}
```

*MCP Tool (Google Gemini only):*
```json
{
  "name": "my-mcp-server",
  "server_url": "http://localhost:7000"
}
```

**Response:**
```json
{
  "text": "The assistant's response text"
}
```

**Supported Roles:**
- `system` - System instructions (converted to provider-specific format)
- `user` - User messages
- `assistant` - Assistant responses

### POST `/speak`

Text-to-speech endpoint. Returns raw audio bytes.

**Request Body:**
```json
{
  "model": "openai/tts-1",
  "text": "Hello, world!",
  "voice": "alloy",
  "response_format": "mp3"
}
```

**Response:**
- Content-Type: `audio/mpeg` (or appropriate MIME type)
- Body: Raw audio bytes

**Supported Audio Formats:**
- `mp3` - MP3 audio
- `opus` - Opus audio
- `aac` - AAC audio
- `flac` - FLAC audio
- `wav` - WAV audio
- `pcm` - PCM audio

**ElevenLabs Formats:**
ElevenLabs uses extended format strings like `mp3_44100_128` (codec_samplerate_bitrate).

### POST `/transcribe`

Speech-to-text endpoint. Accepts audio files via multipart form data.

**Request (multipart/form-data):**
- `file` - Audio file (binary)
- `model` - Model reference (e.g., `openai/whisper-1`)
- `language` - Optional language code (e.g., `en`, `es`)

**Response:**
```json
{
  "text": "Transcribed text from the audio"
}
```

## Usage Examples

### cURL

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer your-master-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [
      {"role": "user", "content": "Tell me a joke"}
    ]
  }'
```

**Speak (save to file):**
```bash
curl -X POST http://localhost:8000/speak \
  -H "Authorization: Bearer your-master-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/tts-1",
    "text": "Hello there",
    "voice": "alloy",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

**Transcribe:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -H "Authorization: Bearer your-master-api-key" \
  -F "model=openai/whisper-1" \
  -F "language=en" \
  -F "file=@audio.wav;type=audio/wav"
```

### Python

```python
import httpx
import base64
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_KEY = "your-master-api-key"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

async def chat_example():
    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS) as client:
        response = await client.post("/chat", json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "user", "content": "What is 2+2?"}
            ]
        })
        print(response.json()["text"])

async def chat_with_image():
    # Read and encode image
    image_path = Path("image.png")
    image_data = base64.b64encode(image_path.read_bytes()).decode()
    
    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS) as client:
        response = await client.post("/chat", json={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": image_data,
                    "mime_type": "image/png"
                },
                {
                    "role": "user",
                    "content": "What's in this image?"
                }
            ]
        })
        print(response.json()["text"])

async def speak_example():
    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=60.0) as client:
        response = await client.post("/speak", json={
            "model": "openai/tts-1",
            "text": "Hello, world!",
            "voice": "alloy",
            "response_format": "mp3"
        })
        Path("output.mp3").write_bytes(response.content)

async def transcribe_example():
    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=120.0) as client:
        with open("audio.wav", "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            data = {"model": "openai/whisper-1", "language": "en"}
            response = await client.post("/transcribe", files=files, data=data)
        print(response.json()["text"])
```

### JavaScript/TypeScript (Node.js)

```javascript
import fetch from "node-fetch";
import fs from "fs/promises";

const BASE_URL = "http://localhost:8000";
const API_KEY = "your-master-api-key";
const HEADERS = { Authorization: `Bearer ${API_KEY}` };

// Chat
async function chatExample() {
  const response = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { ...HEADERS, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "openai/gpt-4o-mini",
      messages: [
        { role: "user", content: "What is 2+2?" }
      ]
    })
  });
  const data = await response.json();
  console.log(data.text);
}

// Speak
async function speakExample() {
  const response = await fetch(`${BASE_URL}/speak`, {
    method: "POST",
    headers: { ...HEADERS, "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "openai/tts-1",
      text: "Hello, world!",
      voice: "alloy",
      response_format: "mp3"
    })
  });
  const buffer = Buffer.from(await response.arrayBuffer());
  await fs.writeFile("output.mp3", buffer);
}

// Transcribe
async function transcribeExample() {
  const FormData = (await import("form-data")).default;
  const formData = new FormData();
  formData.append("model", "openai/whisper-1");
  formData.append("language", "en");
  formData.append("file", await fs.readFile("audio.wav"), {
    filename: "audio.wav",
    contentType: "audio/wav"
  });
  
  const response = await fetch(`${BASE_URL}/transcribe`, {
    method: "POST",
    headers: HEADERS,
    body: formData
  });
  const data = await response.json();
  console.log(data.text);
}
```

## Provider-Specific Details

### OpenAI

- **Chat**: Uses OpenAI's Responses API
- **Speak**: Supports voices: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
- **Transcribe**: Uses Whisper model
- System messages are converted to `developer` role (OpenAI format)

### Google Gemini

- **Chat**: Uses Google GenAI API
- **Tools**: Supports Google Search and MCP (Model Context Protocol) servers
- **Media**: Images must be base64-encoded with `mime_type` specified
- System messages are extracted and used as `system_instruction`
- MCP sessions are cached and reused for efficiency

### ElevenLabs

- **Speak**: TTS with custom voice IDs and high-quality audio
- **Transcribe**: Multilingual speech-to-text
- Format examples: `mp3_44100_128`, `pcm_16000`, `opus_16000_32`

## Error Responses

**401 Unauthorized:**
```json
{
  "detail": "Unauthorized"
}
```

**400 Bad Request - Invalid Model Reference:**
```json
{
  "detail": "Invalid model reference: 'invalid-format'"
}
```

**400 Bad Request - Unknown Provider:**
```json
{
  "detail": "Unknown provider: 'unknown'"
}
```

**400 Bad Request - Unsupported Endpoint:**
```json
{
  "detail": "Agent 'elevenlabs' does not support chat endpoint"
}
```

## Architecture

### Project Structure

```
qalby-proxy/
├── agent/
│   ├── base.py          # Abstract Agent base class
│   ├── openai.py        # OpenAI implementation
│   ├── gemini.py        # Google Gemini implementation
│   └── elevenlabs.py    # ElevenLabs implementation
├── models/
│   ├── inputs.py        # Message and tool schemas
│   ├── requests.py      # Request models
│   ├── responses.py     # Response models
│   └── errors.py        # Custom exceptions
├── main.py              # FastAPI application
├── utils.py             # Utility functions
├── Dockerfile           # Container configuration
└── pyproject.toml       # Project dependencies
```

### Agent Interface

All agents implement the `Agent` abstract base class with three main methods:

- `async chat()` - Chat completions
- `async speak()` - Text-to-speech
- `async transcribe()` - Speech-to-text

Agents raise `NotImplementedError` for unsupported endpoints.

## Development

### Dependencies

Core dependencies (see `pyproject.toml`):
- `fastapi` - Web framework
- `openai` - OpenAI client
- `google-genai` - Google GenAI client
- `elevenlabs` - ElevenLabs client
- `pydantic` - Data validation
- `mcp` - Model Context Protocol support

### Running Tests

```bash
# Install dev dependencies
uv sync --dev

# Run tests
pytest
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues or questions, please [add contact information or issue tracker URL here].
