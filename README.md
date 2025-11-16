# Qalby Proxy

A unified FastAPI proxy server for AI services (LLMs and Audio) with built-in fallback support.

## Features

- 🤖 **Multi-Provider Agent Support**: OpenAI, Anthropic, Google, Groq
- 🔊 **Audio Services**: Text-to-Speech and Speech-to-Text
- 🎙️ **Real-Time Transcription**: WebSocket-based live audio transcription with bidirectional streaming
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

**Note:** All endpoints are prefixed with `/livellm`. For example: `/livellm/providers/configs`

### WebSocket Endpoints

#### Generic WebSocket Connection
```
WS /livellm/ws
```

The WebSocket endpoint provides a unified interface for all AI operations (agent and audio) with support for both streaming and non-streaming responses.

#### Real-Time Transcription WebSocket
```
WS /livellm/ws/transcription
```

A dedicated WebSocket endpoint for real-time audio transcription with bidirectional streaming. This endpoint provides low-latency speech-to-text conversion for live audio streams.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/livellm/ws');
```

**Request Format:**
```json
{
  "action": "agent_run",
  "payload": {
    // Action-specific payload
  }
}
```

**Response Format:**
```json
{
  "status": "success",  // or "streaming" or "error"
  "action": "agent_run",
  "data": {
    // Response data
  },
  "error": null  // Error message if status is "error"
}
```

**Available Actions:**

1. **`agent_run`** - Run agent (non-streaming)
   ```json
   {
     "action": "agent_run",
     "payload": {
       "provider_uid": "openai-1",
       "model": "gpt-4",
       "messages": [{"role": "user", "content": "Hello!"}],
       "tools": []
     }
   }
   ```

2. **`agent_run_stream`** - Run agent (streaming)
   - Same payload as `agent_run`
   - Returns multiple messages with `status: "streaming"` followed by final `status: "success"`

3. **`audio_speak`** - Text-to-Speech (non-streaming)
   ```json
   {
     "action": "audio_speak",
     "payload": {
       "provider_uid": "openai-1",
       "model": "tts-1",
       "text": "Hello world",
       "voice": "alloy",
       "mime_type": "audio/mpeg",
       "sample_rate": 24000
     }
   }
   ```
   - Returns base64-encoded audio in `data.audio`

4. **`audio_speak_stream`** - Text-to-Speech (streaming)
   - Same payload as `audio_speak`
   - Returns multiple chunks with base64-encoded audio

5. **`audio_transcribe`** - Speech-to-Text
   ```json
   {
     "action": "audio_transcribe",
     "payload": {
       "provider_uid": "openai-1",
       "model": "whisper-1",
       "file": ["audio.mp3", "<base64-audio>", "audio/mpeg"],
       "language": "en"
     }
   }
   ```

**Fallback Support:**

All actions support fallback requests through the WebSocket:
```json
{
  "action": "agent_run",
  "payload": {
    "requests": [
      {"provider_uid": "openai-1", "model": "gpt-4", "messages": [...]},
      {"provider_uid": "anthropic-1", "model": "claude-3-opus", "messages": [...]}
    ],
    "strategy": "sequential",
    "timeout_per_request": 360
  }
}
```

**Example Client:**

See `websocket_client_example.html` for a complete working example.

---

### Real-Time Transcription WebSocket

The real-time transcription WebSocket (`/livellm/ws/transcription`) provides bidirectional streaming for live audio transcription. This is ideal for voice assistants, live captioning, and real-time speech recognition applications.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/livellm/ws/transcription');
```

**Protocol Flow:**

1. **Client connects** to the WebSocket
2. **Client sends initialization message** with transcription configuration
3. **Bidirectional streaming begins:**
   - Client continuously sends audio chunks
   - Server continuously sends transcription results
4. **Either side can close** the connection when done

**Step 1: Initialization Message**

After connecting, the client must send an initialization message:

```json
{
  "provider_uid": "openai-1",
  "model": "gpt-4o-mini-transcribe",
  "language": "en",
  "input_sample_rate": 24000,
  "input_audio_format": "audio/pcm",
  "gen_config": {}
}
```

**Initialization Parameters:**
- `provider_uid` (required): Provider configuration UID
- `model` (required): Transcription model (e.g., "gpt-4o-mini-transcribe")
- `language` (optional): Language code (default: "auto" for auto-detection)
- `input_sample_rate` (optional): Audio sample rate in Hz (default: 24000)
- `input_audio_format` (optional): Audio format - "audio/pcm", "audio/ulaw", or "audio/alaw" (default: "audio/pcm")
- `gen_config` (optional): Additional model-specific configuration

**Server Response to Initialization:**
```json
{
  "status": "success",
  "action": "transcription_session",
  "data": {},
  "error": null
}
```

If initialization fails:
```json
{
  "status": "error",
  "action": "transcription_session",
  "data": {},
  "error": "Error description"
}
```

**Step 2: Streaming Audio Chunks**

After successful initialization, send audio chunks continuously:

```json
{
  "audio": "base64_encoded_audio_data"
}
```

**Audio Format Requirements:**
- **Encoding**: Base64-encoded audio bytes
- **Format**: PCM, μ-law, or A-law (as specified in initialization)
- **Sample Rate**: As specified in initialization (e.g., 24000 Hz)
- **Channels**: 1 (mono)
- **Sample Width**: 
  - 16-bit for PCM
  - 8-bit for μ-law/A-law

**Note:** Audio is automatically converted to PCM16 at 24kHz before being sent to the transcription service.

**Step 3: Receiving Transcriptions**

The server sends transcription results as they become available:

```json
{
  "transcription": "Hello, how are you?",
  "is_end": false
}
```

**Response Fields:**
- `transcription`: The transcribed text
- `is_end`: Boolean indicating if this is the final transcription chunk

**Example Client Implementation:**

```javascript
// Connect to transcription WebSocket
const ws = new WebSocket('ws://localhost:8000/livellm/ws/transcription');

ws.onopen = () => {
  console.log('Connected to transcription WebSocket');
  
  // Send initialization message
  ws.send(JSON.stringify({
    provider_uid: 'openai-1',
    model: 'gpt-4o-mini-transcribe',
    language: 'en',
    input_sample_rate: 24000,
    input_audio_format: 'audio/pcm',
    gen_config: {}
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  
  if (response.action === 'transcription_session') {
    if (response.status === 'success') {
      console.log('Transcription session initialized');
      startSendingAudio(ws);
    } else {
      console.error('Initialization failed:', response.error);
    }
  } else {
    // Transcription result
    console.log('Transcription:', response.transcription);
    if (response.is_end) {
      console.log('Transcription complete');
    }
  }
};

function startSendingAudio(ws) {
  // Example: Get audio from microphone
  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(stream => {
      const mediaRecorder = new MediaRecorder(stream);
      
      mediaRecorder.ondataavailable = async (event) => {
        const audioBlob = event.data;
        const arrayBuffer = await audioBlob.arrayBuffer();
        const base64Audio = btoa(
          String.fromCharCode(...new Uint8Array(arrayBuffer))
        );
        
        // Send audio chunk
        ws.send(JSON.stringify({
          audio: base64Audio
        }));
      };
      
      // Send audio chunks every 100ms
      mediaRecorder.start(100);
    });
}

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket connection closed');
};
```

**Python Client Example:**

```python
import asyncio
import websockets
import json
import base64

async def transcribe_audio():
    uri = "ws://localhost:8000/livellm/ws/transcription"
    
    async with websockets.connect(uri) as websocket:
        # Send initialization
        init_message = {
            "provider_uid": "openai-1",
            "model": "gpt-4o-mini-transcribe",
            "language": "en",
            "input_sample_rate": 24000,
            "input_audio_format": "audio/pcm",
            "gen_config": {}
        }
        await websocket.send(json.dumps(init_message))
        
        # Wait for initialization response
        response = json.loads(await websocket.recv())
        if response["status"] != "success":
            print(f"Initialization failed: {response['error']}")
            return
        
        print("Transcription session initialized")
        
        # Start sending audio and receiving transcriptions
        async def send_audio():
            with open("audio.pcm", "rb") as f:
                while chunk := f.read(4096):
                    audio_b64 = base64.b64encode(chunk).decode('utf-8')
                    await websocket.send(json.dumps({"audio": audio_b64}))
                    await asyncio.sleep(0.1)  # Simulate real-time streaming
        
        async def receive_transcriptions():
            while True:
                response = json.loads(await websocket.recv())
                print(f"Transcription: {response['transcription']}")
                if response["is_end"]:
                    break
        
        # Run both tasks concurrently
        await asyncio.gather(
            send_audio(),
            receive_transcriptions()
        )

asyncio.run(transcribe_audio())
```

**Supported Audio Formats:**

| Format | MIME Type | Sample Width | Common Use Case |
|--------|-----------|--------------|-----------------|
| PCM | `audio/pcm` | 16-bit | High quality, uncompressed |
| μ-law | `audio/ulaw` | 8-bit | Telephony (North America/Japan) |
| A-law | `audio/alaw` | 8-bit | Telephony (Europe/rest of world) |

**Error Handling:**

If an error occurs during transcription, the WebSocket will be closed with an appropriate error code and reason. Always implement proper error handling and reconnection logic in production applications.

### Provider Configuration Management

#### Get All Provider Configurations
```http
GET /livellm/providers/configs
```

Returns a list of all provider configurations with masked API keys.

#### Add Provider Configuration
```http
POST /livellm/providers/config
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
DELETE /livellm/providers/config/{uid}
```

### Agent Endpoints

#### Run Agent (Non-Streaming)
```http
POST /livellm/agent/run
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
POST /livellm/agent/run_stream
Content-Type: application/json
```

Accepts the same payload formats as `/livellm/agent/run` but returns NDJSON stream.

### Audio Endpoints

#### Text-to-Speech (Non-Streaming)
```http
POST /livellm/audio/speak
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
POST /livellm/audio/speak_stream
Content-Type: application/json
```

Accepts the same payload formats as `/livellm/audio/speak` but returns streaming audio.

#### Speech-to-Text (Form-based)
```http
POST /livellm/audio/transcribe
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
POST /livellm/audio/transcribe_json
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
┌──────────────────────────────────────────────────────┐
│             FastAPI App (/livellm)                   │
├──────────────────────────────────────────────────────┤
│  Routers                                              │
│  ├── WebSocket Router (/livellm/ws)                  │
│  ├── Transcription WS Router (/livellm/ws/transcription)│
│  ├── Providers Router (/livellm/providers/*)         │
│  ├── Agent Router (/livellm/agent/*)                 │
│  └── Audio Router (/livellm/audio/*)                 │
├──────────────────────────────────────────────────────┤
│  Managers                                             │
│  ├── WsManager (WebSocket operations)                │
│  ├── TranscriptionRTManager (Real-time transcription)│
│  ├── ConfigManager (Provider configs)                │
│  ├── FallbackManager (Failover logic)                │
│  ├── AgentManager (LLM operations)                   │
│  └── AudioManager (TTS/STT operations)               │
├──────────────────────────────────────────────────────┤
│  AI Services                                          │
│  ├── OpenAI (Agent + Audio + RT Transcription)       │
│  ├── Anthropic (Agent)                               │
│  ├── Google (Agent)                                  │
│  ├── Groq (Agent)                                    │
│  └── ElevenLabs (Audio)                              │
└──────────────────────────────────────────────────────┘
```

## Key Features Explained

### Real-Time Transcription
The real-time transcription feature provides low-latency speech-to-text conversion through a dedicated WebSocket endpoint:

- **Bidirectional streaming**: Simultaneous audio upload and transcription download
- **Multiple audio formats**: Support for PCM, μ-law, and A-law audio formats
- **Automatic conversion**: Audio is automatically converted to the optimal format for the transcription service
- **Flexible sample rates**: Configure input sample rate to match your audio source
- **Language support**: Auto-detection or specify language for better accuracy
- **Live results**: Receive transcription results as they become available

**Use cases:**
- Voice assistants and chatbots
- Live captioning and subtitles
- Real-time meeting transcription
- Voice commands and control
- Accessibility features

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

