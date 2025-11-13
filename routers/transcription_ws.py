from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Annotated
import asyncio
import logfire
from managers.transcription_rt import TranscriptionRTManager
from models.audio.transcription_ws import (
    TranscriptionInitWsRequest,
    TranscriptionAudioChunkWsRequest,
    TranscriptionWsResponse
)
from models.ws import WsResponse, WsAction, WsStatus
from typing import AsyncIterator


transcription_ws_router = APIRouter(prefix="/ws/audio", tags=["transcription_ws"])


def get_transcription_rt_manager(websocket: WebSocket) -> TranscriptionRTManager:
    return websocket.app.state.transcription_rt_manager


TranscriptionRTManagerDep = Annotated[TranscriptionRTManager, Depends(get_transcription_rt_manager)]


@transcription_ws_router.websocket("/transcription")
async def transcription_websocket_endpoint(
    websocket: WebSocket, 
    transcription_manager: TranscriptionRTManagerDep
):
    """
    WebSocket endpoint for real-time audio transcription.
    
    Protocol:
    1. Client connects to WebSocket
    2. Client sends initialization message with TranscriptionInitWsRequest format:
       {
           "provider_uid": "openai",
           "model": "gpt-4o-mini-transcribe",
           "language": "en",
           "input_sample_rate": 24000,
           "input_audio_format": "audio/pcm",  # or "audio/ulaw" or "audio/alaw"
           "gen_config": {}
       }
    3. After initialization, bidirectional streaming begins:
       - Client sends audio chunks: {"audio": "base64_encoded_audio"}
       - Server sends transcriptions: {"transcription": "text", "is_end": false/true}
    4. Either side can close the connection
    
    Audio format:
    - Format: PCM, ulaw, or alaw (specified in init, default: PCM)
    - Encoding: base64
    - Sample rate: As specified in init (default 24000)
    - Channels: 1 (mono)
    - Sample width: 16-bit for PCM, 8-bit for ulaw/alaw
    
    Note: Audio is automatically converted to PCM16 at 24kHz before being sent to OpenAI.
    """
    await websocket.accept()
    logfire.info("WebSocket connection accepted for transcription")
    init_data: dict = await websocket.receive_json()
    try:
        init_request = TranscriptionInitWsRequest.model_validate(init_data)
    except Exception as e:
        logfire.error(f"Invalid initialization request: {e}")
        await websocket.send_json(WsResponse(
            status=WsStatus.ERROR,
            action=WsAction.TRANSCRIPTION_SESSION,
            data={},
            error=str(e)
        ).model_dump())
        await websocket.close(code=1011, reason=str(e))
        return
    
    service = transcription_manager.create_service(init_request)
    await service.connect()


    async def audio_source() -> AsyncIterator[bytes]:
        """Generator that yields decoded audio bytes from client messages"""
        try:
            while True:
                data: dict = await websocket.receive_json()
                if "audio" in data:
                    chunk: TranscriptionAudioChunkWsRequest = TranscriptionAudioChunkWsRequest.model_validate(data)
                    yield chunk.audio
                elif data.get("type") == "close":
                    logfire.info("Client requested close")
                    break
                else:
                    logfire.warning(f"Received unknown message from client: {data}")
        except WebSocketDisconnect:
            logfire.info("Client disconnected during audio source")
        except Exception as e:
            logfire.error(f"Error in audio_source: {e}", exc_info=True)
            raise
    
    async def send_transcriptions(queue: asyncio.Queue[TranscriptionWsResponse]):
        """Task that sends transcriptions from queue to client"""
        try:
            while True:
                transcription = await queue.get()
                await websocket.send_json(transcription.model_dump())
                if transcription.is_end:
                    break
        except WebSocketDisconnect:
            logfire.info("Client disconnected during transcription send")
        except Exception as e:
            logfire.error(f"Error sending transcriptions: {e}", exc_info=True)
            raise

    
    try:
        # Create a queue for transcription responses
        transcription_queue: asyncio.Queue[TranscriptionWsResponse] = asyncio.Queue()
        
        # Start the transcription send task
        send_task = asyncio.create_task(send_transcriptions(transcription_queue))
        
        # Start the realtime transcription (this will run until audio source is exhausted)
        await service.realtime_transcribe(
            audio_source=audio_source(),
            audio_sink=transcription_queue,
            input_audio_format=init_request.input_audio_format,
            input_sample_rate=init_request.input_sample_rate
        )
        
        # Wait for all transcriptions to be sent
        await send_task
    except WebSocketDisconnect:
        logfire.info("WebSocket disconnected")
        await service.disconnect()
    except Exception as e:
        logfire.error(f"Error in transcription WebSocket: {e}", exc_info=True)
        await websocket.send_json(WsResponse(
            status=WsStatus.ERROR,
            action=WsAction.TRANSCRIPTION_SESSION,
            data={},
            error=str(e)
        ).model_dump())
        await websocket.close(code=1011, reason=str(e))
        await service.disconnect()
