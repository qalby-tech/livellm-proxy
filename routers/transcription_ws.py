from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, WebSocketException
from typing import Annotated
from managers.transcription_rt import TranscriptionRTManager
from models.audio.transcription_ws import (
    TranscriptionInitWsRequest,
    TranscriptionAudioChunkWsRequest,
    TranscriptionWsResponse
)
from models.ws import WsResponse, WsAction, WsStatus
from typing import AsyncIterator
from starlette.websockets import WebSocketState
import logfire


transcription_ws_router = APIRouter(prefix="/ws", tags=["transcription_ws"])


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
    
    try:
        service = transcription_manager.create_service(init_request)
        await service.connect()
        await websocket.send_json(WsResponse(
            status=WsStatus.SUCCESS,
            action=WsAction.TRANSCRIPTION_SESSION,
            data={},
            error=None
        ).model_dump())
    except Exception as e:
        logfire.error(f"Error creating transcription service: {e}", exc_info=True)
        await websocket.send_json(WsResponse(
            status=WsStatus.ERROR,
            action=WsAction.TRANSCRIPTION_SESSION,
            data={},
            error=str(e)
        ).model_dump())
        await websocket.close(code=1011, reason=str(e))
        return


    async def audio_source() -> AsyncIterator[bytes]:
        """Generator that yields decoded audio bytes from client messages"""
        while websocket.client_state == WebSocketState.CONNECTED:
            data: dict = await websocket.receive_json()
            try:
                chunk: TranscriptionAudioChunkWsRequest = TranscriptionAudioChunkWsRequest.model_validate(data)
                yield chunk.audio
            except Exception as e:
                logfire.error(f"Error validating audio chunk: {e}", exc_info=True)
                continue
    
    async def send_transcription(transcription: TranscriptionWsResponse) -> None:
        """Send a single transcription to client"""
        if websocket.client_state != WebSocketState.CONNECTED:
            return
        try:
            await websocket.send_json(transcription.model_dump())
        except WebSocketException as e:
            if e.code == 1005: 
                # client disconnected
                return
            else:
                raise e

    try:        
        # Start the realtime transcription (this will run until audio source is exhausted)
        await service.realtime_transcribe(
            audio_source=audio_source(),
            audio_sink=send_transcription,
            input_audio_format=init_request.input_audio_format,
            input_sample_rate=init_request.input_sample_rate,
        )
    except WebSocketDisconnect:
        logfire.info("WebSocket disconnected")
    finally:
        await service.disconnect()
        