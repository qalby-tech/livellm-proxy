from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Annotated
import asyncio
import json
import logfire
from managers.transcription_rt import TranscriptionRTManager
from models.audio.transcription_ws import (
    TranscriptionInitWsRequest,
    TranscriptionAudioChunkWsRequest,
    TranscriptionWsResponse
)
import traceback


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
           "gen_config": {}
       }
    3. After initialization, bidirectional streaming begins:
       - Client sends audio chunks: {"audio": "base64_encoded_pcm_audio"}
       - Server sends transcriptions: {"transcription": "text", "is_end": false/true}
    4. Either side can close the connection
    
    Audio format:
    - Format: PCM (raw audio samples)
    - Encoding: base64
    - Sample rate: As specified in init (default 24000)
    - Channels: 1 (mono)
    - Sample width: 16-bit
    """
    await websocket.accept()
    logfire.info("WebSocket connection accepted for transcription")
    
    service = None
    audio_sink = None
    receive_task = None
    provider_task = None
    send_task = None
    
    try:
        # Step 1: Receive initialization parameters
        init_data = await websocket.receive_json()
        logfire.info(f"Received initialization data: {init_data}")
        
        try:
            init_request = TranscriptionInitWsRequest.model_validate(init_data)
        except Exception as e:
            error_msg = f"Invalid initialization request: {e}"
            logfire.error(error_msg)
            await websocket.send_json({
                "error": error_msg,
                "type": "initialization_error"
            })
            await websocket.close(code=1003, reason=error_msg)
            return
        
        # Step 2: Create the transcription service
        try:
            service = transcription_manager.create_service(init_request)
            logfire.info(f"Created transcription service for provider: {init_request.provider_uid}")
        except Exception as e:
            error_msg = f"Failed to create transcription service: {e}"
            logfire.error(error_msg, exc_info=True)
            await websocket.send_json({
                "error": error_msg,
                "type": "service_creation_error"
            })
            await websocket.close(code=1011, reason=error_msg)
            return
        
        # Step 3: Connect to the provider's WebSocket
        try:
            await service.connect()
            logfire.info("Connected to transcription provider")
        except Exception as e:
            error_msg = f"Failed to connect to transcription provider: {e}"
            await websocket.send_json({
                "error": error_msg,
                "type": "connection_error"
            })
            await websocket.close(code=1011, reason=error_msg)
            return
        
        # Send success message
        await websocket.send_json({
            "status": "connected",
            "type": "initialization_success"
        })
        
        # Step 4: Set up bidirectional streaming
        audio_sink = asyncio.Queue()
        
        async def receive_from_client():
            """Receive audio chunks from client and forward to service"""
            try:
                async def audio_generator():
                    while True:
                        data = await websocket.receive_json()
                        if "audio" in data:
                            chunk = TranscriptionAudioChunkWsRequest.model_validate(data)
                            yield chunk
                        elif data.get("type") == "close":
                            logfire.info("Client requested close")
                            break
                        else:
                            logfire.warning(f"Received unknown message from client: {data}")
                
                await service.send_audio_chunk(audio_generator())
            except WebSocketDisconnect:
                logfire.info("Client disconnected during audio streaming")
            except Exception as e:
                logfire.error(f"Error receiving from client: {e}", exc_info=True)
                raise
        
        async def receive_from_provider():
            """Receive transcriptions from provider and put in queue"""
            try:
                await service.receive_audio_chunk(audio_sink)
            except Exception as e:
                logfire.error(f"Error receiving from provider: {e}", exc_info=True)
                raise
        
        async def send_to_client():
            """Read from queue and send transcriptions to client"""
            try:
                while True:
                    try:
                        transcription = await asyncio.wait_for(audio_sink.get(), timeout=0.1)
                        await websocket.send_json(transcription.model_dump())
                        
                        if transcription.is_end:
                            logfire.info("Transcription completed")
                    except asyncio.TimeoutError:
                        # Check if connection is still alive
                        if websocket.client_state.name != "CONNECTED":
                            break
                        continue
            except Exception as e:
                logfire.error(f"Error sending to client: {e}", exc_info=True)
                raise
        
        # Run all three tasks concurrently
        receive_task = asyncio.create_task(receive_from_client())
        provider_task = asyncio.create_task(receive_from_provider())
        send_task = asyncio.create_task(send_to_client())
        
        # Wait for any task to complete or fail
        done, pending = await asyncio.wait(
            [receive_task, provider_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Check if any task raised an exception
        for task in done:
            if task.exception():
                raise task.exception()
        
    except WebSocketDisconnect:
        logfire.info("WebSocket disconnected")
    except Exception as e:
        logfire.error(f"Error in transcription WebSocket: {e}", exc_info=True)
        print(traceback.format_exc())
        try:
            await websocket.send_json({
                "error": str(e),
                "type": "runtime_error"
            })
        except:
            pass
    finally:
        # Cleanup all tasks
        for task_name, task in [("receive", receive_task), ("provider", provider_task), ("send", send_task)]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logfire.debug(f"{task_name} task cancelled")
                except Exception as e:
                    logfire.error(f"Error cancelling {task_name} task: {e}")
        
        if service:
            try:
                await service.disconnect()
                logfire.info("Disconnected from transcription provider")
            except Exception as e:
                logfire.error(f"Error disconnecting service: {e}", exc_info=True)
        
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.close()
        except:
            pass
        
        logfire.info("WebSocket connection closed and cleaned up")

