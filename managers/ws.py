from managers.agent import AgentManager
from managers.audio import AudioManager
from models.ws import WsRequest, WsResponse, WsAction, WsStatus
from models.fallback import AgentFallbackRequest, TranscribeFallbackRequest, AudioFallbackRequest
from models.agent.agent import AgentRequest, AgentResponse
from models.audio.speak import SpeakRequest, EncodedSpeakResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from typing import AsyncIterator, Union
from pydantic import BaseModel, TypeAdapter
import base64
from fastapi import WebSocket

class WsManager:
    def __init__(self, agent_manager: AgentManager, audio_manager: AudioManager):
        self.agent_manager = agent_manager
        self.audio_manager = audio_manager

        self.agent_input_adapter = TypeAdapter[dict](Union[AgentRequest, AgentFallbackRequest])
        self.speak_input_adapter = TypeAdapter[dict](Union[SpeakRequest, AudioFallbackRequest])
        self.transcribe_input_adapter = TypeAdapter[dict](Union[TranscribeRequest, TranscribeFallbackRequest])

    async def handle_agent_run(self, payload: dict) -> AgentResponse:
        payload = self.agent_input_adapter.validate_python(payload)
        return  await self.agent_manager.safe_run(payload)

    async def handle_agent_run_stream(self, payload: dict | AgentRequest | AgentFallbackRequest) -> AsyncIterator[AgentResponse]:
        payload = self.agent_input_adapter.validate_python(payload)
        agent_stream = await self.agent_manager.safe_run(payload, stream=True)
        async for agent_response in agent_stream:
            yield agent_response

    async def handle_speak(self, payload: dict) -> EncodedSpeakResponse:
        payload = self.speak_input_adapter.validate_python(payload)
        result = await self.audio_manager.safe_speak(payload)
        return EncodedSpeakResponse(
            audio=base64.b64encode(result.audio).decode('utf-8'),
            content_type=result.content_type,
            sample_rate=result.sample_rate
        )

    async def handle_speak_stream(self, payload: dict) -> AsyncIterator[EncodedSpeakResponse]:
        payload = self.speak_input_adapter.validate_python(payload)
        generator, mime_type, sample_rate = await self.audio_manager.safe_speak(payload, stream=True)
        async for audio_chunk in generator:
            yield EncodedSpeakResponse(
                audio=audio_chunk,
                content_type=mime_type,
                sample_rate=sample_rate
            )

    async def handle_transcribe(self, payload: dict) -> TranscribeResponse:
        payload = self.transcribe_input_adapter.validate_python(payload)
        return await self.audio_manager.safe_transcribe(payload)
    
    
    async def handle_stream_response(self, data: AsyncIterator[BaseModel], action: WsAction) -> AsyncIterator[WsResponse]:
        try:
            async for chunk in data:
                yield WsResponse(
                    status=WsStatus.STREAMING,
                    action=action,
                    data=chunk.model_dump()
                )
            yield WsResponse(
                status=WsStatus.SUCCESS,
                action=action,
                data={}
            )
        except Exception as e:
            yield WsResponse(
                status=WsStatus.ERROR,
                action=action,
                data={},
                error=str(e)
            )
    
    async def handle_request(self, request: WsRequest) -> WsResponse | AsyncIterator[WsResponse]:
        try:
            match request.action:
                case WsAction.AGENT_RUN:
                    response = await self.handle_agent_run(request.payload)
                case WsAction.AGENT_RUN_STREAM:
                    response = self.handle_agent_run_stream(request.payload)
                case WsAction.AUDIO_SPEAK:
                    response = await self.handle_speak(request.payload)
                case WsAction.AUDIO_SPEAK_STREAM:
                    response = self.handle_speak_stream(request.payload)
                case WsAction.AUDIO_TRANSCRIBE:
                    response = await self.handle_transcribe(request.payload)
            if isinstance(response, AsyncIterator):
                return self.handle_stream_response(response, request.action)
            else:
                return WsResponse(
                    status=WsStatus.SUCCESS,
                    action=request.action,
                    data=response.model_dump()
                )
        except Exception as e:
            return WsResponse(
                status=WsStatus.ERROR,
                action=request.action,
                data={},
                error=str(e)
            )
    
    async def handle_request_with_response(self, websocket: WebSocket, request: WsRequest) -> WsResponse:
        response = await self.handle_request(request)
        if isinstance(response, AsyncIterator):
            async for chunk in response:
                print("CHUNK", chunk.model_dump())
                print("CHUNK TYPE", type(chunk))
                try:
                    await websocket.send_json(chunk.model_dump())
                    print("CHUNK SENT")
                except Exception as e:
                    print("ERROR SENDING CHUNK", e)
                    break
        else:
            await websocket.send_json(response.model_dump())