from contextlib import suppress
from typing import List, Optional, Tuple

from managers.config import ConfigManager, LivellmClient
from managers import telemetry as logfire
from audio_ai.base import AudioRealtimeTranscriptionService
from audio_ai.livellm import LivellmRealtimeTranscriptionService
from audio_ai.openai import OpenAIRealtimeTranscriptionService
from models.audio.transcription_ws import TranscriptionInitWsRequest
from models.common import ProviderKind


class TranscriptionRTManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def _build_service(
        self,
        provider_uid: str,
        model: str,
        request: TranscriptionInitWsRequest,
    ) -> AudioRealtimeTranscriptionService:
        provider_kind, provider_client = self.config_manager.get_provider(provider_uid, model)
        if provider_kind in (ProviderKind.OPENAI, ProviderKind.OPENAI_CHAT):
            return OpenAIRealtimeTranscriptionService(
                openai_client=provider_client,
                model=model,
                language=request.language,
                gen_config=request.gen_config,
            )
        elif provider_kind == ProviderKind.LIVELLM:
            assert isinstance(provider_client, LivellmClient)
            return LivellmRealtimeTranscriptionService(
                base_url=provider_client.base_url,
                api_key=provider_client.api_key,
                model=model,
                language=request.language,
                gen_config=request.gen_config,
            )
        else:
            raise ValueError(f"Provider {provider_kind} not supported for transcription services")

    def create_service(self, request: TranscriptionInitWsRequest) -> AudioRealtimeTranscriptionService:
        """Build (but do not connect) a realtime transcription service for the request's provider."""
        return self._build_service(request.provider_uid, request.model, request)

    def _fallback_targets(self, provider_uid: str, model: str) -> List[Tuple[str, str]]:
        """Ordered (provider_uid, model) connect targets: primary first, then the
        configured fallback (if any).

        Realtime transcription is a long-lived bidirectional WebSocket, so — unlike
        the batch /audio/transcribe path — fallback can only be applied at *connect
        time*: if the primary provider is unreachable we connect the fallback before
        streaming starts. Mid-stream failover is not attempted (the audio buffer is
        already in flight). This reads the same `ModelConfig.fallback` that drives the
        HTTP audio / agent paths, so a single provider config covers every endpoint.
        """
        targets: List[Tuple[str, str]] = [(provider_uid, model)]
        model_config = self.config_manager.get_model_config(provider_uid, model)
        if model_config and model_config.fallback:
            fb = model_config.fallback
            targets.append((fb.fallback_provider_uid, fb.fallback_model))
        return targets

    async def create_connected_service(
        self, request: TranscriptionInitWsRequest
    ) -> AudioRealtimeTranscriptionService:
        """Build and connect a realtime transcription service, falling back to the
        configured provider/model if the primary fails to connect.

        Targets are tried sequentially; the first one that connects is returned. The
        `fallback_strategy` is treated as sequential here regardless of its value —
        racing two long-lived ASR sockets in parallel would leak the losing
        connection, so connect-time fallback is always ordered.
        """
        targets = self._fallback_targets(request.provider_uid, request.model)
        last_exc: Optional[Exception] = None
        for index, (uid, model) in enumerate(targets):
            service: Optional[AudioRealtimeTranscriptionService] = None
            try:
                service = self._build_service(uid, model, request)
                await service.connect()
                if index > 0:
                    logfire.info(
                        f"Realtime transcription connected via fallback target "
                        f"{uid}/{model} (primary {request.provider_uid}/{request.model} failed)"
                    )
                return service
            except Exception as e:
                last_exc = e
                logfire.warning(
                    f"Realtime transcription connect failed for {uid}/{model}: {e}"
                )
                if service is not None:
                    with suppress(Exception):
                        await service.disconnect()
                continue

        raise RuntimeError(
            f"All realtime transcription targets failed for "
            f"{request.provider_uid}/{request.model}"
        ) from last_exc
