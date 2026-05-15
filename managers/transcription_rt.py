from managers.config import ConfigManager, LivellmClient
from audio_ai.base import AudioRealtimeTranscriptionService
from audio_ai.livellm import LivellmRealtimeTranscriptionService
from audio_ai.openai import OpenAIRealtimeTranscriptionService
from models.audio.transcription_ws import TranscriptionInitWsRequest
from models.common import ProviderKind


class TranscriptionRTManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def create_service(self, request: TranscriptionInitWsRequest) -> AudioRealtimeTranscriptionService:
        provider_kind, provider_client = self.config_manager.get_provider(request.provider_uid, request.model)
        if provider_kind in (ProviderKind.OPENAI, ProviderKind.OPENAI_CHAT):
            return OpenAIRealtimeTranscriptionService(
                openai_client=provider_client,
                model=request.model,
                language=request.language,
                gen_config=request.gen_config,
            )
        elif provider_kind == ProviderKind.LIVELLM:
            assert isinstance(provider_client, LivellmClient)
            return LivellmRealtimeTranscriptionService(
                base_url=provider_client.base_url,
                api_key=provider_client.api_key,
                model=request.model,
                language=request.language,
                gen_config=request.gen_config,
            )
        else:
            raise ValueError(f"Provider {provider_kind} not supported for transcription services")