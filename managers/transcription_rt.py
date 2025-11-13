from managers.config import ConfigManager
from audio_ai.base import AudioRealtimeTranscriptionService
from models.audio.transcription_ws import TranscriptionInitWsRequest
from models.common import ProviderKind
from audio_ai.openai import OpenAIRealtimeTranscriptionService


class TranscriptionRTManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def create_service(self, request: TranscriptionInitWsRequest) -> AudioRealtimeTranscriptionService:
        provider_kind, provider_client = self.config_manager.get_provider(request.provider_uid, request.model)
        if provider_kind == ProviderKind.OPENAI:
            return OpenAIRealtimeTranscriptionService(
                openai_client=provider_client,
                model=request.model,
                language=request.language,
                gen_config=request.gen_config
            )
        else:
            raise ValueError(f"Provider {provider_kind} not supported for transcription services")