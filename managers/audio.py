from typing import Union, Tuple, Optional, Type
from audio_ai.base import AudioAIService
from audio_ai.openai import OpenAIAudioAIService
from audio_ai.elevenlabs import ElevenLabsAudioAIService

# pydantic models
from models.common import ProviderKind, BaseRequest, FallbackStrategyType
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from models.fallback import AudioFallbackRequest, TranscribeFallbackRequest, FallbackRequest, FallbackStrategy

# managers
from managers import telemetry as logfire
from managers.config import ConfigManager, ProviderClient
from managers.fallback import FallbackManager

class AudioManager:
    
    def __init__(self, config_manager: ConfigManager, fallback_manager: Optional[FallbackManager] = None):
        self.config_manager = config_manager
        self.fallback_manager = fallback_manager

    def _build_config_fallback(
        self,
        payload: BaseRequest,
        fallback_cls: Type[FallbackRequest],
    ) -> Optional[FallbackRequest]:
        """
        Build an automatic fallback request from the provider's model config.

        Mirrors AgentManager._get_fallback_config for the audio endpoints: if the
        primary provider/model has a `fallback` configured in its ModelConfig, wrap
        the original request and a copy pointed at the fallback provider/model into a
        FallbackRequest. Returns None when no fallback is configured.

        Note: FallbackConfig.context_limit / context_overflow_strategy are intended
        for chat context truncation and do not apply to audio requests (Speak/
        Transcribe have no such fields), so they are intentionally ignored here.
        """
        model_config = self.config_manager.get_model_config(payload.provider_uid, payload.model)
        if not (model_config and model_config.fallback):
            return None

        fb = model_config.fallback
        fallback_request = payload.model_copy(update={
            "provider_uid": fb.fallback_provider_uid,
            "model": fb.fallback_model,
        })

        if fb.fallback_strategy == FallbackStrategyType.PARALLEL:
            strategy = FallbackStrategy.PARALLEL
        else:
            strategy = FallbackStrategy.SEQUENTIAL

        logfire.info(
            f"Using automatic audio fallback from config: "
            f"{payload.provider_uid}/{payload.model} -> "
            f"{fb.fallback_provider_uid}/{fb.fallback_model} (strategy={strategy.value})"
        )
        return fallback_cls(requests=[payload, fallback_request], strategy=strategy)

    def create_service(
        self, 
        uid: str,
        model: str,
    ) -> AudioAIService:
        """
        Create an audio service using the cached provider
        """        
        provider_kind, provider_client = self.config_manager.get_provider(uid, model)
        # Create new service based on type
        service: AudioAIService
        if provider_kind == ProviderKind.OPENAI:
            service = OpenAIAudioAIService(client=provider_client)
        elif provider_kind == ProviderKind.OPENAI_CHAT:
            service = OpenAIAudioAIService(client=provider_client)
        elif provider_kind == ProviderKind.ELEVENLABS:
            service = ElevenLabsAudioAIService(client=provider_client)
        else:
            raise ValueError(f"Provider {provider_kind} not supported for audio services")
        
        return service
    
    async def speak(
        self, 
        payload: SpeakRequest, 
        stream: bool = False
    ) -> Union[SpeakResponse, SpeakStreamResponse]:
        """
        Convert text to speech using the specified provider.
        
        Args:
            settings: Provider settings (provider, API key, base URL)
            payload: SpeakRequest containing model, text, voice, and output format
            stream: If True, returns SpeakStreamResponse. If False, returns SpeakResponse
            
        Returns:
            If stream=False: SpeakResponse containing audio data, content type, and sample rate
            If stream=True: SpeakStreamResponse (tuple of AsyncIterator[bytes], mime_type, sample_rate)
        """
        # Create the service using the cached provider
        service = self.create_service(
            uid=payload.provider_uid,
            model=payload.model
        )
        
        # Call the speak method on the service
        if stream:
            return await service.stream_speak(payload)
        else:
            return await service.speak(payload)
    
    async def transcribe(
        self, 
        payload: TranscribeRequest
    ) -> TranscribeResponse:
        """
        Transcribe audio to text using the specified provider.
        
        Args:
            uid: The unique identifier of the provider configuration
            payload: TranscribeRequest containing model, audio file, and language
            
        Returns:
            TranscribeResponse containing transcribed text, detected language, and usage
        """
        # Create the service using the cached provider
        service = self.create_service(
            uid=payload.provider_uid,
            model=payload.model
        )
        
        # Call the transcribe method on the service
        return await service.transcribe(payload)
    
    async def safe_speak(
        self,
        payload: Union[SpeakRequest, AudioFallbackRequest],
        stream: bool = False
    ) -> Union[SpeakResponse, SpeakStreamResponse]:
        """
        Convert text to speech with optional fallback support.
        
        If payload is SpeakRequest: runs normally
        If payload is AudioFallbackRequest: uses fallback manager to try multiple requests
        
        Args:
            payload: Either SpeakRequest or AudioFallbackRequest
            stream: If True, returns SpeakStreamResponse. If False, returns SpeakResponse
            
        Returns:
            If stream=False: SpeakResponse containing audio data, content type, and sample rate
            If stream=True: SpeakStreamResponse (tuple of AsyncIterator[bytes], mime_type, sample_rate)
        """
        # Simple request: build an automatic fallback from provider config if one
        # is configured (the client-supplied AudioFallbackRequest below takes
        # priority over config, matching the agent path).
        if isinstance(payload, SpeakRequest):
            config_fallback = (
                self._build_config_fallback(payload, AudioFallbackRequest)
                if self.fallback_manager else None
            )
            if config_fallback is None:
                return await self.speak(payload, stream=stream)
            payload = config_fallback

        # If it's a fallback request, use the fallback manager
        if not self.fallback_manager:
            raise ValueError("FallbackManager not configured. Cannot use fallback requests.")

        # Define the executor function for the fallback manager
        async def executor(request: SpeakRequest) -> Union[SpeakResponse, SpeakStreamResponse]:
            return await self.speak(request, stream=stream)

        # Use the fallback manager's catch method
        return await self.fallback_manager.catch(payload, executor)
    
    async def safe_transcribe(
        self,
        payload: Union[TranscribeRequest, TranscribeFallbackRequest]
    ) -> TranscribeResponse:
        """
        Transcribe audio to text with optional fallback support.
        
        If payload is TranscribeRequest: runs normally
        If payload is TranscribeFallbackRequest: uses fallback manager to try multiple requests
        
        Args:
            payload: Either TranscribeRequest or TranscribeFallbackRequest
            
        Returns:
            TranscribeResponse containing transcribed text, detected language, and usage
        """
        # Simple request: build an automatic fallback from provider config if one
        # is configured (a client-supplied TranscribeFallbackRequest takes priority).
        if isinstance(payload, TranscribeRequest):
            config_fallback = (
                self._build_config_fallback(payload, TranscribeFallbackRequest)
                if self.fallback_manager else None
            )
            if config_fallback is None:
                return await self.transcribe(payload)
            payload = config_fallback

        # If it's a fallback request, use the fallback manager
        if not self.fallback_manager:
            raise ValueError("FallbackManager not configured. Cannot use fallback requests.")

        # Define the executor function for the fallback manager
        async def executor(request: TranscribeRequest) -> TranscribeResponse:
            return await self.transcribe(request)

        # Use the fallback manager's catch method
        return await self.fallback_manager.catch(payload, executor)