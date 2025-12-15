from typing import Union, Tuple, Optional
from audio_ai.base import AudioAIService
from audio_ai.openai import OpenAIAudioAIService
from audio_ai.elevenlabs import ElevenLabsAudioAIService

# pydantic models
from models.common import ProviderKind
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from models.fallback import AudioFallbackRequest, TranscribeFallbackRequest

# managers
from managers.config import ConfigManager, ProviderClient
from managers.fallback import FallbackManager

class AudioManager:
    
    def __init__(self, config_manager: ConfigManager, fallback_manager: Optional[FallbackManager] = None):
        self.config_manager = config_manager
        self.fallback_manager = fallback_manager

    
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
        # If it's a simple request, run normally
        if isinstance(payload, SpeakRequest):
            return await self.speak(payload, stream=stream)
        
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
        # If it's a simple request, run normally
        if isinstance(payload, TranscribeRequest):
            return await self.transcribe(payload)
        
        # If it's a fallback request, use the fallback manager
        if not self.fallback_manager:
            raise ValueError("FallbackManager not configured. Cannot use fallback requests.")
        
        # Define the executor function for the fallback manager
        async def executor(request: TranscribeRequest) -> TranscribeResponse:
            return await self.transcribe(request)
        
        # Use the fallback manager's catch method
        return await self.fallback_manager.catch(payload, executor)