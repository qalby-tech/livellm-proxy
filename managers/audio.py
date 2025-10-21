from typing import Dict, Union, Optional
from audio_ai.base import AudioAIService
from audio_ai.openai import OpenAIAudioAIService
from audio_ai.elevenlabs import ElevenLabsAudioAIService

# pydantic models
from models.common import Settings, ProviderKind
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse


class AudioManager:
    
    def __init__(self):
        self.services: Dict[str, Union[OpenAIAudioAIService, ElevenLabsAudioAIService]] = {}
    
    def _get_service_cache_key(self, provider: ProviderKind, api_key: str, base_url: Optional[str] = None) -> str:
        """Generate a unique cache key for a service configuration"""
        base = base_url or "default"
        return f"{provider.value}:{api_key}:{base}"
    
    def create_service(
        self, 
        provider: ProviderKind, 
        api_key: str, 
        base_url: Optional[str] = None
    ) -> AudioAIService:
        """
        Create or retrieve a cached audio service instance.
        If a service with the same configuration already exists, return it.
        Otherwise, create a new one and cache it.
        """
        cache_key = self._get_service_cache_key(provider, api_key, base_url)
        
        # Check if service already exists in cache
        if cache_key in self.services:
            return self.services[cache_key]
        
        # Create new service based on type
        if provider == ProviderKind.OPENAI:
            new_service = OpenAIAudioAIService(api_key=api_key, base_url=base_url)
        elif provider == ProviderKind.ELEVENLABS:
            new_service = ElevenLabsAudioAIService(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Provider {provider} not supported for audio services")
        
        # Cache and return the new service
        self.services[cache_key] = new_service
        return new_service
    
    async def speak(
        self, 
        settings: Settings,
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
            provider=settings.provider,
            api_key=settings.api_key,
            base_url=settings.base_url
        )
        
        # Call the speak method on the service
        if stream:
            return await service.stream_speak(payload)
        else:
            return await service.speak(payload)
    
    async def transcribe(
        self, 
        settings: Settings,
        payload: TranscribeRequest,
        stream: bool = False
    ) -> TranscribeResponse:
        """
        Transcribe audio to text using the specified provider.
        
        Args:
            settings: Provider settings (provider, API key, base URL)
            payload: TranscribeRequest containing model, audio file, and language
            stream: Not used for transcription (kept for consistency)
            
        Returns:
            TranscribeResponse containing transcribed text, detected language, and usage
        """
        # Create the service using the cached provider
        service = self.create_service(
            provider=settings.provider,
            api_key=settings.api_key,
            base_url=settings.base_url
        )
        
        # Call the transcribe method on the service
        return await service.transcribe(payload)