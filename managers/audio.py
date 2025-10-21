from typing import Dict, Union, Optional
from audio_ai.base import AudioAIService
from audio_ai.openai import OpenAIAudioAIService
from audio_ai.elevenlabs import ElevenLabsAudioAIService

# pydantic models
from models.audio.audio import AudioProviderKind
from models.audio.speak import SpeakRequest, SpeakResponse


class AudioManager:
    
    def __init__(self):
        self.services: Dict[str, Union[OpenAIAudioAIService, ElevenLabsAudioAIService]] = {}
    
    def _get_service_cache_key(self, provider: AudioProviderKind, api_key: str, base_url: Optional[str] = None) -> str:
        """Generate a unique cache key for a service configuration"""
        base = base_url or "default"
        return f"{provider.value}:{api_key}:{base}"
    
    def create_service(
        self, 
        provider: AudioProviderKind, 
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
        if provider == AudioProviderKind.OPENAI:
            new_service = OpenAIAudioAIService(api_key=api_key, base_url=base_url)
        elif provider == AudioProviderKind.ELEVENLABS:
            new_service = ElevenLabsAudioAIService(api_key=api_key, base_url=base_url)
        elif provider == AudioProviderKind.GOOGLE:
            raise NotImplementedError("Google audio service is not implemented yet")
        else:
            raise ValueError(f"Provider {provider} not supported")
        
        # Cache and return the new service
        self.services[cache_key] = new_service
        return new_service
    
    async def speak(self, request: SpeakRequest) -> SpeakResponse:
        """
        Convert text to speech using the specified provider.
        
        Args:
            request: SpeakRequest containing settings, text, voice, and output format
            
        Returns:
            SpeakResponse containing audio data, content type, and sample rate
        """
        # Create the service using the cached provider
        service = self.create_service(
            provider=request.settings.provider,
            api_key=request.settings.api_key,
            base_url=request.settings.base_url
        )
        
        # Call the speak method on the service
        return await service.speak(request)