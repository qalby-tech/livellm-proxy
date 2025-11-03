from typing import Dict, Union, Optional, Tuple
from audio_ai.base import AudioAIService
from audio_ai.openai import OpenAIAudioAIService
from audio_ai.elevenlabs import ElevenLabsAudioAIService

# pydantic models
from models.common import Settings, ProviderKind
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse

# managers
from managers.config import ConfigManager, ProviderClient

class AudioManager:
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    
    def create_service(
        self, 
        uid: str,
        model: str,
    ) -> AudioAIService:
        """
        Create an audio service using the cached provider
        """        
        provider_kind, provider_client: Tuple[ProviderKind, ProviderClient] = self.config_manager.get_provider(uid, model)
        # Create new service based on type
        service: AudioAIService
        if provider_kind == ProviderKind.OPENAI:
            service = OpenAIAudioAIService(client=provider_client)
        elif provider_kind == ProviderKind.ELEVENLABS:
            service = ElevenLabsAudioAIService(client=provider_client)
        else:
            raise ValueError(f"Provider {provider_kind} not supported for audio services")
        
        return service
    
    async def speak(
        self, 
        uid: str,
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
            uid=uid,
            model=payload.model
        )
        
        # Call the speak method on the service
        if stream:
            return await service.stream_speak(payload)
        else:
            return await service.speak(payload)
    
    async def transcribe(
        self, 
        uid: str,
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
            uid=uid,
            model=payload.model
        )
        
        # Call the transcribe method on the service
        return await service.transcribe(payload)