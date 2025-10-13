from agent.base import Agent, TextMessage, MediaMessage, WebSearchTool, MCPTool
from elevenlabs.client import AsyncElevenLabs
from typing import Optional, List, Union, Literal, Tuple
import logging

logger = logging.getLogger(__name__)


class ElevenLabsAgent(Agent):
    """ElevenLabs agent implementation (specialized for audio)"""

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        if base_url:
            self.client = AsyncElevenLabs(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncElevenLabs(api_key=api_key)

    async def chat(
        self,
        model: str,
        messages: List[TextMessage | MediaMessage],
        tools: Optional[List[Union[WebSearchTool, MCPTool]]] = None,
        tool_choice: Optional[Literal["auto", "none", "required"]] = "auto",
        **kwargs
    ) -> str:
        """ElevenLabs doesn't have a chat/completion API"""
        raise NotImplementedError("ElevenLabsAgent does not have a main agent/chat API")
    

    async def transcribe(self, file: Tuple[str, bytes, str], model: str, language: Optional[str] = None) -> str:
        """Transcribe audio using ElevenLabs STT"""
        try:
            # ElevenLabs expects file-like object or path
            response = await self.client.speech_to_text.convert(
                file=file,
                language_code=language,
                model_id=model
            )
            
            logger.debug(f"ElevenLabs transcription completed with model: {model}")
            return response.text
            
        except Exception as e:
            logger.error(f"ElevenLabs transcription failed: {str(e)}")
            raise
    
    async def speak(self, text: str, model: str, voice: str, response_format: str = "mp3_44100_128") -> Tuple[bytes, str]:
        """Generate speech using ElevenLabs TTS"""
        try:
            params = {
                "text": text,
                "voice_id": voice,
                "model_id": model,
            }
            # ElevenLabs uses output_format parameter
            if response_format:
                params["output_format"] = response_format
            
            audio_generator = self.client.text_to_speech.convert(**params)
            
            # Collect all audio chunks
            audio_bytes = b""
            async for chunk in audio_generator:
                audio_bytes += chunk
            
            # Parse ElevenLabs format (e.g., "mp3_44100_128" -> "mp3")
            codec = response_format.split("_", maxsplit=1)[0]
            
            # Map codec to MIME type
            mime_type = self.speak_response_format_to_mime_type.get(codec, None)
            if not mime_type:
                raise ValueError(f"Unsupported response format: {response_format}")
            return audio_bytes, mime_type
            
        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {str(e)}")
            raise
