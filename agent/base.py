from abc import ABC, abstractmethod
from typing import Optional, Literal, Union, Tuple, Dict
from models.inputs import TextMessage, MediaMessage, MCPTool, WebSearchTool


class Agent(ABC):
    # utility mapping for speak response format to mime type
    speak_response_format_to_mime_type: Dict[str, str] = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }

    @abstractmethod
    async def transcribe(self, file: Tuple[str, bytes, str], model: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio file.
        
        Args:
            file: Tuple of (filename, file_content, content_type)
            model: Model name to use for transcription
            language: Optional language code
        """
        pass

    @abstractmethod
    async def speak(self, text: str, model: str, voice: str, response_format: Optional[str] = None) -> Tuple[bytes, str]:
        """
        Generate speech from text.
        
        Returns:
            Tuple of (audio_bytes, mime_type)
        """
        pass


    @abstractmethod
    async def chat(
        self, 
        model: str,
        messages: list[TextMessage | MediaMessage], 
        tools: Optional[list[Union[WebSearchTool, MCPTool]]] = None, 
        tool_choice: Optional[Literal["auto", "none", "required"]] = "auto", 
        **kwargs
        ) -> str:
        """
        it is expected that agent will run tools itself
        """
        pass

    
    async def close(self):
        """
        clean up all resources
        """
        pass