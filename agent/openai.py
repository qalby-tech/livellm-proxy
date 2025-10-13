from agent.base import Agent
from models.inputs import WebSearchTool, MCPTool, MessageRole
from models.inputs import TextMessage, MediaMessage
from openai import AsyncOpenAI
from openai.types.responses import Response
from typing import Optional, Literal, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class OpenAIAgent(Agent):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    
    async def __build_messages(self, messages: list[TextMessage | MediaMessage]) -> List[Dict[str, Any]]:
        built_messages = []
        for message in messages:
            msg_dict: Dict[str, Any] = {
                "role": "developer" if message.role == MessageRole.system else message.role.value,
            }
            
            # Handle different message kinds
            if isinstance(message, TextMessage):
                msg_dict["content"] = message.content
            elif isinstance(message, MediaMessage):
                # OpenAI format for images
                msg_dict["content"] = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{message.mime_type};base64,{message.content}"
                        }
                    }
                ]            
            built_messages.append(msg_dict)
        return built_messages
    

    async def __build_tools(self, tools: Optional[list[WebSearchTool | MCPTool]]) -> Optional[list[dict]]:
        if not tools:
            return None
        built_tools: list[dict] = []
        for tool in tools:
            if isinstance(tool, WebSearchTool):
                built_tools.append({
                    "type": "web_search",
                })
            if isinstance(tool, MCPTool):
                built_tools.append({
                    "type": "mcp",
                    "server_label": tool.name,
                    "server_url": tool.server_url,
                    "require_approval": "never"
                })
        return built_tools
    
    
    async def transcribe(self, file: Tuple[str, bytes, str], model: str, language: Optional[str] = None) -> str:
        res = await self.client.audio.transcriptions.create(
            file=file,
            model=model,
            language=language,
        )
        return res.text

    async def speak(self, text: str, model: str, voice: str, response_format: str = "mp3") -> Tuple[bytes, str]:
        params = {
            "model": model,
            "voice": voice,
            "input": text,
        }
        if response_format:
            params["response_format"] = response_format
        
        res = await self.client.audio.speech.create(**params)
        
        # Map OpenAI response format to MIME type
        mime_type = self.speak_response_format_to_mime_type.get(response_format, None)
        if not mime_type:
            raise ValueError(f"Unsupported response format: {response_format}")
        
        return res.content, mime_type
    
    async def chat(
        self, 
        model: str, 
        messages: list[TextMessage | MediaMessage], 
        tools: Optional[list[WebSearchTool, MCPTool]] = None, 
        tool_choice: Optional[Literal["auto", "none", "required"]] = "auto", 
        **kwargs) -> str:
        params = {}
        if tools:
            params["tools"] = await self.__build_tools(tools)
            params["tool_choice"] = tool_choice
        res: Response = await self.client.responses.create(
            model=model,
            input=await self.__build_messages(messages),
            **params,
            **kwargs
        )
        prompt_tokens = res.usage.input_tokens
        completion_tokens = res.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        logger.info(f"Tools calls: {res.tools}")
        logger.info(f"OpenAIAgent chat completed, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}")
        return res.output_text
