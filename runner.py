from agent.base import Agent
from typing import Dict, Optional
from utils import parse_model_ref
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class Method(str, Enum):
    transcribe = "transcribe"
    speak = "speak"
    chat = "chat"


class Runner:
    """
    Runner class that handles execution of agent methods with automatic fallback support.
    
    When a primary agent fails, it automatically falls back to the configured fallback model
    for the given method type (transcribe/speak/chat).
    """

    def __init__(
        self, 
        agents: Dict[str, Agent], 
        transcribe_fallback: Optional[str] = None, 
        speak_fallback: Optional[str] = None, 
        chat_fallback: Optional[str] = None
    ):
        self.agents = agents
        self.fallback_models = {
            Method.transcribe: transcribe_fallback,
            Method.speak: speak_fallback,
            Method.chat: chat_fallback
        }

    async def __run(self, *args, model: str, method: Method, enable_fallback: bool = True, **kwargs) -> str:
        provider, model_name = await parse_model_ref(model)

        agent: Optional[Agent] = self.agents.get(provider, None)
        if agent is None:
            raise ValueError(f"No agent found for provider: {provider}")
        
        runner = getattr(agent, method)
        try:
            return await runner(*args, model=model_name, **kwargs)
        except Exception as e:
            if enable_fallback and (fallback_model := self.fallback_models.get(method, None)):
                logger.warning(f"Fallback model: {fallback_model} used for method: {method} due to error: {e} in model: {model}")
                return await self.__run(*args, model=fallback_model, method=method, enable_fallback=False, **kwargs)
            raise e
    
    async def transcribe(self, *args, model: str, enable_fallback: bool = True, **kwargs) -> str:
        return await self.__run(*args, model=model, method=Method.transcribe, enable_fallback=enable_fallback, **kwargs)
    
    async def speak(self, *args, model: str, enable_fallback: bool = True, **kwargs) -> str:
        return await self.__run(*args, model=model, method=Method.speak, enable_fallback=enable_fallback, **kwargs)
    
    async def chat(self, *args, model: str, enable_fallback: bool = True, **kwargs) -> str:
        return await self.__run(*args, model=model, method=Method.chat, enable_fallback=enable_fallback, **kwargs)