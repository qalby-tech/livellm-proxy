import os
from agent.base import Agent
from models.inputs import WebSearchTool, MCPTool, MessageRole
from models.inputs import TextMessage, MediaMessage
from openai import AsyncOpenAI
from openai.types.responses import Response
from agents import Agent, Runner
from agents.model_settings import ModelSettings
from agents.run import RunResult
from agents.mcp import MCPServerStreamableHttp
from agents import WebSearchTool as AgentsWebSearchTool
# from agents import HostedMCPTool
from typing import Optional, Literal, List, Dict, Any, Tuple
import logging
import asyncio

logger = logging.getLogger(__name__)

# Filter to suppress non-fatal OpenAI tracing warnings
class OpenAITracingFilter(logging.Filter):
    def filter(self, record):
        # Suppress tracing client errors
        if record.name == "openai.agents" and "[non-fatal]" in record.getMessage():
            return False
        return True

# Apply filter to openai.agents logger
openai_agents_logger = logging.getLogger("openai.agents")
openai_agents_logger.addFilter(OpenAITracingFilter())


async def cancel_task(mcp_task: asyncio.Task, server_name: str):
    """Cancel and wait for an MCP task to finish"""
    try:
        # Cancel the background task, which will trigger proper cleanup
        mcp_task.cancel()
        # Wait for the task to finish (with timeout)
        try:
            await asyncio.wait_for(mcp_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for MCP session cleanup for {server_name}")
        except asyncio.CancelledError:
            # Expected when task is cancelled
            pass
    except Exception as e:
        logger.warning(f"Error cleaning up MCP session for {server_name}: {e}")

class OpenAIAgent(Agent):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        # self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
        self.client = AsyncOpenAI()
        self._mcp_servers: Dict[str, Any] = {}  # Cache MCP servers by name
        self._mcp_tasks: Dict[str, asyncio.Task] = {}  # Track background tasks managing servers

    
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
    
    async def __manage_mcp_server(self, name: str, server_url: str, ready_event: asyncio.Event):
        """Background task to manage an MCP server using proper async context managers"""
        try:
            server_inst = MCPServerStreamableHttp(
                name=name,
                params={
                    "url": server_url,
                }
            )
            async with server_inst as server:
                # Store the server for use
                self._mcp_servers[name] = server
                ready_event.set()  # Signal that server is ready
                logger.info(f"MCP server initialized for {name}")
                # Keep the server alive until cancelled
                try:
                    await asyncio.Event().wait()  # Wait forever until cancelled
                except asyncio.CancelledError:
                    logger.info(f"MCP server task for {name} cancelled")
                    raise
        except asyncio.CancelledError:
            # Clean cancellation, expected during shutdown
            pass
        except Exception as e:
            logger.error(f"Error in MCP server for {name}: {e}")
            ready_event.set()  # Set event even on error to prevent hanging
        finally:
            # Remove from cache when done
            self._mcp_servers.pop(name, None)
            logger.info(f"MCP server cleaned up for {name}")
    
    async def __get_or_create_mcp_server(self, name: str, server_url: str):
        """Get cached MCP server or create a new one"""
        if name not in self._mcp_servers:
            # Create a background task to manage the server
            ready_event = asyncio.Event()
            task = asyncio.create_task(self.__manage_mcp_server(name, server_url, ready_event))
            self._mcp_tasks[name] = task
            
            # Wait for the server to be ready (with timeout)
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                await cancel_task(task, name)
                raise RuntimeError(f"Timeout waiting for MCP server initialization for {name}")
            
            if name not in self._mcp_servers:
                raise RuntimeError(f"Failed to initialize MCP server for {name}")
        
        return self._mcp_servers[name]

    async def __build_tools(self, tools: Optional[list[WebSearchTool | MCPTool]]) -> Optional[list[dict]]:
        if not tools:
            return None
        built_tools: list[dict] = []
        for tool in tools:
            if isinstance(tool, WebSearchTool):
                built_tools.append(
                    AgentsWebSearchTool(
                        search_context_size=tool.web_search.get("search_context_size", "medium"),
                    )
                )
        return built_tools

    async def __build_mcp(self, tools: Optional[list[MCPTool]]) -> Optional[list[dict]]:
        if not tools:
            return None
        built_mcp: list[dict] = []
        for tool in tools:
            if isinstance(tool, MCPTool):
                server = await self.__get_or_create_mcp_server(tool.name, tool.server_url)
                built_mcp.append(server)
                
        return built_mcp
    
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
        
        system = "You are a helpful assistant"
        if messages and messages[0].role == MessageRole.system:
            system = messages[0].content
            messages = messages[1:]
        
        agent = Agent(
            name="Assistant", 
            instructions=system,
            model=model,
            tools=await self.__build_tools(tools),
            mcp_servers= await self.__build_mcp(tools),
            model_settings=ModelSettings(
                tool_choice=tool_choice,
                **kwargs
            )
        )
        response: RunResult = await Runner.run(
            agent,
            input=await self.__build_messages(messages),
        )
        res: Response = response.raw_responses[-1]
        prompt_tokens = res.usage.input_tokens
        completion_tokens = res.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        logger.info(f"OpenAIAgent chat completed, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}")
        return response.final_output
    
    async def close(self):
        """Clean up all MCP servers by cancelling their background tasks"""
        for server_name, task in list(self._mcp_tasks.items()):
            await cancel_task(task, server_name)
        self._mcp_servers.clear()
        self._mcp_tasks.clear()
        
        # Close the OpenAI client
        await self.client.close()