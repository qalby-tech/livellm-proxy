from agent.base import Agent
from models.inputs import MessageRole, TextMessage, MediaMessage
from models.inputs import WebSearchTool, MCPTool
from google import genai
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from typing import Optional, Literal, List, Dict, Union, Tuple
import logging
import asyncio
import base64

logger = logging.getLogger(__name__)


async def cancel_task(mcp_task: asyncio.Task, server_url: str):
    try:
        # Cancel the background task, which will trigger proper cleanup
        mcp_task.cancel()
        # Wait for the task to finish (with timeout)
        try:
            await asyncio.wait_for(mcp_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for MCP session cleanup for {server_url}")
        except asyncio.CancelledError:
            # Expected when task is cancelled
            pass
    except Exception as e:
        logger.warning(f"Error cleaning up MCP session for {server_url}: {e}")


class GeminiAgent(Agent):

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = genai.Client(api_key=api_key, http_options=genai.types.HttpOptions(base_url=base_url))
        self._mcp_sessions: Dict[str, ClientSession] = {}  # Cache MCP sessions by server URL
        self._mcp_tasks: Dict[str, asyncio.Task] = {}  # Track background tasks managing sessions
    
    async def __build_contents(self, messages: List[TextMessage | MediaMessage]) -> List[genai.types.Content]:
        built_contents: List[genai.types.Content] = []
        for message in messages:
            parts = []
            
            # Handle different message kinds
            if isinstance(message, TextMessage):
                parts.append(genai.types.Part(text=message.content))
            elif isinstance(message, MediaMessage):
                # Decode base64 content to bytes
                image_data = base64.b64decode(message.content)
                parts.append(genai.types.Part.from_bytes(data=image_data, mime_type=message.mime_type))
            else:
                raise ValueError(f"Unsupported message kind: {type(message)}")
            
            built_contents.append(genai.types.Content(
                role="model" if message.role == MessageRole.assistant else message.role.value,
                parts=parts
            ))
        return built_contents
    
    async def __manage_mcp_session(self, server_url: str, ready_event: asyncio.Event):
        """Background task to manage an MCP session using proper async context managers"""
        try:
            async with streamablehttp_client(url=server_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    # Store the session for use
                    self._mcp_sessions[server_url] = session
                    ready_event.set()  # Signal that session is ready
                    logger.info(f"MCP session initialized for {server_url}")
                    # Keep the session alive until cancelled
                    try:
                        await asyncio.Event().wait()  # Wait forever until cancelled
                    except asyncio.CancelledError:
                        logger.info(f"MCP session task for {server_url} cancelled")
                        raise
        except asyncio.CancelledError:
            # Clean cancellation, expected during shutdown
            pass
        except Exception as e:
            logger.error(f"Error in MCP session for {server_url}: {e}")
            ready_event.set()  # Set event even on error to prevent hanging
        finally:
            # Remove from cache when done
            self._mcp_sessions.pop(server_url, None)
            logger.info(f"MCP session cleaned up for {server_url}")
    
    async def __get_or_create_mcp_session(self, server_url: str):
        """Get cached MCP session or create a new one"""
        if server_url not in self._mcp_sessions:
            # Create a background task to manage the session
            ready_event = asyncio.Event()
            task = asyncio.create_task(self.__manage_mcp_session(server_url, ready_event))
            self._mcp_tasks[server_url] = task
            
            # Wait for the session to be ready (with timeout)
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                await cancel_task(task, server_url)
                raise RuntimeError(f"Timeout waiting for MCP session initialization for {server_url}")
            
            if server_url not in self._mcp_sessions:
                raise RuntimeError(f"Failed to initialize MCP session for {server_url}")
        
        return self._mcp_sessions[server_url]

    async def __build_tools(self, tools: Optional[list[WebSearchTool | MCPTool]]) -> Optional[Union[List[genai.types.Tool], ClientSession]]:
        if not tools:
            return None
        built_tools: List[genai.types.Tool] = []
        web_configured = False
        for tool in tools:
            if isinstance(tool, WebSearchTool):
                if web_configured:
                    logger.warning("WebSearchTool already configured, skipping")
                    continue
                built_tools.append(
                    genai.types.Tool(
                        google_search=genai.types.GoogleSearch(
                            exclude_domains=tool.web_search.get("exclude_domains", None),
                        )
                    )
                )
                web_configured = True
            if isinstance(tool, MCPTool):
                session = await self.__get_or_create_mcp_session(tool.server_url)
                built_tools.append(session)
        return built_tools
    
    async def chat(
        self, 
        model: str, 
        messages: list[TextMessage | MediaMessage], 
        tools: Optional[list[WebSearchTool, MCPTool]] = None, 
        tool_choice: Optional[Literal["auto", "none", "required"]] = "auto", 
        **kwargs):

        system: Optional[str] = None
        if messages and messages[0].role == MessageRole.system:
            system = messages[0].content
            messages = messages[1:]

        res = await self.client.aio.models.generate_content(
            model=model,
            contents=await self.__build_contents(messages),
            config=genai.types.GenerateContentConfig(
                system_instruction=system,
                tools=await self.__build_tools(tools),
                **kwargs
            )
        )
        response = res.candidates[0].content.parts[0].text
        prompt_tokens = res.usage_metadata.prompt_token_count
        completion_tokens = res.usage_metadata.candidates_token_count
        total_tokens = prompt_tokens + completion_tokens
        logger.info(f"Tools calls: {res.function_calls}") # will not show built-in tools calls
        logger.info(f"GeminiAgent run completed, prompt_tokens: {prompt_tokens}, completion_tokens: {completion_tokens}, total_tokens: {total_tokens}")
        return response
    
    async def transcribe(self, file: Tuple[str, bytes, str], model: str, language: Optional[str] = None) -> str:
        raise NotImplementedError("GeminiAgent has not transcribe method yet")
    
    async def speak(self, text: str, model: str, voice: str, response_format: Optional[str] = None) -> Tuple[bytes, str]:
        raise NotImplementedError("GeminiAgent has not speak method yet")
    

    async def clean_mcp_sessions(self):
        """
        Clean up all MCP sessions by cancelling their background tasks
        """
        for server_url, task in list(self._mcp_tasks.items()):
            await cancel_task(task, server_url)
        self._mcp_sessions.clear()
        self._mcp_tasks.clear()
