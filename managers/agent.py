from pydantic_ai import Agent
from typing import Optional, List, Union, Tuple, AsyncIterator

# tools
from pydantic_ai import WebSearchTool
from pydantic_ai.mcp import MCPServerStreamableHTTP

# messages
from pydantic_ai import BinaryContent, ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.messages import UserPromptPart, TextPart, SystemPromptPart

# models
from pydantic_ai import ModelSettings
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.groq import GroqModel

# pydantic models
from models.common import ProviderKind
from models.agent.tools import WebSearchInput, MCPStreamableServerInput, ToolKind, ToolInput
from models.agent.chat import MessageRole, TextMessage, BinaryMessage
from models.agent.run import AgentRequest, AgentResponse, AgentResponseUsage
from managers.config import ConfigManager, ProviderClient
import base64


class AgentManager:
    

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.agent = Agent[None, str]()
        
    def create_model(self, uid: str, model: str, gen_config: Optional[dict] = None):
        """Create a model using the cached provider"""
        provider_kind, provider_client = self.config_manager.get_provider(uid, model)
        
        # Use empty dict if gen_config is None to avoid **None error
        config = gen_config or {}

        model_base: Model
        if provider_kind == ProviderKind.OPENAI:
            model_base = OpenAIResponsesModel
        elif provider_kind == ProviderKind.GOOGLE:
            model_base = GoogleModel
        elif provider_kind == ProviderKind.ANTHROPIC:
            model_base = AnthropicModel
        elif provider_kind == ProviderKind.GROQ:
            model_base = GroqModel
        else:
            raise ValueError(f"Provider {provider_kind} not supported")
        
        return model_base(
            model_name=model, 
            provider=provider_client,
            settings=ModelSettings(
                **config
            )
        )
    
    
    def create_tools(self, tools: List[ToolInput]) -> Tuple[List[WebSearchTool], List[MCPServerStreamableHTTP]]:
        builtin_tools = []
        mcp_servers = []
        init_cache = set() # avoid duplicate tools
        for tool in tools:
            if isinstance(tool, WebSearchInput):
                if ToolKind.WEB_SEARCH not in init_cache:
                    # using tool's unique key to deduplicate
                    builtin_tools.append(WebSearchTool(search_context_size=tool.search_context_size, **tool.kwargs))
                    init_cache.add(ToolKind.WEB_SEARCH)
            elif isinstance(tool, MCPStreamableServerInput):
                if tool.url not in init_cache:
                    # using url to deduplicate mcp
                    mcp_servers.append(MCPServerStreamableHTTP(url=tool.url, tool_prefix=tool.prefix, timeout=tool.timeout, **tool.kwargs))
                    init_cache.add(tool.url)
            else:
                raise ValueError(f"Unknown tool kind: {tool.kind}")
        return builtin_tools, mcp_servers
    
    def convert_msg(self, msg: Union[TextMessage, BinaryMessage]) -> Union[ModelMessage, ModelResponse]:
        parts = []
        if msg.role == MessageRole.USER:
            if isinstance(msg, TextMessage):
                parts.append(UserPromptPart(content=msg.content))
            elif isinstance(msg, BinaryMessage):
                if msg.caption:
                    parts.append(UserPromptPart(content=msg.caption))
                # Decode base64 string to bytes for BinaryContent
                binary_data = base64.b64decode(msg.content)
                parts.append(BinaryContent(data=binary_data, media_type=msg.mime_type))
            else:
                raise ValueError(f"Unknown message type: {type(msg)}")
            return ModelRequest(parts=parts)
        elif msg.role == MessageRole.MODEL:
            if isinstance(msg, TextMessage):
                parts.append(TextPart(content=msg.content))
            else:
                raise ValueError(f"Unknown message type: {type(msg)}")
            return ModelResponse(parts=parts)
        elif msg.role == MessageRole.SYSTEM:
            parts.append(SystemPromptPart(content=msg.content))
            return ModelRequest(parts=parts)
        else:
            raise ValueError(f"Unknown message role: {msg.role}")
    
    def convert_msgs(self, msgs: List[Union[TextMessage, BinaryMessage]]) -> List[Union[ModelMessage, ModelResponse]]:
        return [self.convert_msg(msg) for msg in msgs]
        
    async def _run_stream_generator(
        self,
        model,
        prompt: List[Union[str, BinaryContent]],
        history: List[Union[ModelMessage, ModelResponse]],
        builtin_tools: List,
        mcp_servers: List
    ) -> AsyncIterator[AgentResponse]:
        """Internal generator that properly manages the streaming context"""
        async with self.agent:
            async with self.agent.run_stream(
                model=model,
                user_prompt=prompt,
                message_history=history,
                builtin_tools=builtin_tools,
                toolsets=mcp_servers
            ) as stream_response:
                async for text in stream_response.stream_output(debounce_by=0.1):
                    usage = stream_response.usage()
                    yield AgentResponse(
                        output=text,
                        usage=AgentResponseUsage(
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                        )
                    )
                # Final chunk with complete usage statistics
                usage = stream_response.usage()
                yield AgentResponse(
                    output="",
                    usage=AgentResponseUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )
                )
    
    async def _validated_stream_generator(
        self,
        model,
        prompt: List[Union[str, BinaryContent]],
        history: List[Union[ModelMessage, ModelResponse]],
        builtin_tools: List,
        mcp_servers: List
    ) -> AsyncIterator[AgentResponse]:
        """
        Validates first token generation before returning the stream.
        This ensures errors are caught early before the StreamingResponse is created.
        """
        generator = self._run_stream_generator(
            model=model,
            prompt=prompt,
            history=history,
            builtin_tools=builtin_tools,
            mcp_servers=mcp_servers
        )
        
        # Get the first chunk to validate the stream works
        first_chunk = await generator.__anext__()
        
        # If we got here, the stream is working, now yield everything
        yield first_chunk.model_dump_json() + "\n"
        async for chunk in generator:
            yield chunk.model_dump_json() + "\n"
    
    async def run(
        self,
        uid: str,
        payload: AgentRequest,
        stream: bool = False
    ) -> Union[AsyncIterator[AgentResponse], AgentResponse]:
        """
        Run an agent in stateless mode with message history.
        
        Args:
            settings: Provider settings (provider, API key, base URL)
            payload: Agent request payload (model, messages, tools, gen_config)
            stream: If True, returns an async generator for streaming responses
            
        Returns:
            If stream=False: AgentResponse with the complete output and usage
            If stream=True: AsyncIterator[AgentResponse] that yields chunks as they arrive
        """
        if not payload.messages:
            raise ValueError("Messages list cannot be empty")
        
        # Create the model using the cached provider
        model = self.create_model(
            uid=uid,
            model=payload.model,
            gen_config=payload.gen_config
        )
        
        # Create the user prompt from the last message
        prompt = payload.messages[-1]
        if prompt.role != MessageRole.USER:
            raise ValueError("Last message must be from user")
        converted_messages = self.convert_msgs(payload.messages)
        prompt = [part.content if isinstance(part, UserPromptPart) else part 
                for part in converted_messages[-1].parts]
        history = converted_messages[:-1]
        # Setup builtin tools
        builtin_tools, mcp_servers = self.create_tools(payload.tools)
        
        # Run the agent with all parameters
        if stream:
            # Return the validated generator that checks first token before streaming
            return self._validated_stream_generator(
                model=model,
                prompt=prompt,
                history=history,
                builtin_tools=builtin_tools,
                mcp_servers=mcp_servers
            )
        else:
            async with self.agent:
                result = await self.agent.run(
                    model=model,
                    user_prompt=prompt,
                    message_history=history,
                    builtin_tools=builtin_tools,
                    toolsets=mcp_servers
                )
                
                usage = result.usage()
                return AgentResponse(
                    output=result.output, 
                    usage=AgentResponseUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    ))


    


    