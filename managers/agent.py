from pydantic_ai import Agent
from typing import Optional, List, Dict, Union, Tuple, AsyncIterator
import logfire

# tools
from pydantic_ai import WebSearchTool
from pydantic_ai.mcp import MCPServerStreamableHTTP

# messages
from pydantic_ai import BinaryContent, ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.messages import UserPromptPart, TextPart, SystemPromptPart

# openai
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
# google
from google import genai
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.google import GoogleModelSettings
# anthropic
from anthropic import AsyncAnthropic
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.anthropic import AnthropicModelSettings
# groq
from groq import AsyncGroq
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.groq import GroqModelSettings

# pydantic models
from models.common import Settings, ProviderKind
from models.agent.tools import WebSearchInput, MCPStreamableServerInput, ToolKind, ToolInput
from models.agent.chat import MessageRole, TextMessage, BinaryMessage
from models.agent.run import AgentRequest, AgentResponse, AgentResponseUsage
import base64


class AgentManager:
    

    def __init__(self):
        self.providers: Dict[str, Union[OpenAIProvider, GoogleProvider, AnthropicProvider, GroqProvider]] = {}
        self.agent = Agent()
    
    def _get_provider_cache_key(self, provider: ProviderKind, api_key: str, base_url: Optional[str] = None) -> str:
        """Generate a unique cache key for a provider configuration"""
        base = base_url or "default"
        return f"{provider.value}:{api_key}:{base}"
    
    def create_provider(
        self, 
        provider: ProviderKind, 
        api_key: str, 
        base_url: Optional[str] = None
    ) -> Union[OpenAIProvider, GoogleProvider, AnthropicProvider, GroqProvider]:
        """
        Create or retrieve a cached provider instance.
        If a provider with the same configuration already exists, return it.
        Otherwise, create a new one and cache it.
        """
        cache_key = self._get_provider_cache_key(provider, api_key, base_url)
        
        # Check if provider already exists in cache
        if cache_key in self.providers:
            return self.providers[cache_key]
        
        # Create new provider based on type
        if provider == ProviderKind.OPENAI:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            new_provider = OpenAIProvider(openai_client=client)
        elif provider == ProviderKind.GOOGLE:
            client = genai.Client(api_key=api_key, http_options=genai.types.HttpOptions(base_url=base_url))
            new_provider = GoogleProvider(client=client)
        elif provider == ProviderKind.ANTHROPIC:
            client = AsyncAnthropic(api_key=api_key, base_url=base_url)
            new_provider = AnthropicProvider(anthropic_client=client)
        elif provider == ProviderKind.GROQ:
            client = AsyncGroq(api_key=api_key, base_url=base_url)
            new_provider = GroqProvider(groq_client=client)
        else:
            raise ValueError(f"Provider {provider} not supported")
        
        # Cache and return the new provider
        self.providers[cache_key] = new_provider
        return new_provider
        
    def create_model(self, provider: ProviderKind, model: str, api_key: str, base_url: Optional[str] = None, gen_config: Optional[dict] = None):
        """Create a model using the cached provider"""
        provider_instance = self.create_provider(provider, api_key, base_url)
        
        # Use empty dict if gen_config is None to avoid **None error
        config = gen_config or {}
        
        if provider == ProviderKind.OPENAI:
            return OpenAIResponsesModel(
                model_name=model, 
                provider=provider_instance,
                settings=OpenAIResponsesModelSettings(
                    **config
                )
            )
        elif provider == ProviderKind.GOOGLE:
            return GoogleModel(
                model_name=model, 
                provider=provider_instance,
                settings=GoogleModelSettings(
                    **config
                )
            )
        elif provider == ProviderKind.ANTHROPIC:
            return AnthropicModel(
                model_name=model, 
                provider=provider_instance,
                settings=AnthropicModelSettings(
                    **config
                )
            )
        elif provider == ProviderKind.GROQ:
            return GroqModel(
                model_name=model, 
                provider=provider_instance,
                settings=GroqModelSettings(
                    **config
                )
            )
        else:
            raise ValueError(f"Provider {provider} not supported")
    
    
    def create_tools(self, tools: List[ToolInput]) -> Tuple[List[WebSearchTool], List[MCPServerStreamableHTTP]]:
        builtin_tools = []
        mcp_servers = []
        init_cache = set() # avoid duplicate tools
        for tool in tools:
            if isinstance(tool, WebSearchInput):
                if ToolKind.WEB_SEARCH not in init_cache:
                    # using tool's unique key to deduplicate
                    builtin_tools.append(WebSearchTool(search_context_size=tool.search_context_size))
                    init_cache.add(ToolKind.WEB_SEARCH)
            elif isinstance(tool, MCPStreamableServerInput):
                if tool.url not in init_cache:
                    # using url to deduplicate mcp
                    mcp_servers.append(MCPServerStreamableHTTP(url=tool.url, tool_prefix=tool.prefix))
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
        settings: Settings,
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
            provider=settings.provider,
            model=payload.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
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


    


    