from pydantic_ai import Agent, StructuredDict
from typing import Any, Optional, List, Union, Tuple, AsyncIterator
import json
import logfire

# tools
from pydantic_ai import WebSearchTool
from pydantic_ai.mcp import MCPServerStreamableHTTP

# messages
from pydantic_ai import BinaryContent, ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.messages import UserPromptPart, TextPart, SystemPromptPart, ToolCallPart, ToolReturnPart

# models
from pydantic_ai import ModelSettings
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider
# pydantic models
from models.common import ProviderKind, ContextOverflowStrategyType, FallbackStrategyType
from models.agent.tools import WebSearchInput, MCPStreamableServerInput, ToolKind, ToolInput
from models.agent.chat import MessageRole, TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage
from models.agent.agent import AgentRequest, AgentResponse, AgentResponseUsage, OutputSchema, ContextOverflowStrategy
from models.fallback import AgentFallbackRequest, FallbackStrategy
from managers.config import ConfigManager
from managers.fallback import FallbackManager
from managers.context import ContextOverflowManager
import base64


class AgentManager:
    

    def __init__(self, config_manager: ConfigManager, fallback_manager: Optional[FallbackManager] = None):
        self.config_manager = config_manager
        self.fallback_manager = fallback_manager
        self.agent = Agent[None, str]()
        self.context_manager = ContextOverflowManager()
        
    def create_model(self, uid: str, model: str, gen_config: Optional[dict] = None):
        """Create a model using the cached provider"""
        provider_kind, provider_client = self.config_manager.get_provider(uid, model)
        
        # Use empty dict if gen_config is None to avoid **None error
        config = gen_config or {}

        model_base: Model
        if provider_kind == ProviderKind.OPENAI:
            provider_client = OpenAIProvider(openai_client=provider_client)
            model_base = OpenAIResponsesModel
        elif provider_kind == ProviderKind.OPENAI_CHAT:
            provider_client = OpenAIProvider(openai_client=provider_client)
            model_base = OpenAIChatModel
        elif provider_kind == ProviderKind.GOOGLE:
            provider_client = GoogleProvider(client=provider_client)
            model_base = GoogleModel
        elif provider_kind == ProviderKind.ANTHROPIC:
            provider_client = AnthropicProvider(anthropic_client=provider_client)
            model_base = AnthropicModel
        elif provider_kind == ProviderKind.GROQ:
            provider_client = GroqProvider(groq_client=provider_client)
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
        elif msg.role == MessageRole.TOOL_CALL:
            parts.append(ToolCallPart(tool_name=msg.tool_name, args=msg.args))
            return ModelResponse(parts=parts)
        elif msg.role == MessageRole.TOOL_RETURN:
            parts.append(ToolReturnPart(tool_name=msg.tool_name, content=msg.content))
            return ModelRequest(parts=parts)
        else:
            raise ValueError(f"Unknown message role: {msg.role}")
    
    def convert_msgs(self, msgs: List[Union[TextMessage, BinaryMessage]]) -> List[Union[ModelMessage, ModelResponse]]:
        return [self.convert_msg(msg) for msg in msgs]
    
    def convert_history_to_msgs(self, history: List[Union[ModelRequest, ModelResponse]]) -> List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]]:
        msgs = []
        for msg in history:
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content[0] if isinstance(part.content, list) else part.content
                    msgs.append(TextMessage(role=MessageRole.USER, content=content))
                elif isinstance(part, TextPart):
                    content = part.content[0] if isinstance(part.content, list) else part.content
                    msgs.append(TextMessage(role=MessageRole.MODEL, content=content))
                elif isinstance(part, SystemPromptPart):
                    content = part.content[0] if isinstance(part.content, list) else part.content
                    msgs.append(TextMessage(role=MessageRole.SYSTEM, content=content))
                elif isinstance(part, ToolCallPart):
                    msgs.append(ToolCallMessage(
                        role=MessageRole.TOOL_CALL,
                        tool_name=part.tool_name,
                        args=part.args
                    ))
                elif isinstance(part, ToolReturnPart):
                    msgs.append(ToolReturnMessage(
                        role=MessageRole.TOOL_RETURN,
                        tool_name=part.tool_name,
                        content=part.content
                    ))
                elif isinstance(part, BinaryContent):
                    # Encode binary data back to base64 for the response
                    encoded_data = base64.b64encode(part.data).decode('utf-8')
                    msgs.append(BinaryMessage(
                        role=MessageRole.USER,
                        content=encoded_data,
                        mime_type=part.media_type
                    ))
        return msgs
    
    def _extract_text_from_prompt(self, prompt: List[Union[str, BinaryContent]]) -> str:
        """Extract all text content from a prompt list."""
        text_parts = []
        for item in prompt:
            if isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)
    
    def _apply_truncation_to_prompt(
        self, 
        prompt: List[Union[str, BinaryContent]], 
        context_limit: int,
        system_prompt: Optional[str] = None
    ) -> List[Union[str, BinaryContent]]:
        """
        Apply truncation to text content in the prompt while preserving binary content.
        System prompt is preserved and its tokens are excluded from the content limit.
        """
        # Separate text and binary content
        text_parts = []
        binary_parts = []
        for item in prompt:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, BinaryContent):
                binary_parts.append(item)
        
        # Combine and truncate text (system prompt tokens are reserved)
        combined_text = "\n".join(text_parts)
        truncated_text = self.context_manager.truncate_text(
            combined_text, context_limit, system_prompt=system_prompt
        )
        
        # Rebuild prompt with truncated text and original binary content
        new_prompt: List[Union[str, BinaryContent]] = [truncated_text]
        new_prompt.extend(binary_parts)
        return new_prompt
    
    def _extract_system_prompt(self, messages: List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]]) -> Optional[str]:
        """Extract the system prompt from the message list."""
        for msg in messages:
            if isinstance(msg, TextMessage) and msg.role == MessageRole.SYSTEM:
                return msg.content
        return None
    
    async def _run_with_recycle(
        self,
        payload: AgentRequest,
        model,
        prompt: List[Union[str, BinaryContent]],
        history: List[Union[ModelMessage, ModelResponse]],
        builtin_tools: List,
        mcp_servers: List,
        system_prompt: Optional[str] = None,
    ) -> AgentResponse:
        """
        Run agent with recycle strategy for context overflow.
        
        Iteratively processes chunks of text, passing previous results
        to be merged with new data. System prompt is preserved and excluded
        from chunking calculations.
        
        Supports both structured output (with output_schema) and plain text output.
        """
        is_structured = payload.output_schema is not None
        
        # Use provided system prompt or extract from messages
        if not system_prompt:
            system_prompt = self._extract_system_prompt(payload.messages)
        if not system_prompt:
            system_prompt = "Generate a structured response based on the provided data." if is_structured else "Process and summarize the provided data."
        
        # Extract text content from prompt
        text_content = self._extract_text_from_prompt(prompt)
        
        # Check if overflow handling is needed (considering system prompt)
        if not self.context_manager.should_apply_overflow_handling(
            text_content, payload.context_limit, system_prompt=system_prompt
        ):
            # No overflow, run normally
            return await self._run_non_stream(payload, model, prompt, history, builtin_tools, mcp_servers)
        
        logfire.info(f"Context overflow detected, using recycle strategy (structured={is_structured}) for provider={payload.provider_uid}, model={payload.model}")
        
        total_input_tokens = 0
        total_output_tokens = 0
        
        if is_structured:
            # Prepare the schema for structured output
            schema_dict = payload.output_schema.to_json_schema()
            output_type = StructuredDict(
                schema_dict,
                name=payload.output_schema.title,
                description=payload.output_schema.description
            )
            
            # Define executor function for structured recycle processing
            async def chunk_executor(chunk_text: str, current_system_prompt: str) -> str:
                nonlocal total_input_tokens, total_output_tokens
                
                # Create messages with the current system prompt and chunk text
                structured_agent: Agent[None, dict[str, Any]] = Agent(output_type=output_type)
                
                # Build history with the current system prompt
                chunk_history = [
                    ModelRequest(parts=[SystemPromptPart(content=current_system_prompt)])
                ]
                chunk_history.extend(history)  # Add any existing conversation history
                
                async with structured_agent:
                    result = await structured_agent.run(
                        model=model,
                        user_prompt=[chunk_text],
                        message_history=chunk_history,
                        builtin_tools=builtin_tools,
                        toolsets=mcp_servers
                    )
                    
                    usage = result.usage()
                    total_input_tokens += usage.input_tokens
                    total_output_tokens += usage.output_tokens
                    
                    return json.dumps(result.output, ensure_ascii=False)
        else:
            # Define executor function for plain text recycle processing
            async def chunk_executor(chunk_text: str, current_system_prompt: str) -> str:
                nonlocal total_input_tokens, total_output_tokens
                
                # Use plain text agent
                text_agent: Agent[None, str] = Agent()
                
                # Build history with the current system prompt
                chunk_history = [
                    ModelRequest(parts=[SystemPromptPart(content=current_system_prompt)])
                ]
                chunk_history.extend(history)  # Add any existing conversation history
                
                async with text_agent:
                    result = await text_agent.run(
                        model=model,
                        user_prompt=[chunk_text],
                        message_history=chunk_history,
                        builtin_tools=builtin_tools,
                        toolsets=mcp_servers
                    )
                    
                    usage = result.usage()
                    total_input_tokens += usage.input_tokens
                    total_output_tokens += usage.output_tokens
                    
                    return result.output
        
        # Process with recycle strategy
        final_response = await self.context_manager.process_with_recycle(
            text=text_content,
            context_limit=payload.context_limit,
            system_prompt=system_prompt,
            executor=chunk_executor,
            is_structured=is_structured
        )
        
        logfire.info(f"Request succeeded (recycle) for provider={payload.provider_uid}, model={payload.model}, total_input_tokens={total_input_tokens}, total_output_tokens={total_output_tokens}")
        
        return AgentResponse(
            output=final_response,
            usage=AgentResponseUsage(
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
            ),
            history=None  # History not supported in recycle mode
        )
    
    async def _run_non_stream(
        self,
        payload: AgentRequest,
        model,
        prompt: List[Union[str, BinaryContent]],
        history: List[Union[ModelMessage, ModelResponse]],
        builtin_tools: List,
        mcp_servers: List,
    ) -> AgentResponse:
        """
        Run agent without streaming (extracted for reuse).
        """
        if payload.output_schema:
            schema_dict = payload.output_schema.to_json_schema()
            output_type = StructuredDict(
                schema_dict,
                name=payload.output_schema.title,
                description=payload.output_schema.description
            )
            structured_agent: Agent[None, dict[str, Any]] = Agent(output_type=output_type)
            async with structured_agent:
                result = await structured_agent.run(
                    model=model,
                    user_prompt=prompt,
                    message_history=history,
                    builtin_tools=builtin_tools,
                    toolsets=mcp_servers
                )
                
                usage = result.usage()
                history_msgs = self.convert_history_to_msgs(result.all_messages()) if payload.include_history else None
                output_json = json.dumps(result.output, ensure_ascii=False)
                logfire.info(f"Request succeeded for provider={payload.provider_uid}, model={payload.model}, input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}")
                return AgentResponse(
                    output=output_json, 
                    usage=AgentResponseUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    ),
                    history=history_msgs
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
                history_msgs = self.convert_history_to_msgs(result.all_messages()) if payload.include_history else None
                logfire.info(f"Request succeeded for provider={payload.provider_uid}, model={payload.model}, input_tokens={usage.input_tokens}, output_tokens={usage.output_tokens}")
                return AgentResponse(
                    output=result.output, 
                    usage=AgentResponseUsage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    ),
                    history=history_msgs
                )
        
    async def _run_stream_generator(
        self,
        model,
        prompt: List[Union[str, BinaryContent]],
        history: List[Union[ModelMessage, ModelResponse]],
        builtin_tools: List,
        mcp_servers: List,
        include_history: bool = False,
        output_schema: Optional[OutputSchema] = None
    ) -> AsyncIterator[AgentResponse]:
        """Internal generator that properly manages the streaming context"""
        # Handle structured output with custom JSON schema
        if output_schema:
            schema_dict = output_schema.to_json_schema()
            output_type = StructuredDict(
                schema_dict,
                name=output_schema.title,
                description=output_schema.description
            )
            structured_agent: Agent[None, dict[str, Any]] = Agent(output_type=output_type)
            async with structured_agent:
                async with structured_agent.run_stream(
                    model=model,
                    user_prompt=prompt,
                    message_history=history,
                    builtin_tools=builtin_tools,
                    toolsets=mcp_servers
                ) as stream_response:
                    async for output_dict in stream_response.stream_output(debounce_by=0.05):
                        usage = stream_response.usage()
                        # Convert dict output to JSON string
                        output_json = json.dumps(output_dict, ensure_ascii=False) if output_dict else ""
                        yield AgentResponse(
                            output=output_json,
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
                        ),
                        history=self.convert_history_to_msgs(stream_response.all_messages()) if include_history else None
                    )
        else:
            async with self.agent:
                async with self.agent.run_stream(
                    model=model,
                    user_prompt=prompt,
                    message_history=history,
                    builtin_tools=builtin_tools,
                    toolsets=mcp_servers
                ) as stream_response:
                    async for text in stream_response.stream_output(debounce_by=None):
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
                        ),
                        history=self.convert_history_to_msgs(stream_response.all_messages()) if include_history else None
                    )
    
    async def _validated_stream_generator(
        self,
        model,
        prompt: List[Union[str, BinaryContent]],
        history: List[Union[ModelMessage, ModelResponse]],
        builtin_tools: List,
        mcp_servers: List,
        include_history: bool = False,
        output_schema: Optional[OutputSchema] = None
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
            mcp_servers=mcp_servers,
            include_history=include_history,
            output_schema=output_schema
        )
        
        # Get the first chunk to validate the stream works
        first_chunk = await generator.__anext__()
        
        # If we got here, the stream is working, now yield everything
        yield first_chunk
        async for chunk in generator:
            yield chunk
    
    def _get_effective_context_config(
        self, 
        payload: AgentRequest
    ) -> Tuple[int, ContextOverflowStrategy]:
        """
        Get effective context limit and strategy, prioritizing user request over config.
        
        User-specified values in the request take priority over provider model config.
        
        Args:
            payload: The agent request
            
        Returns:
            Tuple of (context_limit, context_overflow_strategy)
        """
        # Get model-specific config from provider settings
        model_config = self.config_manager.get_model_config(payload.provider_uid, payload.model)
        
        # Determine effective context limit
        # User's request takes priority (>0 means user specified it)
        if payload.context_limit > 0:
            context_limit = payload.context_limit
        elif model_config and model_config.context_limit > 0:
            context_limit = model_config.context_limit
        else:
            context_limit = 0
        
        # Determine effective overflow strategy
        # User's request takes priority (non-default means user specified it)
        if payload.context_overflow_strategy != ContextOverflowStrategy.TRUNCATE:
            # User explicitly set a non-default strategy
            strategy = payload.context_overflow_strategy
        elif model_config:
            # Convert config strategy to request strategy type
            if model_config.context_overflow_strategy == ContextOverflowStrategyType.RECYCLE:
                strategy = ContextOverflowStrategy.RECYCLE
            else:
                strategy = ContextOverflowStrategy.TRUNCATE
        else:
            strategy = payload.context_overflow_strategy  # Use default from request
        
        return context_limit, strategy
    
    def _get_fallback_config(self, payload: AgentRequest) -> Optional[Tuple[str, str, FallbackStrategyType, int, ContextOverflowStrategy]]:
        """
        Get fallback configuration for the model if configured.
        
        Args:
            payload: The agent request
            
        Returns:
            Tuple of (fallback_provider_uid, fallback_model, fallback_strategy, context_limit, context_overflow_strategy) if configured, None otherwise
        """
        model_config = self.config_manager.get_model_config(payload.provider_uid, payload.model)
        if model_config and model_config.fallback:
            # Convert context overflow strategy type
            if model_config.fallback.context_overflow_strategy == ContextOverflowStrategyType.RECYCLE:
                overflow_strategy = ContextOverflowStrategy.RECYCLE
            else:
                overflow_strategy = ContextOverflowStrategy.TRUNCATE
            
            return (
                model_config.fallback.fallback_provider_uid,
                model_config.fallback.fallback_model,
                model_config.fallback.fallback_strategy,
                model_config.fallback.context_limit,
                overflow_strategy,
            )
        return None

    async def run(
        self,
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
        
        # Get effective context config (user request takes priority over provider config)
        context_limit, context_overflow_strategy = self._get_effective_context_config(payload)
        
        # Create the model using the cached provider
        model = self.create_model(
            uid=payload.provider_uid,
            model=payload.model,
            gen_config=payload.gen_config
        )
        
        # Validate that last message is from user
        if payload.messages[-1].role != MessageRole.USER:
            raise ValueError("Last message must be from user")
        
        # Convert all messages
        converted_messages = self.convert_msgs(payload.messages)
        
        # Find all consecutive user messages from the end
        # User messages are ModelRequest with UserPromptPart or BinaryContent parts
        user_messages = []
        for msg in reversed(converted_messages):
            if isinstance(msg, ModelRequest) and any(
                isinstance(p, (UserPromptPart, BinaryContent)) for p in msg.parts
            ):
                user_messages.insert(0, msg)
            else:
                break
        
        if not user_messages:
            raise ValueError("No user messages found")
        
        # Combine all user message parts into a single prompt list
        # This ensures BinaryContent stays in user_prompt, not in message_history
        prompt = []
        for msg in user_messages:
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    prompt.append(part.content)
                elif isinstance(part, BinaryContent):
                    prompt.append(part)
        
        # History is everything except the user messages we just extracted
        history = converted_messages[:-len(user_messages)]
        # Setup builtin tools
        builtin_tools, mcp_servers = self.create_tools(payload.tools)
        
        # Handle context overflow if context_limit is set
        if context_limit > 0:
            # Extract system prompt to preserve it during overflow handling
            system_prompt = self._extract_system_prompt(payload.messages)
            
            text_content = self._extract_text_from_prompt(prompt)
            needs_overflow_handling = self.context_manager.should_apply_overflow_handling(
                text_content, context_limit, system_prompt=system_prompt
            )
            
            if needs_overflow_handling:
                if context_overflow_strategy == ContextOverflowStrategy.RECYCLE:
                    # Recycle strategy: iteratively process chunks (non-streaming only)
                    if stream:
                        raise ValueError("Recycle strategy is not supported with streaming")
                    return await self._run_with_recycle(
                        payload=payload,
                        model=model,
                        prompt=prompt,
                        history=history,
                        builtin_tools=builtin_tools,
                        mcp_servers=mcp_servers,
                        system_prompt=system_prompt
                    )
                else:
                    # Truncate strategy: apply truncation to prompt (system prompt preserved)
                    logfire.info(f"Context overflow detected, using truncate strategy for provider={payload.provider_uid}, model={payload.model}")
                    prompt = self._apply_truncation_to_prompt(
                        prompt, context_limit, system_prompt=system_prompt
                    )
        
        # Run the agent with all parameters
        if stream:
            # Return the validated generator that checks first token before streaming
            return self._validated_stream_generator(
                model=model,
                prompt=prompt,
                history=history,
                builtin_tools=builtin_tools,
                mcp_servers=mcp_servers,
                include_history=payload.include_history,
                output_schema=payload.output_schema
            )
        else:
            # Use extracted method for non-streaming execution
            return await self._run_non_stream(
                payload=payload,
                model=model,
                prompt=prompt,
                history=history,
                builtin_tools=builtin_tools,
                mcp_servers=mcp_servers
            )

    async def safe_run(
        self,
        payload: Union[AgentRequest, AgentFallbackRequest],
        stream: bool = False
    ) -> Union[AsyncIterator[AgentResponse], AgentResponse]:
        """
        Run agent with optional fallback support.
        
        If payload is AgentRequest: runs normally, with automatic fallback from config if available
        If payload is AgentFallbackRequest: uses fallback manager to try multiple requests
            (user explicitly provided fallback list - takes priority over config)
        
        Args:
            payload: Either AgentRequest or AgentFallbackRequest
            stream: If True, returns an async generator for streaming responses
            
        Returns:
            If stream=False: AgentResponse with the complete output and usage
            If stream=True: AsyncIterator[AgentResponse] that yields chunks as they arrive
        """
        # If it's a fallback request (user explicitly specified fallbacks), use them directly
        if isinstance(payload, AgentFallbackRequest):
            if not self.fallback_manager:
                raise ValueError("FallbackManager not configured. Cannot use fallback requests.")
            
            # Define the executor function for the fallback manager
            async def executor(request: AgentRequest) -> Union[AsyncIterator[AgentResponse], AgentResponse]:
                return await self.run(request, stream=stream)
            
            # Use the fallback manager's catch method
            return await self.fallback_manager.catch(payload, executor)
        
        # For simple AgentRequest, try with automatic fallback from config if available
        fallback_config = self._get_fallback_config(payload)
        
        if not fallback_config:
            # No fallback configured, run normally
            return await self.run(payload, stream=stream)
        
        # We have fallback config - try primary first, then fallback on failure
        if not self.fallback_manager:
            # No fallback manager, just try primary
            return await self.run(payload, stream=stream)
        
        # Build fallback request list: primary first, then fallback from config
        fallback_provider_uid, fallback_model, fallback_strategy, fb_context_limit, fb_context_overflow_strategy = fallback_config
        
        # Create the fallback request with same messages/tools but different provider/model
        # Use the context settings from the fallback config
        fallback_request = AgentRequest(
            provider_uid=fallback_provider_uid,
            model=fallback_model,
            messages=payload.messages,
            tools=payload.tools,
            gen_config=payload.gen_config,
            include_history=payload.include_history,
            output_schema=payload.output_schema,
            context_limit=fb_context_limit,
            context_overflow_strategy=fb_context_overflow_strategy
        )
        
        # Map the config strategy type to the fallback model strategy enum
        if fallback_strategy == FallbackStrategyType.PARALLEL:
            fb_strategy = FallbackStrategy.PARALLEL
        else:
            fb_strategy = FallbackStrategy.SEQUENTIAL
        
        # Create fallback request with primary first, then configured fallback
        fallback_payload = AgentFallbackRequest(
            requests=[payload, fallback_request],
            strategy=fb_strategy,
            timeout_per_request=360
        )
        
        logfire.info(
            f"Using automatic fallback from config: {payload.provider_uid}/{payload.model} -> "
            f"{fallback_provider_uid}/{fallback_model} (strategy={fb_strategy.value})"
        )
        
        # Define the executor function for the fallback manager
        async def executor(request: AgentRequest) -> Union[AsyncIterator[AgentResponse], AgentResponse]:
            return await self.run(request, stream=stream)
        
        # Use the fallback manager's catch method
        return await self.fallback_manager.catch(fallback_payload, executor)


    


    