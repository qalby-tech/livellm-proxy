"""Context overflow management for handling large text inputs."""

import tiktoken
import logfire
from typing import List, Optional, Callable, Awaitable, TypeVar
from models.agent.agent import ContextOverflowStrategy


T = TypeVar('T')

# Safety overhead multiplier for tiktoken token counting
# Tiktoken may undercount tokens for some models, so we add buffer
# Default is 20% (1.20), can be overridden via set_token_count_overhead()
_token_count_overhead: float = 1.20


def get_token_count_overhead() -> float:
    """Get the current token count overhead multiplier."""
    return _token_count_overhead


def set_token_count_overhead(overhead: float) -> None:
    """
    Set the token count overhead multiplier.
    
    Args:
        overhead: The overhead multiplier (e.g., 1.20 for 20% overhead).
                  Must be >= 1.0 (no overhead reduction allowed).
    """
    global _token_count_overhead
    if overhead < 1.0:
        raise ValueError(f"Token count overhead must be >= 1.0, got {overhead}")
    _token_count_overhead = overhead
    logfire.info(f"Token count overhead set to {overhead} ({(overhead - 1) * 100:.0f}% buffer)")


class Tokenizer:
    """Tokenizer wrapper using tiktoken for token counting and text splitting."""
    
    # Default encoding - cl100k_base is used by GPT-4, GPT-3.5-turbo
    # It's a good general-purpose encoding for most modern LLMs
    DEFAULT_ENCODING = "cl100k_base"
    
    def __init__(self, encoding_name: str = DEFAULT_ENCODING):
        """
        Initialize tokenizer with specified encoding.
        
        Args:
            encoding_name: The tiktoken encoding to use. 
                          Common options: "cl100k_base" (GPT-4), "o200k_base" (GPT-4o)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str, with_overhead: bool = True) -> int:
        """
        Count the number of tokens in the given text.
        
        Args:
            text: The text to count tokens for
            with_overhead: If True, applies 20% safety overhead to account for tiktoken inaccuracies
            
        Returns:
            Token count (with overhead if enabled)
        """
        raw_count = len(self.encoding.encode(text))
        if with_overhead:
            return int(raw_count * get_token_count_overhead())
        return raw_count
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.encoding.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.encoding.decode(tokens)
    
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens."""
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self.decode(tokens[:max_tokens])


class ContextOverflowManager:
    """
    Manages context overflow for agent requests.
    
    Supports two strategies:
    - TRUNCATE: Takes beginning, middle, and end portions of text to fit context limit
    - RECYCLE: Iteratively processes chunks and merges results
    
    System prompts are always preserved and excluded from truncation/chunking.
    """
    
    RECYCLE_SYSTEM_PROMPT_TEMPLATE = """You are a json generator tool.
<system_prompt>
{system_prompt}
</system_prompt>

The text might be too big to fit into your context window, so you need to iteratively update the response given new data.
<last_response>
{last_response}
</last_response>
Given new data, update the response according to the system prompt.
Updating might require inserting new information or updating existing information.
Return the updated response in JSON format strictly following the system prompt.
"""

    RECYCLE_TEXT_SYSTEM_PROMPT_TEMPLATE = """You are a text processing assistant.
<system_prompt>
{system_prompt}
</system_prompt>

The text might be too big to fit into your context window, so you need to iteratively process and merge the response given new data.
<last_response>
{last_response}
</last_response>
Given new data, update and merge the response according to the system prompt.
Combine the previous response with new information extracted from the new data.
Return the merged response as plain text.
"""
    
    def __init__(self, encoding_name: str = Tokenizer.DEFAULT_ENCODING):
        """
        Initialize context overflow manager.
        
        Args:
            encoding_name: The tiktoken encoding to use for token counting
        """
        self.tokenizer = Tokenizer(encoding_name)
    
    def validate_system_prompt(self, system_prompt: Optional[str], context_limit: int) -> int:
        """
        Validate that the system prompt fits within the context limit.
        
        Args:
            system_prompt: The system prompt to validate (can be None)
            context_limit: The total context limit in tokens
            
        Returns:
            Number of tokens available for content after reserving space for system prompt
            
        Raises:
            ValueError: If system prompt exceeds context limit
        """
        if not system_prompt:
            return context_limit
        
        system_prompt_tokens = self.tokenizer.count_tokens(system_prompt, with_overhead=True)
        
        if system_prompt_tokens >= context_limit:
            raise ValueError(
                f"System prompt ({system_prompt_tokens} tokens with 20% overhead) exceeds or equals "
                f"context limit ({context_limit} tokens). Cannot process request - system prompt must "
                f"be smaller than context limit to leave room for content."
            )
        
        available_for_content = context_limit - system_prompt_tokens
        logfire.info(
            f"Context overflow: system prompt uses {system_prompt_tokens} tokens (with overhead), "
            f"{available_for_content} tokens available for content"
        )
        return available_for_content
    
    def truncate_text(self, text: str, context_limit: int, system_prompt: Optional[str] = None) -> str:
        """
        Truncate text using start/middle/end strategy, preserving system prompt.
        
        If text exceeds available context (after reserving space for system prompt),
        takes approximately equal portions from:
        - Beginning of text
        - Middle of text  
        - End of text
        
        Args:
            text: The text to truncate
            context_limit: Maximum number of tokens allowed (total context)
            system_prompt: The system prompt to preserve (tokens reserved from context_limit)
            
        Returns:
            Truncated text fitting within available context
            
        Raises:
            ValueError: If system prompt exceeds context limit
        """
        # Validate and calculate available tokens for content
        available_tokens = self.validate_system_prompt(system_prompt, context_limit)
        
        # Get token count with overhead
        text_tokens_with_overhead = self.tokenizer.count_tokens(text, with_overhead=True)
        
        if text_tokens_with_overhead <= available_tokens:
            logfire.info(f"Context overflow: text fits within limit ({text_tokens_with_overhead} <= {available_tokens})")
            return text
        
        # Use raw token count for actual truncation (we apply overhead to the limit instead)
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        
        # Apply overhead reduction to available tokens for safety
        safe_available_tokens = int(available_tokens / get_token_count_overhead())
        
        # Calculate tokens per section (divide into 3 equal parts)
        # Leave small buffer for separator tokens
        separator = "\n\n[...content truncated...]\n\n"
        separator_tokens = self.tokenizer.count_tokens(separator, with_overhead=False) * 2  # Two separators
        truncation_available = safe_available_tokens - separator_tokens
        tokens_per_section = truncation_available // 3
        
        if tokens_per_section <= 0:
            raise ValueError(
                f"Not enough context available for content after system prompt. "
                f"Available: {available_tokens} tokens, need at least enough for 3 sections plus separators."
            )
        
        # Get start tokens
        start_tokens = tokens[:tokens_per_section]
        
        # Get middle tokens (from center of text)
        middle_start = (total_tokens - tokens_per_section) // 2
        middle_tokens = tokens[middle_start:middle_start + tokens_per_section]
        
        # Get end tokens
        end_tokens = tokens[-tokens_per_section:]
        
        # Decode sections back to text
        start_text = self.tokenizer.decode(start_tokens)
        middle_text = self.tokenizer.decode(middle_tokens)
        end_text = self.tokenizer.decode(end_tokens)
        
        truncated_text = f"{start_text}{separator}{middle_text}{separator}{end_text}"
        
        logfire.info(
            f"Context overflow: truncated text from {total_tokens} to ~{tokens_per_section * 3} tokens "
            f"(start: {len(start_tokens)}, middle: {len(middle_tokens)}, end: {len(end_tokens)}), "
            f"system prompt preserved ({self.tokenizer.count_tokens(system_prompt, with_overhead=True) if system_prompt else 0} tokens)"
        )
        
        return truncated_text
    
    def split_text_for_recycle(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks that fit within the specified token limit.
        
        Args:
            text: The text to split
            chunk_size: Maximum tokens per chunk (should already account for system prompt)
            
        Returns:
            List of text chunks, each fitting within chunk_size tokens
        """
        # Apply overhead reduction to chunk size for safety
        safe_chunk_size = int(chunk_size / get_token_count_overhead())
        
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        
        if total_tokens <= safe_chunk_size:
            return [text]
        
        chunks = []
        for i in range(0, total_tokens, safe_chunk_size):
            chunk_tokens = tokens[i:i + safe_chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        logfire.info(f"Context overflow: split text into {len(chunks)} chunks of ~{safe_chunk_size} tokens each (with overhead safety)")
        return chunks
    
    def get_recycle_system_prompt(self, original_system_prompt: str, last_response: str, is_structured: bool = True) -> str:
        """
        Generate the system prompt for subsequent recycle iterations.
        
        Args:
            original_system_prompt: The original system prompt from the request
            last_response: JSON string of the previous iteration's response (or plain text)
            is_structured: If True, uses JSON template; if False, uses plain text template
            
        Returns:
            System prompt for the next iteration with merge instructions
        """
        template = self.RECYCLE_SYSTEM_PROMPT_TEMPLATE if is_structured else self.RECYCLE_TEXT_SYSTEM_PROMPT_TEMPLATE
        return template.format(
            system_prompt=original_system_prompt,
            last_response=last_response
        )
    
    async def process_with_recycle(
        self,
        text: str,
        context_limit: int,
        system_prompt: str,
        executor: Callable[[str, str], Awaitable[str]],
        is_structured: bool = True,
    ) -> str:
        """
        Process text using the recycle strategy.
        
        Iteratively processes chunks of text, passing the previous response
        to be merged with new data in subsequent iterations.
        
        System prompt is preserved and excluded from chunking calculations.
        
        Args:
            text: The full text to process
            context_limit: Maximum tokens for total context
            system_prompt: The original system prompt (will be preserved)
            executor: Async function that takes (text_chunk, system_prompt) and returns response string
            is_structured: If True, uses JSON merging template; if False, uses plain text merging
            
        Returns:
            Final merged response after processing all chunks
            
        Raises:
            ValueError: If system prompt exceeds context limit
        """
        # Validate system prompt fits and get available tokens for content
        # For first chunk, we use original system prompt
        available_for_first_chunk = self.validate_system_prompt(system_prompt, context_limit)
        
        # For subsequent chunks, we use the recycle template which is larger
        # Estimate the recycle system prompt size with a reasonable last_response estimate
        template = self.RECYCLE_SYSTEM_PROMPT_TEMPLATE if is_structured else self.RECYCLE_TEXT_SYSTEM_PROMPT_TEMPLATE
        placeholder_response = '{"placeholder": "estimated response of moderate size for calculation"}' if is_structured else 'Estimated response of moderate size for calculation purposes.'
        recycle_prompt_estimate = template.format(
            system_prompt=system_prompt,
            last_response=placeholder_response
        )
        recycle_prompt_tokens = self.tokenizer.count_tokens(recycle_prompt_estimate, with_overhead=True)
        
        if recycle_prompt_tokens >= context_limit:
            raise ValueError(
                f"Recycle system prompt template ({recycle_prompt_tokens} tokens with overhead) exceeds "
                f"context limit ({context_limit} tokens). Cannot use recycle strategy - try truncate instead "
                f"or increase context_limit."
            )
        
        available_for_recycle_chunks = context_limit - recycle_prompt_tokens
        
        # Use the smaller of the two as chunk size to be safe
        # Also leave some buffer for response tokens
        response_buffer = 1000  # Reserve tokens for model response
        effective_chunk_size = min(available_for_first_chunk, available_for_recycle_chunks) - response_buffer
        
        if effective_chunk_size <= 0:
            raise ValueError(
                f"Not enough context available for content chunks after reserving space for system prompt "
                f"and response buffer. Available: {min(available_for_first_chunk, available_for_recycle_chunks)} tokens, "
                f"need at least {response_buffer} for response buffer."
            )
        
        logfire.info(
            f"Context overflow recycle: effective chunk size is {effective_chunk_size} tokens "
            f"(context_limit={context_limit}, system_prompt reserved, response_buffer={response_buffer})"
        )
        
        chunks = self.split_text_for_recycle(text, effective_chunk_size)
        
        logfire.info(f"Context overflow recycle: processing {len(chunks)} chunks")
        
        last_response: Optional[str] = None
        
        for i, chunk in enumerate(chunks):
            if last_response is None:
                # First chunk: use original system prompt
                current_system_prompt = system_prompt
            else:
                # Subsequent chunks: include previous response for merging
                current_system_prompt = self.get_recycle_system_prompt(system_prompt, last_response, is_structured=is_structured)
            
            logfire.info(f"Context overflow recycle: processing chunk {i + 1}/{len(chunks)}")
            last_response = await executor(chunk, current_system_prompt)
            logfire.info(f"Context overflow recycle: chunk {i + 1} processed")
        
        if last_response is None:
            raise ValueError("No response generated from recycle processing")
        
        return last_response
    
    def should_apply_overflow_handling(
        self, 
        text: str, 
        context_limit: int, 
        system_prompt: Optional[str] = None
    ) -> bool:
        """
        Check if context overflow handling is needed.
        
        Args:
            text: The text to check
            context_limit: The context limit (0 or negative means no limit)
            system_prompt: The system prompt (tokens will be reserved from context_limit)
            
        Returns:
            True if overflow handling should be applied
        """
        if context_limit <= 0:
            return False
        
        # Calculate available tokens after system prompt (with overhead)
        system_prompt_tokens = 0
        if system_prompt:
            system_prompt_tokens = self.tokenizer.count_tokens(system_prompt, with_overhead=True)
        
        available_for_content = context_limit - system_prompt_tokens
        
        # Check if text exceeds available space (with overhead)
        text_token_count = self.tokenizer.count_tokens(text, with_overhead=True)
        return text_token_count > available_for_content
    
    def get_text_token_count(self, text: str, with_overhead: bool = True) -> int:
        """
        Get the token count for the given text.
        
        Args:
            text: The text to count tokens for
            with_overhead: If True, applies 20% safety overhead
            
        Returns:
            Token count
        """
        return self.tokenizer.count_tokens(text, with_overhead=with_overhead)
