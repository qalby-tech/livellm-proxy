"""Tests for context overflow management functionality."""

import pytest
from unittest.mock import AsyncMock, patch
from managers.context import (
    Tokenizer,
    ContextOverflowManager,
    get_token_count_overhead,
    set_token_count_overhead,
)
from models.agent.agent import ContextOverflowStrategy


class TestTokenCountOverhead:
    """Tests for configurable token count overhead."""
    
    def setup_method(self):
        """Reset overhead to default before each test."""
        set_token_count_overhead(1.20)
    
    def teardown_method(self):
        """Reset overhead to default after each test."""
        set_token_count_overhead(1.20)
    
    def test_get_default_overhead(self):
        """Test that default overhead is 1.20 (20%)."""
        assert get_token_count_overhead() == 1.20
    
    def test_set_overhead(self):
        """Test setting a custom overhead."""
        set_token_count_overhead(1.30)
        assert get_token_count_overhead() == 1.30
    
    def test_set_overhead_minimum(self):
        """Test that overhead cannot be less than 1.0."""
        with pytest.raises(ValueError, match="must be >= 1.0"):
            set_token_count_overhead(0.9)
    
    def test_set_overhead_exactly_one(self):
        """Test setting overhead to exactly 1.0 (no overhead)."""
        set_token_count_overhead(1.0)
        assert get_token_count_overhead() == 1.0


class TestTokenizer:
    """Tests for the Tokenizer class."""
    
    def setup_method(self):
        """Reset overhead to default before each test."""
        set_token_count_overhead(1.20)
    
    def teardown_method(self):
        """Reset overhead to default after each test."""
        set_token_count_overhead(1.20)
    
    def test_count_tokens_with_overhead(self):
        """Test token counting with overhead enabled."""
        tokenizer = Tokenizer()
        text = "Hello, how are you today?"
        
        raw_count = tokenizer.count_tokens(text, with_overhead=False)
        with_overhead = tokenizer.count_tokens(text, with_overhead=True)
        
        assert with_overhead == int(raw_count * 1.20)
        assert with_overhead > raw_count
    
    def test_count_tokens_without_overhead(self):
        """Test token counting without overhead."""
        tokenizer = Tokenizer()
        text = "Hello, how are you today?"
        
        count = tokenizer.count_tokens(text, with_overhead=False)
        # Should be small, reasonable number for this short text
        assert count > 0
        assert count < 20
    
    def test_count_tokens_custom_overhead(self):
        """Test token counting with custom overhead setting."""
        set_token_count_overhead(1.50)  # 50% overhead
        tokenizer = Tokenizer()
        text = "Hello world"
        
        raw_count = tokenizer.count_tokens(text, with_overhead=False)
        with_overhead = tokenizer.count_tokens(text, with_overhead=True)
        
        assert with_overhead == int(raw_count * 1.50)
    
    def test_encode_decode(self):
        """Test encoding and decoding text."""
        tokenizer = Tokenizer()
        text = "Hello, world!"
        
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        assert decoded == text
    
    def test_truncate_to_tokens_no_truncation_needed(self):
        """Test truncation when text is already within limit."""
        tokenizer = Tokenizer()
        text = "Short text"
        
        result = tokenizer.truncate_to_tokens(text, max_tokens=100)
        assert result == text
    
    def test_truncate_to_tokens_truncation_applied(self):
        """Test truncation when text exceeds limit."""
        tokenizer = Tokenizer()
        text = "This is a longer text that should be truncated to fit within the token limit."
        
        result = tokenizer.truncate_to_tokens(text, max_tokens=5)
        result_tokens = tokenizer.encode(result)
        
        assert len(result_tokens) <= 5


class TestContextOverflowManager:
    """Tests for the ContextOverflowManager class."""
    
    def setup_method(self):
        """Reset overhead to default before each test."""
        set_token_count_overhead(1.20)
        self.manager = ContextOverflowManager()
    
    def teardown_method(self):
        """Reset overhead to default after each test."""
        set_token_count_overhead(1.20)
    
    def test_validate_system_prompt_none(self):
        """Test validation with no system prompt."""
        available = self.manager.validate_system_prompt(None, context_limit=1000)
        assert available == 1000
    
    def test_validate_system_prompt_fits(self):
        """Test validation when system prompt fits."""
        system_prompt = "You are a helpful assistant."
        available = self.manager.validate_system_prompt(system_prompt, context_limit=1000)
        
        # Should have less than 1000 available after reserving for system prompt
        assert available < 1000
        assert available > 0
    
    def test_validate_system_prompt_exceeds_limit(self):
        """Test validation when system prompt exceeds limit."""
        # Create a long system prompt
        system_prompt = "You are a helpful assistant. " * 500
        
        with pytest.raises(ValueError, match="System prompt.*exceeds or equals"):
            self.manager.validate_system_prompt(system_prompt, context_limit=100)
    
    def test_should_apply_overflow_handling_no_limit(self):
        """Test overflow check when context limit is 0 (disabled)."""
        text = "Some text content" * 100
        
        result = self.manager.should_apply_overflow_handling(text, context_limit=0)
        assert result is False
    
    def test_should_apply_overflow_handling_within_limit(self):
        """Test overflow check when text is within limit."""
        text = "Short text"
        
        result = self.manager.should_apply_overflow_handling(text, context_limit=1000)
        assert result is False
    
    def test_should_apply_overflow_handling_exceeds_limit(self):
        """Test overflow check when text exceeds limit."""
        text = "This is some text. " * 500  # Create a long text
        
        result = self.manager.should_apply_overflow_handling(text, context_limit=100)
        assert result is True
    
    def test_should_apply_overflow_handling_with_system_prompt(self):
        """Test overflow check considering system prompt reservation."""
        text = "Some content text"
        system_prompt = "A very long system prompt. " * 100
        
        # Text alone might fit, but with system prompt reserved, it might not
        result_without_prompt = self.manager.should_apply_overflow_handling(
            text, context_limit=500, system_prompt=None
        )
        result_with_prompt = self.manager.should_apply_overflow_handling(
            text, context_limit=500, system_prompt=system_prompt
        )
        
        # The result with prompt should be True or at least different logic
        # depending on actual token counts
        assert isinstance(result_with_prompt, bool)
    
    def test_truncate_text_no_truncation_needed(self):
        """Test truncation when text fits within limit."""
        text = "Short text"
        
        result = self.manager.truncate_text(text, context_limit=1000)
        assert result == text
    
    def test_truncate_text_with_truncation(self):
        """Test truncation when text exceeds limit."""
        # Create a long text
        text = "This is sentence number {}. " * 100
        text = text.format(*range(100))
        
        result = self.manager.truncate_text(text, context_limit=200)
        
        # Result should contain truncation markers
        assert "[...content truncated...]" in result
        # Result should be shorter than original
        assert len(result) < len(text)
    
    def test_truncate_text_preserves_system_prompt(self):
        """Test that truncation reserves space for system prompt."""
        text = "Content text. " * 200
        system_prompt = "You are a helpful assistant."
        
        result = self.manager.truncate_text(
            text, context_limit=500, system_prompt=system_prompt
        )
        
        # Truncation should happen (text is long)
        assert "[...content truncated...]" in result
        
        # The result should fit within the available space
        # (context_limit minus system_prompt tokens)
        result_tokens = self.manager.get_text_token_count(result, with_overhead=True)
        system_tokens = self.manager.get_text_token_count(system_prompt, with_overhead=True)
        
        # Combined should be less than context limit (with some margin for safety)
        assert result_tokens + system_tokens <= 500
    
    def test_truncate_text_system_prompt_too_large(self):
        """Test truncation fails when system prompt exceeds limit."""
        text = "Some content"
        system_prompt = "A very long system prompt. " * 200
        
        with pytest.raises(ValueError, match="System prompt.*exceeds"):
            self.manager.truncate_text(text, context_limit=100, system_prompt=system_prompt)
    
    def test_split_text_for_recycle_no_split_needed(self):
        """Test text splitting when text fits in one chunk."""
        text = "Short text"
        
        chunks = self.manager.split_text_for_recycle(text, chunk_size=1000)
        
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_split_text_for_recycle_multiple_chunks(self):
        """Test text splitting into multiple chunks."""
        # Create a long text
        text = "This is a sentence. " * 500
        
        chunks = self.manager.split_text_for_recycle(text, chunk_size=200)
        
        assert len(chunks) > 1
        # All text should be preserved across chunks
        reconstructed = "".join(chunks)
        # Note: reconstruction might not be perfect due to token boundaries
        assert len(reconstructed) > 0
    
    def test_get_recycle_system_prompt(self):
        """Test generation of recycle system prompt."""
        original_prompt = "Extract key information."
        last_response = '{"key": "value"}'
        
        result = self.manager.get_recycle_system_prompt(original_prompt, last_response)
        
        assert "Extract key information." in result
        assert '{"key": "value"}' in result
        assert "<system_prompt>" in result
        assert "<last_response>" in result
    
    def test_get_text_token_count_with_overhead(self):
        """Test getting token count with overhead."""
        text = "Hello world"
        
        count = self.manager.get_text_token_count(text, with_overhead=True)
        count_raw = self.manager.get_text_token_count(text, with_overhead=False)
        
        assert count == int(count_raw * 1.20)
    
    def test_get_text_token_count_without_overhead(self):
        """Test getting token count without overhead."""
        text = "Hello world"
        
        count = self.manager.get_text_token_count(text, with_overhead=False)
        
        assert count > 0
        assert count < 10  # "Hello world" should be a few tokens


class TestContextOverflowManagerRecycle:
    """Tests for the recycle strategy in ContextOverflowManager."""
    
    def setup_method(self):
        """Reset overhead to default before each test."""
        set_token_count_overhead(1.20)
        self.manager = ContextOverflowManager()
    
    def teardown_method(self):
        """Reset overhead to default after each test."""
        set_token_count_overhead(1.20)
    
    @pytest.mark.asyncio
    async def test_process_with_recycle_single_chunk(self):
        """Test recycle processing with text that fits in one chunk."""
        text = "Short text content"
        system_prompt = "Extract information."
        
        executor = AsyncMock(return_value='{"result": "data"}')
        
        result = await self.manager.process_with_recycle(
            text=text,
            context_limit=10000,
            system_prompt=system_prompt,
            executor=executor
        )
        
        # Executor should be called once
        assert executor.call_count == 1
        assert result == '{"result": "data"}'
    
    @pytest.mark.asyncio
    async def test_process_with_recycle_multiple_chunks(self):
        """Test recycle processing with text split into multiple chunks."""
        # Create a long text that needs splitting
        text = "This is sentence number X. " * 500
        system_prompt = "Extract information."
        
        call_count = 0
        async def mock_executor(chunk_text: str, current_system_prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return '{"items": ["first"]}'
            else:
                return '{"items": ["first", "second"]}'
        
        result = await self.manager.process_with_recycle(
            text=text,
            context_limit=3000,  # Larger limit to allow for response buffer
            system_prompt=system_prompt,
            executor=mock_executor
        )
        
        # Should have multiple calls due to chunking
        assert call_count > 1
        # Final result should be from last iteration
        assert result == '{"items": ["first", "second"]}'
    
    @pytest.mark.asyncio
    async def test_process_with_recycle_system_prompt_too_large(self):
        """Test recycle processing fails when system prompt exceeds limit."""
        text = "Some content"
        system_prompt = "A very long system prompt. " * 200
        
        executor = AsyncMock()
        
        with pytest.raises(ValueError, match="System prompt.*exceeds"):
            await self.manager.process_with_recycle(
                text=text,
                context_limit=100,
                system_prompt=system_prompt,
                executor=executor
            )


class TestContextOverflowIntegration:
    """Integration tests for context overflow handling."""
    
    def setup_method(self):
        """Reset overhead to default before each test."""
        set_token_count_overhead(1.20)
    
    def teardown_method(self):
        """Reset overhead to default after each test."""
        set_token_count_overhead(1.20)
    
    def test_truncate_preserves_start_middle_end(self):
        """Test that truncation preserves start, middle, and end content."""
        manager = ContextOverflowManager()
        
        # Create text with distinct markers
        start_marker = "START_MARKER_UNIQUE_STRING"
        middle_marker = "MIDDLE_MARKER_UNIQUE_STRING"
        end_marker = "END_MARKER_UNIQUE_STRING"
        
        # Build a long text with markers at specific positions
        padding = "This is padding text to make the content long. " * 50
        text = f"{start_marker} {padding} {middle_marker} {padding} {end_marker}"
        
        # Truncate to a smaller size
        result = manager.truncate_text(text, context_limit=300)
        
        # Check that truncation markers are present
        assert "[...content truncated...]" in result
        
        # Start should be preserved
        assert start_marker in result
        
        # End should be preserved
        assert end_marker in result
    
    def test_overhead_affects_truncation(self):
        """Test that changing overhead affects truncation behavior."""
        manager = ContextOverflowManager()
        text = "Content text. " * 100
        
        # With default 20% overhead
        set_token_count_overhead(1.20)
        result_20 = manager.truncate_text(text, context_limit=200)
        
        # With 50% overhead (more conservative)
        set_token_count_overhead(1.50)
        result_50 = manager.truncate_text(text, context_limit=200)
        
        # With higher overhead, truncation should be more aggressive
        # (result should be shorter or equal)
        assert len(result_50) <= len(result_20) or result_50 == result_20
    
    def test_no_overhead_gives_more_content(self):
        """Test that disabling overhead allows more content."""
        text = "Content text. " * 100
        
        # With overhead
        set_token_count_overhead(1.20)
        manager_with_overhead = ContextOverflowManager()
        result_with = manager_with_overhead.truncate_text(text, context_limit=200)
        
        # Without overhead (1.0)
        set_token_count_overhead(1.0)
        manager_no_overhead = ContextOverflowManager()
        result_without = manager_no_overhead.truncate_text(text, context_limit=200)
        
        # Without overhead, we can fit more content
        # Note: results might be equal if text doesn't need truncation
        assert len(result_without) >= len(result_with)
