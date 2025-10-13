from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum


class MessageRole(Enum):
    user = "user"
    assistant = "assistant"
    system = "system"

class Message(BaseModel):
    role: MessageRole = Field(default=MessageRole.user)
    content: str = Field(description="The content of the message; base64 encoded if not text")


class TextMessage(Message):
    pass

class MediaMessage(Message):
    mime_type: str = Field(description="The MIME type of the media")


class MCPTool(BaseModel):
    name: str = Field(description="The name of the tool")
    server_url: str = Field(description="The URL of the MCP server")

class WebSearchTool(BaseModel):
    web_search: dict = Field(description="arguments for the web search tools")
