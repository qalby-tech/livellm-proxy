# models for chat messages
from pydantic import BaseModel, Field, model_validator
from enum import Enum
from typing import Optional, Any

class MessageRole(str, Enum):
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RETURN = "tool_return"


class Message(BaseModel):
    role: MessageRole = Field(..., description="The role of the message")


class TextMessage(Message):
    content: str = Field(..., description="The content of the message")

class BinaryMessage(Message):
    """always from user"""
    content: str = Field(..., description="The base64 encoded content of the message")
    mime_type: str = Field(..., description="The MIME type of the content, only user can supply such")
    caption: Optional[str] = Field(None, description="Caption for the binary message")

    @model_validator(mode="after")
    def validate_content(self) -> "BinaryMessage":
        if self.role == MessageRole.MODEL:
            raise ValueError("MIME type are meant for user messages only")
        return self

class ToolMessage(Message):
    tool_name: str = Field(..., description="The name of the tool")

class ToolCallMessage(ToolMessage):
    args: str | dict[str, Any] = Field(..., description="The arguments of the tool call")

class ToolReturnMessage(ToolMessage):
    content: Any = Field(..., description="The result of the tool call")

