from pydantic import BaseModel, Field
from models.inputs import TextMessage, MediaMessage, WebSearchTool, MCPTool
from typing import Literal, Union, Optional

class ModelParams(BaseModel):
    model: str = Field(description="The model to use for the request")

class SpeakRequest(ModelParams):
    text: str = Field(description="The text to speak")
    voice: str = Field(description="The voice to use for the request")
    response_format: str = Field(default="mp3", description="The audio format for the response")

class ChatRequest(ModelParams):
    messages: list[TextMessage | MediaMessage] = Field(description="The messages to run")
    tools: Optional[list[Union[WebSearchTool, MCPTool]]] = Field(default=None, description="The tools to use for the request")
    tool_choice: Literal["auto", "none", "required"] = Field(default="auto", description="The tool choice for the request")