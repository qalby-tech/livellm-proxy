# models for full run: AgentRequest, AgentResponse

from pydantic import BaseModel, Field
from typing import Optional, List, Union
from models.agent.chat import TextMessage, BinaryMessage
from models.agent.tools import WebSearchInput, MCPStreamableServerInput


class AgentRequest(BaseModel):
    model: str = Field(..., description="The model to use")
    messages: List[Union[TextMessage, BinaryMessage]]
    tools: List[Union[WebSearchInput, MCPStreamableServerInput]]
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")


class AgentResponseUsage(BaseModel):
    input_tokens: int = Field(..., description="The number of input tokens used")
    output_tokens: int = Field(..., description="The number of output tokens used")

class AgentResponse(BaseModel):
    output: str = Field(..., description="The output of the response")
    usage: AgentResponseUsage = Field(..., description="The usage of the response")