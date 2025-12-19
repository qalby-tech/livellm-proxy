# models for full run: AgentRequest, AgentResponse

from pydantic import BaseModel, Field
from typing import Optional, List, Union
from models.agent.chat import TextMessage, BinaryMessage
from models.agent.tools import WebSearchInput, MCPStreamableServerInput
from models.common import BaseRequest
from models.agent.chat import TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage

class AgentRequest(BaseRequest):
    model: str = Field(..., description="The model to use")
    messages: List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]]
    tools: List[Union[WebSearchInput, MCPStreamableServerInput]]
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")
    include_history: bool = Field(default=False, description="Whether to include the history in the response")


class AgentResponseUsage(BaseModel):
    input_tokens: int = Field(..., description="The number of input tokens used")
    output_tokens: int = Field(..., description="The number of output tokens used")

class AgentResponse(BaseModel):
    output: str = Field(..., description="The output of the response")
    usage: AgentResponseUsage = Field(..., description="The usage of the response")
    history: Optional[List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]]] = Field(default=None, description="The history of the response")