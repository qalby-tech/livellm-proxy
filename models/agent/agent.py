# models for full run: AgentRequest, AgentResponse

from pydantic import BaseModel, Field
from typing import Optional, List, Union, Any, Dict, Literal
from models.agent.chat import TextMessage, BinaryMessage
from models.agent.tools import WebSearchInput, MCPStreamableServerInput
from models.common import BaseRequest
from models.agent.chat import TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage


class JsonSchemaProperty(BaseModel):
    """JSON Schema property definition for nested properties."""
    type: Optional[Union[str, List[str]]] = Field(default=None, description="The type of the property (string, integer, number, boolean, array, object, null)")
    description: Optional[str] = Field(default=None, description="Description of the property")
    enum: Optional[List[Any]] = Field(default=None, description="Allowed values for the property")
    default: Optional[Any] = Field(default=None, description="Default value for the property")
    # String constraints
    min_length: Optional[int] = Field(default=None, alias="minLength", description="Minimum string length")
    max_length: Optional[int] = Field(default=None, alias="maxLength", description="Maximum string length")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for string validation")
    # Number constraints
    minimum: Optional[float] = Field(default=None, description="Minimum value for numbers")
    maximum: Optional[float] = Field(default=None, description="Maximum value for numbers")
    exclusive_minimum: Optional[float] = Field(default=None, alias="exclusiveMinimum", description="Exclusive minimum value")
    exclusive_maximum: Optional[float] = Field(default=None, alias="exclusiveMaximum", description="Exclusive maximum value")
    # Array constraints
    items: Optional[Union["JsonSchemaProperty", Dict[str, Any]]] = Field(default=None, description="Schema for array items")
    min_items: Optional[int] = Field(default=None, alias="minItems", description="Minimum number of array items")
    max_items: Optional[int] = Field(default=None, alias="maxItems", description="Maximum number of array items")
    unique_items: Optional[bool] = Field(default=None, alias="uniqueItems", description="Whether array items must be unique")
    # Object constraints (for nested objects)
    properties: Optional[Dict[str, Union["JsonSchemaProperty", Dict[str, Any]]]] = Field(default=None, description="Properties for nested objects")
    required: Optional[List[str]] = Field(default=None, description="Required properties for nested objects")
    additional_properties: Optional[Union[bool, "JsonSchemaProperty", Dict[str, Any]]] = Field(default=None, alias="additionalProperties", description="Schema for additional properties")

    model_config = {"populate_by_name": True}


class OutputSchema(BaseModel):
    """JSON Schema definition for structured output."""
    type: Literal["object"] = Field(default="object", description="The type of the schema (must be 'object' for structured output)")
    title: str = Field(..., description="Name of the schema, used as the output tool name")
    description: Optional[str] = Field(default=None, description="Description of the schema, helps the model understand what to output")
    properties: Dict[str, Union[JsonSchemaProperty, Dict[str, Any]]] = Field(..., description="Properties of the object")
    required: Optional[List[str]] = Field(default=None, description="List of required property names")
    additional_properties: Optional[Union[bool, JsonSchemaProperty, Dict[str, Any]]] = Field(default=None, alias="additionalProperties", description="Whether additional properties are allowed")

    model_config = {"populate_by_name": True}

    def to_json_schema(self) -> dict:
        """Convert to a JSON schema dict for use with StructuredDict."""
        return self.model_dump(by_alias=True, exclude_none=True)


class AgentRequest(BaseRequest):
    model: str = Field(..., description="The model to use")
    messages: List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]]
    tools: List[Union[WebSearchInput, MCPStreamableServerInput]]
    gen_config: Optional[dict] = Field(default=None, description="The configuration for the generation")
    include_history: bool = Field(default=False, description="Whether to include the history in the response")
    output_schema: Optional[OutputSchema] = Field(default=None, description="JSON schema for structured output. When provided, the agent will return a JSON string matching this schema.")


class AgentResponseUsage(BaseModel):
    input_tokens: int = Field(..., description="The number of input tokens used")
    output_tokens: int = Field(..., description="The number of output tokens used")

class AgentResponse(BaseModel):
    output: str = Field(..., description="The output of the response")
    usage: AgentResponseUsage = Field(..., description="The usage of the response")
    history: Optional[List[Union[TextMessage, BinaryMessage, ToolCallMessage, ToolReturnMessage]]] = Field(default=None, description="The history of the response")