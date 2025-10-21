from fastapi import APIRouter, Depends, Request, Header
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from typing import Annotated, AsyncIterator, Optional
from managers.agent import AgentManager
from models.common import Settings, ProviderKind
from models.agent.run import AgentRequest, AgentResponse
import json


agent_router = APIRouter(prefix="/agent", tags=["agent"])

def get_agent_manager(request: Request) -> AgentManager:
    return request.app.state.agent_manager

AgentManagerType = Annotated[AgentManager, Depends(get_agent_manager)]

@agent_router.post("/run")
async def agent_run(
    payload: AgentRequest,
    agent_manager: AgentManagerType,
    x_api_key: str = Header(..., description="API key for the provider"),
    x_provider: str = Header(..., description="Provider to use (openai, google, anthropic, groq)"),
    x_base_url: Optional[str] = Header(None, description="Optional custom base URL for the provider")
) -> AgentResponse:
    """
    Run an agent with the specified provider settings from headers.
    
    Headers:
        X-Api-Key: API key for the provider
        X-Provider: Provider to use (openai, google, anthropic, groq)
        X-Base-Url: Optional custom base URL for the provider
    """
    try:
        # Build settings from headers
        settings = Settings(
            provider=ProviderKind(x_provider.lower()),
            api_key=x_api_key,
            base_url=x_base_url
        )
        result: AgentResponse = await agent_manager.run(settings, payload, stream=False)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/run_stream")
async def agent_run_stream(
    payload: AgentRequest,
    agent_manager: AgentManagerType,
    x_api_key: str = Header(..., description="API key for the provider"),
    x_provider: str = Header(..., description="Provider to use (openai, google, anthropic, groq)"),
    x_base_url: Optional[str] = Header(None, description="Optional custom base URL for the provider")
):
    """
    Stream agent responses as newline-delimited JSON (NDJSON).
    Each line is a JSON object with 'output' and 'usage' fields.
    
    Headers:
        X-Api-Key: API key for the provider
        X-Provider: Provider to use (openai, google, anthropic, groq)
        X-Base-Url: Optional custom base URL for the provider
    """
    try:
        # Build settings from headers
        settings = Settings(
            provider=ProviderKind(x_provider.lower()),
            api_key=x_api_key,
            base_url=x_base_url
        )
        
        agent_stream: AsyncIterator[AgentResponse] = await agent_manager.run(settings, payload, stream=True)
 
        return StreamingResponse(
            agent_stream,
            media_type="application/x-ndjson"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))