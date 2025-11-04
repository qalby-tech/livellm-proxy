from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from typing import Annotated, AsyncIterator, Union
from managers.agent import AgentManager
from models.agent.agent import AgentRequest, AgentResponse
from models.fallback import AgentFallbackRequest
import logfire

agent_router = APIRouter(prefix="/agent", tags=["agent"])

def get_agent_manager(request: Request) -> AgentManager:
    return request.app.state.agent_manager

AgentManagerType = Annotated[AgentManager, Depends(get_agent_manager)]

@agent_router.post("/run")
async def agent_run(
    payload: Union[AgentRequest, AgentFallbackRequest],
    agent_manager: AgentManagerType,
) -> AgentResponse:
    """
    Run an agent request using the specified provider configuration.
    Supports both single requests and fallback requests with multiple providers.
    
    For single request: provide AgentRequest
    For fallback: provide AgentFallbackRequest with list of requests and strategy
    
    The provider UID must be configured first using POST /config endpoint.
    """
    try:
        result: AgentResponse = await agent_manager.safe_run(payload, stream=False)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logfire.error(f"Unexpected error in agent_run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@agent_router.post("/run_stream")
async def agent_run_stream(
    payload: Union[AgentRequest, AgentFallbackRequest],
    agent_manager: AgentManagerType,
):
    """
    Stream agent responses as newline-delimited JSON (NDJSON).
    Each line is a JSON object with 'output' and 'usage' fields.
    Supports both single requests and fallback requests with multiple providers.
    
    For single request: provide AgentRequest
    For fallback: provide AgentFallbackRequest with list of requests and strategy
    
    The provider UID must be configured first using POST /config endpoint.
    """
    try:
        agent_stream: AsyncIterator[AgentResponse] = await agent_manager.safe_run(payload, stream=True)
        return StreamingResponse(
            agent_stream,
            media_type="application/x-ndjson"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logfire.error(f"Unexpected error in agent_run_stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")