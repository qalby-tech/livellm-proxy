from fastapi import APIRouter, Depends, Request, Header
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from typing import Annotated, AsyncIterator, Optional
from managers.agent import AgentManager
from models.common import Settings, ProviderKind
from models.agent.run import AgentRequest, AgentResponse
import logfire

agent_router = APIRouter(prefix="/agent", tags=["agent"])

def get_agent_manager(request: Request) -> AgentManager:
    return request.app.state.agent_manager

AgentManagerType = Annotated[AgentManager, Depends(get_agent_manager)]

@agent_router.post("/run")
async def agent_run(
    payload: AgentRequest,
    agent_manager: AgentManagerType,
    x_provider_uid: str = Header(..., alias="X-Provider-UID", description="Provider UID configured via POST /config"),
) -> AgentResponse:
    """
    Run an agent request using the specified provider configuration.
    
    The provider UID must be configured first using POST /config endpoint.
    
    Headers:
        X-Provider-UID: The unique identifier of the provider configuration to use
    """
    try:
        result: AgentResponse = await agent_manager.run(x_provider_uid, payload, stream=False)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logfire.error(f"Unexpected error in agent_run: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@agent_router.post("/run_stream")
async def agent_run_stream(
    payload: AgentRequest,
    agent_manager: AgentManagerType,
    x_provider_uid: str = Header(..., alias="X-Provider-UID", description="Provider UID configured via POST /config"),
):
    """
    Stream agent responses as newline-delimited JSON (NDJSON).
    Each line is a JSON object with 'output' and 'usage' fields.
    
    The provider UID must be configured first using POST /config endpoint.
    
    Headers:
        X-Provider-UID: The unique identifier of the provider configuration to use
    """
    try:
        agent_stream: AsyncIterator[AgentResponse] = await agent_manager.run(x_provider_uid, payload, stream=True)
        return StreamingResponse(
            agent_stream,
            media_type="application/x-ndjson"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logfire.error(f"Unexpected error in agent_run_stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")