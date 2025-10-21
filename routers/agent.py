from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from typing import Annotated
from managers.agent import AgentManager
from models.agent.run import AgentRequest, AgentResponse


agent_router = APIRouter(prefix="/agent", tags=["agent"])

def get_agent_manager(request: Request) -> AgentManager:
    return request.app.state.agent_manager

AgentManagerType = Annotated[AgentManager, Depends(get_agent_manager)]

@agent_router.post("/run")
async def agent_run(
    payload: AgentRequest,
    agent_manager: AgentManagerType
) -> AgentResponse:
    try:
        return await agent_manager.run(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
