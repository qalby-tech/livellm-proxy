from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from typing import Annotated, AsyncIterator
from managers.agent import AgentManager
from models.agent.run import AgentRequest, AgentResponse
import json


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
        result: AgentResponse = await agent_manager.run(payload, stream=False)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/run_stream")
async def agent_run_stream(
    payload: AgentRequest,
    agent_manager: AgentManagerType
):
    """
    Stream agent responses as newline-delimited JSON (NDJSON).
    Each line is a JSON object with 'output' and 'usage' fields.
    """
    try:
        async def _stream_generator():
            async for chunk in agent_manager.run(payload, stream=True):
                # Convert Pydantic model to JSON and add newline for streaming
                data: AgentResponse = chunk
                # yield json.dumps(data.model_dump()) + "\n"
                yield data
        return StreamingResponse(
            _stream_generator(),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))