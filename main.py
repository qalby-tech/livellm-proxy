import os
from fastapi import FastAPI, Depends
from fastapi.exceptions import HTTPException
from contextlib import asynccontextmanager
from typing import Annotated
from agent import AgentManager
from models.run import AgentRequest, AgentResponse
import logfire

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent_manager = AgentManager()
    yield


app = FastAPI(lifespan=lifespan)

# configure logfire
logfire_token = os.getenv('LOGFIRE_WRITE_TOKEN', None)
otel_exporter = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')
logfire.configure(send_to_logfire="if-token-present", token=logfire_token)
logfire.instrument_pydantic_ai()
logfire.instrument_mcp()
logfire.instrument_fastapi(app)

@app.get("/ping")
async def ping():
    return {"status": "ok"}


AgentManagerType = Annotated[AgentManager, Depends(lambda: app.state.agent_manager)]

@app.post("/chat")
async def chat(
    payload: AgentRequest,
    agent_manager: AgentManagerType
) -> AgentResponse:
    try:
        return await agent_manager.run(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
