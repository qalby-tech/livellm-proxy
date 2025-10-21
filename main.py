import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from managers.agent import AgentManager
from managers.audio import AudioManager
from routers.agent import agent_router
from routers.audio import audio_router

import logfire

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent_manager = AgentManager()
    app.state.audio_manager = AudioManager()
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(agent_router)
app.include_router(audio_router)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
