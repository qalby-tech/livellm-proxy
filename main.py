import os
from fastapi import FastAPI, Request, Depends
from fastapi.exceptions import HTTPException
from typing import Annotated
from contextlib import asynccontextmanager
from managers.agent import AgentManager
from managers.audio import AudioManager
from managers.config import ConfigManager
from routers.agent import agent_router
from routers.audio import audio_router
from models.common import Settings, SuccessResponse

import logfire

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config_manager = ConfigManager()
    app.state.agent_manager = AgentManager(config_manager=app.state.config_manager)
    app.state.audio_manager = AudioManager(config_manager=app.state.config_manager)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(agent_router)
app.include_router(audio_router)

# configure logfire
logfire_token = os.getenv('LOGFIRE_WRITE_TOKEN', None)
otel_exporter = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')
logfire.configure(service_name="livellm-proxy", send_to_logfire="if-token-present", token=logfire_token)
logfire.instrument_pydantic_ai()
logfire.instrument_mcp()
logfire.instrument_fastapi(app, capture_headers=True, excluded_urls=["/ping"])


def get_config_manager(request: Request) -> ConfigManager:
    return request.app.state.config_manager

ConfigManagerType = Annotated[ConfigManager, Depends(get_config_manager)]

@app.get("/ping")
async def ping():
    return SuccessResponse()


@app.post("/config")
async def config(request: Settings, config_manager: ConfigManagerType) -> SuccessResponse:
    try:
        config_manager.add_config(request)
        return SuccessResponse()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/config/{uid}")
async def delete_config(uid: str, config_manager: ConfigManagerType) -> SuccessResponse:
    """
    Delete a provider configuration by uid.
    
    Args:
        uid: The unique identifier of the provider configuration to delete
    """
    try:
        config_manager.delete_config(uid)
        return SuccessResponse()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))
