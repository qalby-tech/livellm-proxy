import os
from fastapi import FastAPI
from typing import Optional
from contextlib import asynccontextmanager
from managers.agent import AgentManager
from managers.audio import AudioManager
from managers.config import ConfigManager
from managers.fallback import FallbackManager
from managers.ws import WsManager
from routers.agent import agent_router
from routers.audio import audio_router
from routers.providers import providers_router
from routers.ws import ws_router
from models.common import SuccessResponse
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

import logfire


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    logfire_write_token: Optional[str] = Field(None, description="Logfire write token")
    otel_exporter_otlp_endpoint: Optional[str] = Field(None, description="OTEL exporter otlp endpoint")
    host: str = Field("0.0.0.0", description="Host")
    port: int = Field(8000, description="Port")

env_settings = EnvSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.config_manager = ConfigManager()
    app.state.fallback_manager = FallbackManager()
    app.state.agent_manager = AgentManager(
        config_manager=app.state.config_manager, 
        fallback_manager=app.state.fallback_manager
    )
    app.state.audio_manager = AudioManager(
        config_manager=app.state.config_manager,
        fallback_manager=app.state.fallback_manager
    )
    app.state.ws_manager = WsManager(
        agent_manager=app.state.agent_manager,
        audio_manager=app.state.audio_manager
    )
    yield


app = FastAPI(lifespan=lifespan, root_path="/livellm")
app.include_router(agent_router)
app.include_router(audio_router)
app.include_router(providers_router)
app.include_router(ws_router)

# configure logfire
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = env_settings.otel_exporter_otlp_endpoint or ""
logfire.configure(service_name="livellm-proxy", send_to_logfire="if-token-present", token=env_settings.logfire_write_token)
logfire.instrument_pydantic_ai()
logfire.instrument_mcp()
logfire.instrument_fastapi(app, capture_headers=True, excluded_urls=["/ping"])


@app.get("/ping")
async def ping():
    return SuccessResponse()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=env_settings.host, port=env_settings.port)
