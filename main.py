import os
from fastapi import FastAPI
from typing import Optional
from contextlib import asynccontextmanager
from managers.agent import AgentManager
from managers.audio import AudioManager
from managers.config import ConfigManager
from managers.fallback import FallbackManager
from managers.ws import WsManager
from managers.transcription_rt import TranscriptionRTManager
from managers.redis import RedisManager
from routers.agent import agent_router
from routers.audio import audio_router
from routers.providers import providers_router
from routers.ws import ws_router
from routers.transcription_ws import transcription_ws_router
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
    redis_url: Optional[str] = Field(None, description="Redis URL for provider settings storage")
    redis_encryption_salt: Optional[str] = Field(None, description="Salt for encrypting Redis data")

env_settings = EnvSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Redis manager
    app.state.redis_manager = RedisManager(
        redis_url=env_settings.redis_url,
        encryption_salt=env_settings.redis_encryption_salt
    )
    await app.state.redis_manager.connect()
    
    # Initialize config manager with Redis support
    app.state.config_manager = ConfigManager(redis_manager=app.state.redis_manager)
    
    # Load provider configurations from Redis
    await app.state.config_manager.load_providers_from_redis()
    
    app.state.fallback_manager = FallbackManager()
    app.state.agent_manager = AgentManager(
        config_manager=app.state.config_manager, 
        fallback_manager=app.state.fallback_manager
    )
    app.state.audio_manager = AudioManager(
        config_manager=app.state.config_manager,
        fallback_manager=app.state.fallback_manager
    )
    app.state.transcription_rt_manager = TranscriptionRTManager(
        config_manager=app.state.config_manager
    )
    app.state.ws_manager = WsManager(
        agent_manager=app.state.agent_manager,
        audio_manager=app.state.audio_manager
    )
    yield
    
    # Cleanup: disconnect from Redis
    await app.state.redis_manager.disconnect()


app = FastAPI(lifespan=lifespan, root_path="/livellm")
app.include_router(agent_router)
app.include_router(audio_router)
app.include_router(providers_router)
app.include_router(ws_router)
app.include_router(transcription_ws_router)

# configure logfire
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = env_settings.otel_exporter_otlp_endpoint or ""
logfire.configure(service_name="livellm-proxy", send_to_logfire="if-token-present", token=env_settings.logfire_write_token)
logfire.instrument_pydantic_ai()
logfire.instrument_mcp()
logfire.instrument_fastapi(app, capture_headers=True, excluded_urls=["/ping"])
logfire.instrument_openai_agents()


@app.get("/ping")
async def ping():
    return SuccessResponse()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=env_settings.host, port=env_settings.port)
