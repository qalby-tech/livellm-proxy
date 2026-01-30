import os
import logging
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
from managers.file_store import FileStoreManager
from routers.agent import agent_router
from routers.audio import audio_router
from routers.providers import providers_router
from routers.ws import ws_router
from routers.transcription_ws import transcription_ws_router
from models.common import SuccessResponse
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

import logfire
from logging import basicConfig


class PingFilter(logging.Filter):
    """Filter out /ping health check requests from access logs."""
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "/ping" not in message

def scrubbing_callback(m: logfire.ScrubMatch):
    if (
        m.path == ('message', 'e')
        or m.path == ('attributes', 'e')
    ):
        return m.value

class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    logfire_write_token: Optional[str] = Field(None, description="Logfire write token")
    otel_exporter_otlp_endpoint: Optional[str] = Field(None, description="OTEL exporter otlp endpoint")
    host: str = Field("0.0.0.0", description="Host")
    port: int = Field(8000, description="Port")
    redis_url: Optional[str] = Field(None, description="Redis URL for provider settings storage")
    encryption_salt: Optional[str] = Field(None, description="Salt for encrypting data. If not provided, data will be stored unencrypted.")
    file_storage_path: str = Field("/data/providers.json", description="Path to file storage for provider settings")

env_settings = EnvSettings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize persistence manager
    if env_settings.redis_url:
        app.state.persistence_manager = RedisManager(
            redis_url=env_settings.redis_url,
            encryption_salt=env_settings.encryption_salt
        )
        await app.state.persistence_manager.connect()
        logfire.info("Using Redis for persistence")
    else:
        app.state.persistence_manager = FileStoreManager(
            file_path=env_settings.file_storage_path,
            encryption_salt=env_settings.encryption_salt
        )
        logfire.info(f"Using File Storage for persistence at {env_settings.file_storage_path}")
    
    # Initialize config manager with persistence support
    app.state.config_manager = ConfigManager(persistence_manager=app.state.persistence_manager)
    
    # Load provider configurations from persistence
    await app.state.config_manager.load_providers_from_persistence()
    
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
    
    # Cleanup: disconnect from persistence if needed
    if hasattr(app.state.persistence_manager, 'disconnect'):
        await app.state.persistence_manager.disconnect()


app = FastAPI(lifespan=lifespan, root_path="/livellm")
app.include_router(agent_router)
app.include_router(audio_router)
app.include_router(providers_router)
app.include_router(ws_router)
app.include_router(transcription_ws_router)

# configure logfire
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = env_settings.otel_exporter_otlp_endpoint or ""
logfire.configure(
    service_name="livellm-proxy", 
    send_to_logfire="if-token-present", 
    token=env_settings.logfire_write_token,
    scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback)
    )

# Set up Logfire as a logging sink for standard library logging
logfire_handler = logfire.LogfireLoggingHandler()
basicConfig(handlers=[logfire_handler], level=logging.INFO)

# Filter out /ping health check requests from uvicorn access logs
uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(PingFilter())

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
