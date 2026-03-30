import asyncio
import logging
import os
from contextlib import asynccontextmanager
from logging import basicConfig
from typing import Optional

import logfire
from fastapi import FastAPI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from managers.agent import AgentManager
from managers.audio import AudioManager
from managers.config import ConfigManager
from managers.context import set_token_count_overhead
from managers.fallback import FallbackManager
from managers.redis import RedisManager
from managers.transcription_rt import TranscriptionRTManager
from managers.ws import WsManager
from models.common import SuccessResponse
from routers.agent import agent_router
from routers.audio import audio_router
from routers.providers import providers_router
from routers.transcription_ws import transcription_ws_router
from routers.ws import ws_router


class PingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return "/ping" not in message


def scrubbing_callback(m: logfire.ScrubMatch):
    if m.path == ("message", "e") or m.path == ("attributes", "e"):
        return m.value


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    logfire_write_token: Optional[str] = Field(None, description="Logfire write token")
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        None, description="OTEL exporter OTLP endpoint"
    )
    host: str = Field("0.0.0.0", description="Host")
    port: int = Field(8000, description="Port")
    # Redis is the sole persistence backend — required.
    redis_url: str = Field(
        ..., description="Redis URL (required). Example: redis://localhost:6379/0"
    )
    encryption_salt: Optional[str] = Field(
        None,
        description="Salt for Fernet encryption of provider data. If not set, data is stored as plain JSON.",
    )
    token_count_overhead: float = Field(
        1.20,
        description="Safety overhead multiplier for tiktoken token counting (e.g. 1.20 for 20 %% buffer)",
    )


env_settings = EnvSettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Token count overhead ---
    if env_settings.token_count_overhead != 1.20:
        set_token_count_overhead(env_settings.token_count_overhead)

    # --- Redis (required) ---
    app.state.redis_manager = RedisManager(
        redis_url=env_settings.redis_url,
        encryption_salt=env_settings.encryption_salt,
    )
    await app.state.redis_manager.connect()
    logfire.info("Redis manager connected")

    # --- Config manager — loads all providers from Redis on startup ---
    app.state.config_manager = ConfigManager(
        redis_manager=app.state.redis_manager,
    )
    await app.state.config_manager.load_providers_from_persistence()

    # --- Start Pub/Sub listener so every replica hot-reloads on changes ---
    app.state.pubsub_task = asyncio.create_task(
        app.state.config_manager.pubsub_listener_task()
    )

    # --- Business logic managers ---
    app.state.fallback_manager = FallbackManager()
    app.state.agent_manager = AgentManager(
        config_manager=app.state.config_manager,
        fallback_manager=app.state.fallback_manager,
    )
    app.state.audio_manager = AudioManager(
        config_manager=app.state.config_manager,
        fallback_manager=app.state.fallback_manager,
    )
    app.state.transcription_rt_manager = TranscriptionRTManager(
        config_manager=app.state.config_manager,
    )
    app.state.ws_manager = WsManager(
        agent_manager=app.state.agent_manager,
        audio_manager=app.state.audio_manager,
    )

    yield

    # --- Shutdown ---
    app.state.pubsub_task.cancel()
    try:
        await app.state.pubsub_task
    except asyncio.CancelledError:
        pass

    await app.state.redis_manager.disconnect()


app = FastAPI(lifespan=lifespan, root_path="/livellm")
app.include_router(agent_router)
app.include_router(audio_router)
app.include_router(providers_router)
app.include_router(ws_router)
app.include_router(transcription_ws_router)

# --- Logfire ---
os.environ["LOGFIRE_DISTRIBUTED_TRACING"] = "false"
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
    env_settings.otel_exporter_otlp_endpoint or ""
)
logfire.configure(
    service_name="livellm-proxy",
    send_to_logfire="if-token-present",
    token=env_settings.logfire_write_token,
    scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback),
)

logfire_handler = logfire.LogfireLoggingHandler()
basicConfig(handlers=[logfire_handler], level=logging.INFO)

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
