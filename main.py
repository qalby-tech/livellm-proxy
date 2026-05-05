import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import Response
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from managers import telemetry as tel
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
        return "/ping" not in record.getMessage()


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    otel_exporter_otlp_endpoint: Optional[str] = Field(
        None, description="Base OTLP/HTTP endpoint, e.g. http://otel-collector:4318"
    )
    otel_service_name: str = Field(
        "livellm-proxy", description="OTel service.name resource attribute"
    )
    otel_environment: Optional[str] = Field(
        None, description="deployment.environment.name (e.g. prod, staging)"
    )
    log_prompts: bool = Field(
        False, description="If true, prompt/completion text is recorded on spans"
    )
    default_project: Optional[str] = Field(
        None, description="Fallback project name when X-Project header is absent"
    )
    host: str = Field("0.0.0.0", description="Host")
    port: int = Field(8000, description="Port")
    redis_url: str = Field(
        ..., description="Redis URL (required). Example: redis://localhost:6379/0"
    )
    encryption_salt: Optional[str] = Field(
        None,
        description="Salt for Fernet encryption of provider data. If not set, data is stored as plain JSON.",
    )
    token_count_overhead: float = Field(
        1.20,
        description="Safety overhead multiplier for tiktoken token counting (e.g. 1.20 for 20%% buffer)",
    )


env_settings = EnvSettings()

# --- OpenTelemetry tracing must be set up before FastAPI is instrumented ---
import os

if env_settings.log_prompts:
    os.environ["LOG_PROMPTS"] = "true"

tel.configure_tracing(
    service_name=env_settings.otel_service_name,
    otlp_endpoint=env_settings.otel_exporter_otlp_endpoint,
    environment=env_settings.otel_environment,
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("uvicorn.access").addFilter(PingFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    if env_settings.token_count_overhead != 1.20:
        set_token_count_overhead(env_settings.token_count_overhead)

    app.state.redis_manager = RedisManager(
        redis_url=env_settings.redis_url,
        encryption_salt=env_settings.encryption_salt,
    )
    await app.state.redis_manager.connect()
    tel.info("Redis manager connected")

    app.state.config_manager = ConfigManager(
        redis_manager=app.state.redis_manager,
    )
    await app.state.config_manager.load_providers_from_persistence()

    app.state.pubsub_task = asyncio.create_task(
        app.state.config_manager.pubsub_listener_task()
    )

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


@app.middleware("http")
async def project_baggage_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    project = request.headers.get(tel.PROJECT_HEADER) or env_settings.default_project
    token = tel.attach_project(project) if project else None
    try:
        return await call_next(request)
    finally:
        if token is not None:
            tel.detach_project(token)


# Instrument FastAPI for HTTP server spans (excluding the noisy /ping endpoint).
FastAPIInstrumentor.instrument_app(app, excluded_urls="/ping")


@app.get("/ping")
async def ping():
    return SuccessResponse()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=env_settings.host, port=env_settings.port)
