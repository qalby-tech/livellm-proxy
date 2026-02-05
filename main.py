import os
import logging
import asyncio
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
from managers.context import set_token_count_overhead
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

# Constant file storage path for provider settings (used for file-based persistence and Redis backup)
FILE_STORAGE_PATH = "data/providers.json"

class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    logfire_write_token: Optional[str] = Field(None, description="Logfire write token")
    otel_exporter_otlp_endpoint: Optional[str] = Field(None, description="OTEL exporter otlp endpoint")
    host: str = Field("0.0.0.0", description="Host")
    port: int = Field(8000, description="Port")
    redis_url: Optional[str] = Field(None, description="Redis URL for provider settings storage")
    encryption_salt: Optional[str] = Field(None, description="Salt for encrypting data. If not provided, data will be stored unencrypted.")
    token_count_overhead: float = Field(1.20, description="Safety overhead multiplier for tiktoken token counting (e.g., 1.20 for 20% buffer)")
    enable_storage_reconciliation: bool = Field(True, description="Enable reconciliation between file storage and Redis on startup/shutdown")
    reconciliation_interval_seconds: int = Field(300, description="Interval in seconds for periodic reconciliation between file storage and Redis (default: 300 = 5 minutes)")

env_settings = EnvSettings()


async def backup_redis_to_file(
    redis_manager: RedisManager, 
    file_store: FileStoreManager
) -> int:
    """
    Backup all provider settings from Redis to file storage.
    Overwrites existing file storage data.
    
    Returns:
        Number of settings backed up
    """
    try:
        # Load all settings from Redis
        redis_settings = await redis_manager.load_all_provider_settings()
        if not redis_settings:
            logfire.info("No settings found in Redis to backup")
            return 0
        
        backed_up_count = 0
        for uid, settings_dict in redis_settings.items():
            success = await file_store.save_provider_settings(uid, settings_dict)
            if success:
                backed_up_count += 1
            else:
                logfire.error(f"Failed to backup provider {uid} to file storage")
        
        if backed_up_count > 0:
            logfire.info(f"Backed up {backed_up_count} provider settings from Redis to file storage")
        
        return backed_up_count
    except Exception as e:
        logfire.error(f"Failed to backup Redis to file storage: {e}")
        return 0


async def reconcile_storages(
    file_store: FileStoreManager,
    redis_manager: RedisManager
) -> tuple[int, int]:
    """
    Bidirectional reconciliation between file storage and Redis.
    - Settings in file but not in Redis → add to Redis
    - Settings in Redis but not in file → add to file
    
    Returns:
        Tuple of (file_to_redis_count, redis_to_file_count)
    """
    file_to_redis = 0
    redis_to_file = 0
    
    try:
        file_settings = await file_store.load_all_provider_settings()
        redis_settings = await redis_manager.load_all_provider_settings()
        
        # File → Redis (settings in file but not in Redis)
        for uid, settings_dict in file_settings.items():
            if uid not in redis_settings:
                success = await redis_manager.save_provider_settings(uid, settings_dict)
                if success:
                    file_to_redis += 1
                    logfire.debug(f"Reconciled provider {uid} from file to Redis")
        
        # Redis → File (settings in Redis but not in file)
        for uid, settings_dict in redis_settings.items():
            if uid not in file_settings:
                success = await file_store.save_provider_settings(uid, settings_dict)
                if success:
                    redis_to_file += 1
                    logfire.debug(f"Reconciled provider {uid} from Redis to file")
        
        if file_to_redis > 0 or redis_to_file > 0:
            logfire.info(f"Reconciliation complete: {file_to_redis} file→Redis, {redis_to_file} Redis→file")
        
        return file_to_redis, redis_to_file
    except Exception as e:
        logfire.error(f"Failed to reconcile storages: {e}")
        return 0, 0


async def periodic_reconciliation_task(
    file_store: FileStoreManager,
    redis_manager: RedisManager,
    interval_seconds: int
):
    """
    Background task that periodically reconciles file storage and Redis.
    """
    logfire.info(f"Starting periodic reconciliation task with interval {interval_seconds}s")
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            logfire.debug("Running periodic storage reconciliation")
            await reconcile_storages(file_store, redis_manager)
        except asyncio.CancelledError:
            logfire.info("Periodic reconciliation task cancelled")
            break
        except Exception as e:
            logfire.error(f"Error in periodic reconciliation: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Configure token count overhead for context overflow handling
    if env_settings.token_count_overhead != 1.20:
        set_token_count_overhead(env_settings.token_count_overhead)
    
    # Always initialize file store (PVC) - this is the primary persistence
    app.state.file_store = FileStoreManager(
        file_path=FILE_STORAGE_PATH,
        encryption_salt=env_settings.encryption_salt
    )
    logfire.info(f"Initialized file store at {FILE_STORAGE_PATH}")
    
    # Initialize Redis manager if configured (optional secondary persistence)
    app.state.redis_manager = None
    app.state.reconciliation_task = None
    
    if env_settings.redis_url:
        app.state.redis_manager = RedisManager(
            redis_url=env_settings.redis_url,
            encryption_salt=env_settings.encryption_salt
        )
        await app.state.redis_manager.connect()
        logfire.info("Redis manager initialized for secondary persistence")
        
        # Initial reconciliation: sync file storage and Redis
        if env_settings.enable_storage_reconciliation:
            await reconcile_storages(app.state.file_store, app.state.redis_manager)
            
            # Start periodic reconciliation background task
            app.state.reconciliation_task = asyncio.create_task(
                periodic_reconciliation_task(
                    app.state.file_store,
                    app.state.redis_manager,
                    env_settings.reconciliation_interval_seconds
                )
            )
        else:
            logfire.info("Storage reconciliation disabled")
    else:
        logfire.info("Redis not configured, using file store only")
    
    # Initialize config manager with both persistence backends
    # ConfigManager will always save to file store, and also to Redis if available
    app.state.config_manager = ConfigManager(
        file_store=app.state.file_store,
        redis_manager=app.state.redis_manager
    )
    
    # Load provider configurations from persistence (prefers Redis if available)
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
    
    # Shutdown: cancel periodic reconciliation task
    if app.state.reconciliation_task:
        app.state.reconciliation_task.cancel()
        try:
            await app.state.reconciliation_task
        except asyncio.CancelledError:
            pass
    
    # Shutdown: final backup Redis to file storage for safety
    if app.state.redis_manager:
        if env_settings.enable_storage_reconciliation:
            logfire.info("Final backup: syncing Redis to file storage on shutdown")
            await backup_redis_to_file(app.state.redis_manager, app.state.file_store)
        
        # Disconnect from Redis
        await app.state.redis_manager.disconnect()


app = FastAPI(lifespan=lifespan, root_path="/livellm")
app.include_router(agent_router)
app.include_router(audio_router)
app.include_router(providers_router)
app.include_router(ws_router)
app.include_router(transcription_ws_router)

# configure logfire
os.environ["LOGFIRE_DISTRIBUTED_TRACING"] = "false"
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
