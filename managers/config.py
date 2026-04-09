"""Config manager — Redis-only persistence with Pub/Sub hot-reload across replicas"""

import asyncio
import json
from typing import Dict, Optional, Tuple, TypeAlias, Union

import logfire
from anthropic import AsyncAnthropic
from elevenlabs import AsyncElevenLabs
from google import genai
from groq import AsyncGroq
from openai import AsyncOpenAI

from managers.redis import RedisManager
from models.common import ProviderKind, Settings, ModelConfig

ProviderClient: TypeAlias = Union[
    AsyncOpenAI, genai.Client, AsyncAnthropic, AsyncGroq, AsyncElevenLabs
]


class ConfigManager:
    """
    Manages LLM provider configurations.

    Persistence: Redis is the single source of truth.
    Scalability: all replicas stay in sync via Redis Pub/Sub — when any instance
                 (or the operator) writes a provider to Redis it publishes an event;
                 every replica's background listener reacts by hot-reloading that
                 provider into its in-memory state without restarting.
    """

    def __init__(self, redis_manager: RedisManager):
        self.configs: Dict[str, Settings] = {}  # uid → Settings
        self.providers: Dict[str, ProviderClient] = {}  # uid → SDK client instance
        self.redis_manager = redis_manager

    # ------------------------------------------------------------------
    # Startup / persistence
    # ------------------------------------------------------------------

    async def load_providers_from_persistence(self):
        """Load all provider configs from Redis on startup."""
        all_settings = await self.redis_manager.load_all_provider_settings()
        for uid, settings_dict in all_settings.items():
            try:
                settings = Settings(**settings_dict)
                self.configs[uid] = settings
                self.providers[uid] = self.create_provider_client(settings)
            except Exception as e:
                logfire.error(f"Failed to load provider '{uid}' from Redis: {e}")

        logfire.info(f"Loaded {len(self.configs)} provider(s) from Redis on startup")

    # ------------------------------------------------------------------
    # CRUD — called by the REST API routers
    # ------------------------------------------------------------------

    async def add_config(self, config: Settings):
        """
        Register or update a provider config.

        Updates this replica's in-memory state immediately, persists to Redis,
        and publishes a Pub/Sub event so all other replicas hot-reload.
        """
        self.configs[config.uid] = config
        self.providers[config.uid] = self.create_provider_client(config)

        # Serialize — replace the masked SecretStr with the real value
        settings_dict = config.model_dump(mode="json")
        settings_dict["api_key"] = config.api_key.get_secret_value()

        success = await self.redis_manager.save_provider_settings(
            config.uid, settings_dict
        )
        if not success:
            logfire.error(
                f"Failed to persist provider '{config.uid}' to Redis — "
                "in-memory state updated but other replicas will not be notified"
            )

    async def delete_config(self, uid: str):
        """
        Remove a provider config.

        Updates this replica's in-memory state immediately, deletes from Redis,
        and publishes a Pub/Sub event so all other replicas drop it too.
        """
        if uid not in self.configs:
            raise ValueError(f"Config '{uid}' not found")

        self.configs.pop(uid)
        self.providers.pop(uid)

        await self.redis_manager.delete_provider_settings(uid)

    # ------------------------------------------------------------------
    # Lookups — called by route handlers
    # ------------------------------------------------------------------

    def get_config_client(self, uid: str, model: str) -> Optional[ProviderClient]:
        if uid not in self.configs:
            return None
        settings = self.configs[uid]
        if settings.blacklist_models and model in settings.blacklist_models:
            return None
        return self.providers[uid]

    def get_config_provider(self, uid: str) -> Optional[ProviderKind]:
        if uid not in self.configs:
            return None
        return self.configs[uid].provider

    def get_provider(self, uid: str, model: str) -> Tuple[ProviderKind, ProviderClient]:
        provider_client = self.get_config_client(uid, model)
        if provider_client is None:
            if uid not in self.configs:
                raise ValueError(
                    f"Provider config with uid '{uid}' not found. "
                    "Please register the config first using POST /providers/config"
                )
            settings = self.configs[uid]
            if settings.blacklist_models and model in settings.blacklist_models:
                raise ValueError(f"Model '{model}' is blacklisted for provider '{uid}'")
            raise ValueError(f"Provider '{uid}' client could not be created")
        provider_kind = self.get_config_provider(uid)
        return provider_kind, provider_client

    def get_model_config(self, uid: str, model: str) -> Optional[ModelConfig]:
        """
        Get the model-specific configuration for a given provider and model.
        
        Args:
            uid: The provider UID
            model: The model name
            
        Returns:
            ModelConfig if configured for this model, None otherwise
        """
        if uid not in self.configs:
            return None
        settings = self.configs[uid]
        if not settings.model_configs:
            return None
        return settings.model_configs.get(model)

    # ------------------------------------------------------------------
    # Provider client factory
    # ------------------------------------------------------------------

    def create_provider_client(self, settings: Settings) -> ProviderClient:
        api_key = settings.api_key.get_secret_value()

        if settings.provider in (ProviderKind.OPENAI, ProviderKind.OPENAI_CHAT):
            return AsyncOpenAI(api_key=api_key, base_url=settings.base_url)
        elif settings.provider == ProviderKind.GOOGLE:
            return genai.Client(
                api_key=api_key,
                http_options=genai.types.HttpOptions(base_url=settings.base_url),
            )
        elif settings.provider == ProviderKind.ANTHROPIC:
            return AsyncAnthropic(api_key=api_key, base_url=settings.base_url)
        elif settings.provider == ProviderKind.GROQ:
            return AsyncGroq(api_key=api_key, base_url=settings.base_url)
        elif settings.provider == ProviderKind.ELEVENLABS:
            return AsyncElevenLabs(api_key=api_key, base_url=settings.base_url)
        else:
            raise ValueError(f"Provider '{settings.provider}' is not supported")

    # ------------------------------------------------------------------
    # Pub/Sub hot-reload — long-running background task
    # ------------------------------------------------------------------

    async def pubsub_listener_task(self):
        """
        Long-running background task.

        Subscribes to the Redis Pub/Sub channel and reacts to provider change
        events published by any replica or by the operator.  Reconnects
        automatically on connection loss.

        Event message format (JSON):
            {"action": "upsert", "uid": "<provider-uid>"}
            {"action": "delete", "uid": "<provider-uid>"}
        """
        logfire.info("Starting Redis Pub/Sub listener for provider hot-reload")

        while True:
            pubsub = None
            try:
                pubsub = self.redis_manager.redis_client.pubsub()
                await pubsub.subscribe(RedisManager.PROVIDERS_CHANNEL)
                logfire.info(
                    f"Subscribed to Redis channel '{RedisManager.PROVIDERS_CHANNEL}'"
                )

                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    try:
                        await self._handle_pubsub_event(message["data"])
                    except Exception as e:
                        logfire.error(f"Error processing Pub/Sub message: {e}")

            except asyncio.CancelledError:
                logfire.info("Pub/Sub listener task cancelled — shutting down")
                if pubsub:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass
                raise

            except Exception as e:
                logfire.error(f"Pub/Sub listener error: {e} — reconnecting in 5 s")
                if pubsub:
                    try:
                        await pubsub.aclose()
                    except Exception:
                        pass
                await asyncio.sleep(5)

    async def _handle_pubsub_event(self, raw: bytes):
        """React to a single Pub/Sub event from Redis."""
        event = json.loads(raw)
        action = event.get("action")
        uid = event.get("uid")

        if not action or not uid:
            logfire.warn(f"Received malformed Pub/Sub event: {event}")
            return

        if action == "upsert":
            settings_dict = await self.redis_manager.load_provider_settings(uid)
            if settings_dict is None:
                logfire.warn(
                    f"Pub/Sub upsert event for '{uid}' but provider not found in Redis "
                    "(possible race condition — will be corrected on next reconcile)"
                )
                return
            try:
                settings = Settings(**settings_dict)
                self.configs[uid] = settings
                self.providers[uid] = self.create_provider_client(settings)
                logfire.info(f"Hot-reloaded provider '{uid}' via Pub/Sub")
            except Exception as e:
                logfire.error(f"Failed to hot-reload provider '{uid}': {e}")

        elif action == "delete":
            if uid in self.configs:
                self.configs.pop(uid, None)
                self.providers.pop(uid, None)
                logfire.info(f"Hot-removed provider '{uid}' via Pub/Sub")

        else:
            logfire.warn(f"Unknown Pub/Sub action '{action}' for uid '{uid}'")
