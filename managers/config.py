from typing import Dict, Union, TypeAlias, Optional, Tuple
from models.common import Settings
from models.common import ProviderKind
import logfire

# providers
from openai import AsyncOpenAI
from google import genai
from anthropic import AsyncAnthropic
from groq import AsyncGroq
from elevenlabs import AsyncElevenLabs

ProviderClient: TypeAlias = Union[AsyncOpenAI, genai.Client, AsyncAnthropic, AsyncGroq, AsyncElevenLabs]


class ConfigManager:

    def __init__(self, redis_manager=None):
        self.configs: Dict[str, Settings] = {} # config_id: Settings
        self.providers: Dict[str, ProviderClient] = {} # provider_id: provider kind's client instance
        self.redis_manager = redis_manager
    

    async def load_providers_from_redis(self):
        """Load all provider configurations from Redis on startup"""
        if not self.redis_manager:
            return
        
        all_settings = await self.redis_manager.load_all_provider_settings()
        for uid, settings_dict in all_settings.items():
            try:
                # Reconstruct Settings object from dictionary
                settings = Settings(**settings_dict)
                self.configs[uid] = settings
                self.providers[uid] = self.create_provider_client(settings)
            except Exception as e:
                logfire.error(f"Failed to load provider {uid} from Redis: {e}")
    
    async def add_config(self, config: Settings):
        """Add provider configuration and persist to Redis"""
        self.configs[config.uid] = config
        self.providers[config.uid] = self.create_provider_client(config)
        
        # Persist to Redis if available
        if self.redis_manager:
            # Manually serialize to include the actual secret values
            settings_dict = config.model_dump(mode='json')
            # Replace the masked api_key with the actual secret value
            settings_dict['api_key'] = config.api_key.get_secret_value()
            await self.redis_manager.save_provider_settings(
                config.uid, 
                settings_dict
            )
    
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
        provider_client: Optional[ProviderClient] = self.get_config_client(uid, model)
        if provider_client is None:
            if uid not in self.configs:
                raise ValueError(f"Provider config with uid '{uid}' not found. Please register the config first using POST /config")
            settings = self.configs[uid]
            if settings.blacklist_models and model in settings.blacklist_models:
                raise ValueError(f"Model '{model}' is blacklisted for provider '{uid}'")
            raise ValueError(f"Provider '{uid}' client could not be created")
        provider_kind: ProviderKind = self.get_config_provider(uid)
        return provider_kind, provider_client
    
    async def delete_config(self, uid: str):
        """Delete provider configuration and remove from Redis"""
        if uid not in self.configs:
            raise ValueError(f"Config {uid} not found")
        self.configs.pop(uid)
        self.providers.pop(uid)
        
        # Remove from Redis if available
        if self.redis_manager:
            await self.redis_manager.delete_provider_settings(uid)
    
    def create_provider_client(self, settings: Settings) -> ProviderClient:
        # Extract the actual API key from SecretStr
        api_key = settings.api_key.get_secret_value()
        
        if settings.provider == ProviderKind.OPENAI or settings.provider == ProviderKind.OPENAI_CHAT:
            return AsyncOpenAI(api_key=api_key, base_url=settings.base_url)
        elif settings.provider == ProviderKind.GOOGLE:
            return genai.Client(api_key=api_key, http_options=genai.types.HttpOptions(base_url=settings.base_url))
        elif settings.provider == ProviderKind.ANTHROPIC:
            return AsyncAnthropic(api_key=api_key, base_url=settings.base_url)
        elif settings.provider == ProviderKind.GROQ:
            return AsyncGroq(api_key=api_key, base_url=settings.base_url)
        elif settings.provider == ProviderKind.ELEVENLABS:
            return AsyncElevenLabs(api_key=api_key, base_url=settings.base_url)
        else:
            raise ValueError(f"Provider {settings.provider} not supported")