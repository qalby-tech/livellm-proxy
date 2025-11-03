from typing import Dict, Union, TypeAlias, Optional, Tuple
from models.common import Settings
from models.common import ProviderKind

# providers
from openai import AsyncOpenAI
from google import genai
from anthropic import AsyncAnthropic
from groq import AsyncGroq
from elevenlabs import ElevenLabs

ProviderClient: TypeAlias = Union[AsyncOpenAI, genai.Client, AsyncAnthropic, AsyncGroq, ElevenLabs]


class ConfigManager:

    def __init__(self):
        self.configs: Dict[str, Settings] = {} # config_id: Settings
        self.providers: Dict[str, ProviderClient] = {} # provider_id: provider kind's client instance
    

    def add_config(self, config: Settings):
        self.configs[config.uid] = config
        self.providers[config.uid] = self.create_provider_client(config)
    
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
    
    def delete_config(self, uid: str):
        if uid not in self.configs:
            raise ValueError(f"Config {uid} not found")
        self.configs.pop(uid)
        self.providers.pop(uid)
    
    def create_provider_client(self, settings: Settings) -> ProviderClient:
        if settings.provider == ProviderKind.OPENAI:
            return AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
        elif settings.provider == ProviderKind.GOOGLE:
            return genai.Client(api_key=settings.api_key, http_options=genai.types.HttpOptions(base_url=settings.base_url))
        elif settings.provider == ProviderKind.ANTHROPIC:
            return AsyncAnthropic(api_key=settings.api_key, base_url=settings.base_url)
        elif settings.provider == ProviderKind.GROQ:
            return AsyncGroq(api_key=settings.api_key, base_url=settings.base_url)
        elif settings.provider == ProviderKind.ELEVENLABS:
            return ElevenLabs(api_key=settings.api_key, base_url=settings.base_url)
        else:
            raise ValueError(f"Provider {settings.provider} not supported")