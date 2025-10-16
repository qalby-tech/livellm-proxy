"""
proxy server for google-genai, elevenlabs and openai itself
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings import PydanticBaseSettingsSource, YamlConfigSettingsSource
from contextlib import asynccontextmanager
from agent.base import Agent
from agent.openai import OpenAIAgent
from agent.gemini import GeminiAgent
from agent.elevenlabs import ElevenLabsAgent
from models.requests import SpeakRequest, ChatRequest
from models.responses import TranscribeResponse, ChatResponse
from models.errors import InvalidModelRefError
from runner import Runner
from typing import Optional, Annotated, List, Dict, Tuple, Type
from pydantic import BaseModel, Field
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProviderKind(str, Enum):
    openai = "openai"
    google = "google"
    elevenlabs = "elevenlabs"

class ProviderConfig(BaseModel):
    kind: str = Field(..., description="Provider SDK kind: openai | google | elevenlabs")
    name: str = Field(..., description="The name of the provider")
    api_key: Optional[str] = Field(None, description="API key value OR env var name if api_key_env is not set")
    base_url: Optional[str] = Field(None, description="Optional base URL for the provider API")


class FallbackConfig(BaseModel):
    transcribe_model: str = Field(..., description="The model to use for the fallback e.g. openai/whisper-1")
    speak_model: str = Field(..., description="The model to use for the fallback e.g. openai/tts-1")
    chat_model: str = Field(..., description="The model to use for the fallback e.g. openai/gpt-4o-mini")

class Settings(BaseSettings):
    model_config = SettingsConfigDict(yaml_file="config.yaml")
    master_api_key: str = Field(..., description="The master API key, used to authenticate requests")
    host: str = Field(default="0.0.0.0", description="The host to run the server on")
    port: int = Field(default=8000, description="The port to run the server on")
    providers: List[ProviderConfig] = Field(..., description="The providers to use")
    fallback: Optional[FallbackConfig] = Field(None, description="The fallback to use")
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (YamlConfigSettingsSource(settings_cls),)

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    PROVIDER_KIND_TO_CLASS = {
        ProviderKind.openai: OpenAIAgent,
        ProviderKind.google: GeminiAgent,
        ProviderKind.elevenlabs: ElevenLabsAgent,
    }
    app.state.master_api_key = settings.master_api_key
    
    # Initialize agents
    agents: Dict[str, Agent] = {}
    for provider in settings.providers:
        agent_class = PROVIDER_KIND_TO_CLASS[provider.kind]
        agents[provider.name] = agent_class(provider.api_key, provider.base_url)
    
    # Initialize runner with agents and fallback configuration
    app.state.runner = Runner(
        agents=agents,
        transcribe_fallback=settings.fallback.transcribe_model if settings.fallback else None,
        speak_fallback=settings.fallback.speak_model if settings.fallback else None,
        chat_fallback=settings.fallback.chat_model if settings.fallback else None,
    )
    
    yield
    
    # Cleanup
    for agent in app.state.runner.agents.values():
        await agent.close()


app = FastAPI(lifespan=lifespan)

# Add observability middleware
# app.add_middleware(ObservabilityMiddleware) # TODO: Add observability middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer()

Credentials = Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)]
MasterApiKey = Annotated[str, Depends(lambda: app.state.master_api_key)]
RunnerDep = Annotated[Runner, Depends(lambda: app.state.runner)]



@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    credentials: Credentials,
    master_api_key: MasterApiKey,
    runner: RunnerDep,
    file: UploadFile = File(..., description="The audio file to transcribe"),
    model: str = Form(..., description="The model to use for the request"),
    language: Optional[str] = Form(None, description="The language of the audio"),
) -> TranscribeResponse:
    if credentials.credentials != master_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Read file and prepare tuple format
        file_content = await file.read()
        file_tuple = (file.filename, file_content, file.content_type)
        
        # Use runner which handles fallback automatically
        text = await runner.transcribe(
            model=model,
            file=file_tuple,
            language=language,
        )
        return TranscribeResponse(text=text)
    except InvalidModelRefError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error transcribing audio with model '{model}'")
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")


@app.post("/speak")
async def speak(
    payload: SpeakRequest,
    credentials: Credentials,
    master_api_key: MasterApiKey,
    runner: RunnerDep,
) -> Response:
    if credentials.credentials != master_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Use runner which handles fallback automatically
        audio_bytes, mime_type = await runner.speak(
            model=payload.model,
            text=payload.text,
            voice=payload.voice,
            response_format=payload.response_format,
        )
        
        return Response(
            content=audio_bytes,
            media_type=mime_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(audio_bytes)),
                "Cache-Control": "no-cache",
            },
        )
    except InvalidModelRefError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error generating speech with model '{payload.model}'")
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


@app.post("/chat")
async def chat(
    payload: ChatRequest,
    credentials: Credentials,
    master_api_key: MasterApiKey,
    runner: RunnerDep,
) -> ChatResponse:
    if credentials.credentials != master_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # Use runner which handles fallback automatically
        text = await runner.chat(
            model=payload.model,
            messages=payload.messages,
            tools=payload.tools,
            tool_choice=payload.tool_choice,
        )
        return ChatResponse(text=text)
    except InvalidModelRefError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except NotImplementedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Error in chat with model '{payload.model}'")
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)