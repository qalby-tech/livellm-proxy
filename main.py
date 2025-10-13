"""
proxy server for google-genai, elevenlabs and openai itself
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from contextlib import asynccontextmanager
from agent.base import Agent
from agent.openai import OpenAIAgent
from agent.gemini import GeminiAgent
from agent.elevenlabs import ElevenLabsAgent
from models.requests import SpeakRequest, ChatRequest
from models.responses import TranscribeResponse, ChatResponse
from models.errors import InvalidModelRefError
from utils import parse_model_ref
from typing import Optional, Annotated
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    master_api_key: str = Field(..., description="The master API key, used to authenticate requests")
    host: str = Field(default="0.0.0.0", description="The host to run the server on")
    port: int = Field(default=8000, description="The port to run the server on")
    openai_api_key: Optional[str] = Field(None, description="The OpenAI API key")
    google_api_key: Optional[str] = Field(None, description="The Google API key")
    elevenlabs_api_key: Optional[str] = Field(None, description="The ElevenLabs API key")
    elevenlabs_base_url: Optional[str] = Field(None, description="The ElevenLabs API base URL")
    google_base_url: Optional[str] = Field(None, description="The Google API base URL")
    openai_base_url: Optional[str] = Field(None, description="The OpenAI API base URL")

settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.master_api_key = settings.master_api_key
    app.state.openai_agent = None
    app.state.google_agent = None
    app.state.elevenlabs_agent = None
    # main agents
    if settings.openai_api_key:
        app.state.openai_agent = OpenAIAgent(settings.openai_api_key, settings.openai_base_url)
    if settings.google_api_key:
        app.state.google_agent = GeminiAgent(settings.google_api_key, settings.google_base_url)
    if settings.elevenlabs_api_key:
        app.state.elevenlabs_agent = ElevenLabsAgent(settings.elevenlabs_api_key, settings.elevenlabs_base_url)
    # agent map
    app.state.agents = {
        "openai": app.state.openai_agent,
        "google": app.state.google_agent,
        "elevenlabs": app.state.elevenlabs_agent,
    }
    yield
    if app.state.google_agent:
        await app.state.google_agent.close()
    if app.state.elevenlabs_agent:
        await app.state.elevenlabs_agent.close()
    if app.state.openai_agent:
        await app.state.openai_agent.close()




app = FastAPI(lifespan=lifespan)

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
Agents = Annotated[dict, Depends(lambda: app.state.agents)]



@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    credentials: Credentials,
    master_api_key: MasterApiKey,
    agents: Agents,
    file: UploadFile = File(..., description="The audio file to transcribe"),
    model: str = Form(..., description="The model to use for the request"),
    language: Optional[str] = Form(None, description="The language of the audio"),
    ):
    if credentials.credentials != master_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        provider, model_name = await parse_model_ref(model)
        agent: Agent = agents[provider]
        
        # Read file and prepare tuple format
        file_content = await file.read()
        file_tuple = (file.filename, file_content, file.content_type)
        
        text = await agent.transcribe(
            model=model_name,
            file=file_tuple,
            language=language,
        )
        return TranscribeResponse(text=text)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: '{provider}'")
    except InvalidModelRefError:
        raise HTTPException(status_code=400, detail=f"Invalid model reference: '{model}'")
    except NotImplementedError:
        raise HTTPException(status_code=400, detail=f"Agent '{provider}' does not support transcribe endpoint")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error transcribing audio: {str(e)}")


@app.post("/speak")
async def speak(
    payload: SpeakRequest,
    credentials: Credentials,
    master_api_key: MasterApiKey,
    agents: Agents,
    ):
    if credentials.credentials != master_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        provider, model_name = await parse_model_ref(payload.model)
        agent: Agent = agents[provider]
        audio_bytes, mime_type = await agent.speak(
            model=model_name,
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
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: '{provider}'")
    except InvalidModelRefError:
        raise HTTPException(status_code=400, detail=f"Invalid model reference: '{payload.model}'")
    except NotImplementedError:
        raise HTTPException(status_code=400, detail=f"Agent '{provider}' does not support speak endpoint")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error speaking: {str(e)}")


@app.post("/chat")
async def chat(
    payload: ChatRequest,
    credentials: Credentials,
    master_api_key: MasterApiKey,
    agents: Agents,
    ) -> ChatResponse:
    if credentials.credentials != master_api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        provider, model_name = await parse_model_ref(payload.model)
        agent: Agent = agents[provider]
        text = await agent.chat(
            model=model_name,
            messages=payload.messages,
            tools=payload.tools,
            tool_choice=payload.tool_choice,
        )
        return ChatResponse(text=text)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unknown provider: '{provider}'")
    except InvalidModelRefError:
        raise HTTPException(status_code=400, detail=f"Invalid model reference: '{payload.model}'")
    except NotImplementedError:
        raise HTTPException(status_code=400, detail=f"Agent '{provider}' does not support run endpoint")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)