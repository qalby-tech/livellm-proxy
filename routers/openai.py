"""
OpenAI-compatible API router.
"""
from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import HTTPException
from typing import Annotated
from managers.config import ConfigManager
from models.common import Settings, SuccessResponse

openai_router = APIRouter(prefix="/openai", tags=["openai"])

@openai_router.get("/v1/models")
async def get_models():
    pass

@openai_router.post("/v1/chat/completions")
async def create_chat_completion():
    pass


@openai_router.post("/v1/audio/transcriptions")
async def create_transcription():
    pass
