from fastapi import APIRouter, Depends, Request, Response, Form, UploadFile, File, Header
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from typing import Annotated, Optional, AsyncIterator
import json
from managers.audio import AudioManager
from models.common import Settings, ProviderKind
from models.audio.speak import SpeakRequest, SpeakResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse

audio_router = APIRouter(prefix="/audio", tags=["audio"])


def get_audio_manager(request: Request) -> AudioManager:
    return request.app.state.audio_manager


AudioManagerType = Annotated[AudioManager, Depends(get_audio_manager)]


@audio_router.post("/speak", response_class=Response)
async def audio_speak(
    payload: SpeakRequest,
    audio_manager: AudioManagerType,
    x_api_key: str = Header(..., description="API key for the provider"),
    x_provider: str = Header(..., description="Provider to use (openai, elevenlabs)"),
    x_base_url: Optional[str] = Header(None, description="Optional custom base URL for the provider")
) -> Response:
    """
    Convert text to speech using the specified audio provider.
    Returns raw audio bytes with appropriate content type.
    
    Headers:
        X-Api-Key: API key for the provider
        X-Provider: Provider to use (openai, elevenlabs)
        X-Base-Url: Optional custom base URL for the provider
    """
    try:
        # Build settings from headers
        settings = Settings(
            provider=ProviderKind(x_provider.lower()),
            api_key=x_api_key,
            base_url=x_base_url
        )
        result: SpeakResponse = await audio_manager.speak(settings, payload, stream=False)
        return Response(
            content=result.audio,
            media_type=result.content_type,
            headers={
                "X-Sample-Rate": str(result.sample_rate)
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@audio_router.post("/speak_stream", response_class=Response)
async def audio_speak_stream(
    payload: SpeakRequest,
    audio_manager: AudioManagerType,
    x_api_key: str = Header(..., description="API key for the provider"),
    x_provider: str = Header(..., description="Provider to use (openai, elevenlabs)"),
    x_base_url: Optional[str] = Header(None, description="Optional custom base URL for the provider")
) -> Response:
    """
    Convert text to speech using the specified audio provider.
    Returns streaming audio bytes with appropriate content type.
    
    Headers:
        X-Api-Key: API key for the provider
        X-Provider: Provider to use (openai, elevenlabs)
        X-Base-Url: Optional custom base URL for the provider
    """
    try:
        # Build settings from headers
        settings = Settings(
            provider=ProviderKind(x_provider.lower()),
            api_key=x_api_key,
            base_url=x_base_url
        )
        generator, mime_type, sample_rate = await audio_manager.speak(settings, payload, stream=True)
        async def _generator() -> AsyncIterator[bytes]:
            async for chunk in generator:
                yield chunk
        return StreamingResponse(
            _generator(),
            media_type=mime_type,
            headers={
                "X-Sample-Rate": str(sample_rate)
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@audio_router.post("/transcribe")
async def audio_transcribe(
    audio_manager: AudioManagerType,
    x_api_key: str = Header(..., description="API key for the provider"),
    x_provider: str = Header(..., description="Provider to use (openai, elevenlabs)"),
    model: str = Form(...),
    file: UploadFile = File(...),
    x_base_url: Optional[str] = Header(None, description="Optional custom base URL for the provider"),
    language: Optional[str] = Form(None),
    gen_config: Optional[str] = Form(None)
) -> TranscribeResponse:
    """
    Transcribe audio to text using the specified audio provider.
    Returns the transcribed text, language, and usage information.
    
    Headers:
        X-Api-Key: API key for the provider
        X-Provider: Provider to use (openai, elevenlabs)
        X-Base-Url: Optional custom base URL for the provider
    """
    try:
        # Build settings from headers
        settings = Settings(
            provider=ProviderKind(x_provider.lower()),
            api_key=x_api_key,
            base_url=x_base_url
        )
        
        # Parse gen_config if provided
        parsed_gen_config = None
        if gen_config:
            parsed_gen_config = json.loads(gen_config)
        
        # Read file contents
        file_content = await file.read()
        file_tuple = (file.filename, file_content, file.content_type)
        
        # Create TranscribeRequest
        transcribe_request = TranscribeRequest(
            model=model,
            file=file_tuple,
            language=language,
            gen_config=parsed_gen_config
        )
        
        # Call the transcribe method on the manager
        result: TranscribeResponse = await audio_manager.transcribe(settings, transcribe_request, stream=False)
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid provider: {str(e)}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in gen_config: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))