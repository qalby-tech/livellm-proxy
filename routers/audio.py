from fastapi import APIRouter, Depends, Request, Response, Form, UploadFile, File, Header
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from typing import Annotated, Optional, AsyncIterator
import json
import logfire
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
    x_provider_uid: str = Header(..., alias="X-Provider-UID", description="Provider UID configured via POST /config"),
) -> Response:
    """
    Convert text to speech using the specified audio provider.
    Returns raw audio bytes with appropriate content type.
    
    The provider UID must be configured first using POST /config endpoint.
    
    Headers:
        X-Provider-UID: The unique identifier of the provider configuration to use
    """
    try:
        result: SpeakResponse = await audio_manager.speak(x_provider_uid, payload, stream=False)
        return Response(
            content=result.audio,
            media_type=result.content_type,
            headers={
                "X-Sample-Rate": str(result.sample_rate)
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logfire.error(f"Unexpected error in audio_speak: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@audio_router.post("/speak_stream", response_class=Response)
async def audio_speak_stream(
    payload: SpeakRequest,
    audio_manager: AudioManagerType,
    x_provider_uid: str = Header(..., alias="X-Provider-UID", description="Provider UID configured via POST /config"),
) -> Response:
    """
    Convert text to speech using the specified audio provider.
    Returns streaming audio bytes with appropriate content type.
    
    The provider UID must be configured first using POST /config endpoint.
    
    Headers:
        X-Provider-UID: The unique identifier of the provider configuration to use
    """
    try:
        generator, mime_type, sample_rate = await audio_manager.speak(x_provider_uid, payload, stream=True)
        return StreamingResponse(
            generator,
            media_type=mime_type,
            headers={
                "X-Sample-Rate": str(sample_rate)
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logfire.error(f"Unexpected error in audio_speak_stream: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@audio_router.post("/transcribe")
async def audio_transcribe(
    audio_manager: AudioManagerType,
    x_provider_uid: str = Header(..., alias="X-Provider-UID", description="Provider UID configured via POST /config"),
    model: str = Form(...),
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    gen_config: Optional[str] = Form(None)
) -> TranscribeResponse:
    """
    Transcribe audio to text using the specified audio provider.
    Returns the transcribed text, language, and usage information.
    
    The provider UID must be configured first using POST /config endpoint.
    
    Headers:
        X-Provider-UID: The unique identifier of the provider configuration to use
    """
    try:
        # Validate content type
        if file.content_type and not file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail=f"Invalid file type. Expected audio file, got {file.content_type}")
        
        # Parse gen_config if provided
        parsed_gen_config = None
        if gen_config:
            try:
                parsed_gen_config = json.loads(gen_config)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON in gen_config: {str(e)}")
        
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
        result: TranscribeResponse = await audio_manager.transcribe(x_provider_uid, transcribe_request)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logfire.error(f"Unexpected error in audio_transcribe: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await file.close()