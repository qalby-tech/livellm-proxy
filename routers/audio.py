from fastapi import APIRouter, Depends, Request, Response, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.exceptions import HTTPException
from typing import Annotated, Optional, AsyncIterator
import json
from managers.audio import AudioManager
from models.audio.speak import SpeakRequest, SpeakResponse
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from models.audio.audio import AudioSettings

audio_router = APIRouter(prefix="/audio", tags=["audio"])


def get_audio_manager(request: Request) -> AudioManager:
    return request.app.state.audio_manager


AudioManagerType = Annotated[AudioManager, Depends(get_audio_manager)]


@audio_router.post("/speak", response_class=Response)
async def audio_speak(
    payload: SpeakRequest,
    audio_manager: AudioManagerType
) -> Response:
    """
    Convert text to speech using the specified audio provider.
    Returns raw audio bytes with appropriate content type.
    """
    try:
        result: SpeakResponse = await audio_manager.speak(payload)
        return Response(
            content=result.audio,
            media_type=result.content_type,
            headers={
                "X-Sample-Rate": str(result.sample_rate)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@audio_router.post("/speak_stream", response_class=Response)
async def audio_speak_stream(
    payload: SpeakRequest,
    audio_manager: AudioManagerType
) -> Response:
    """
    Convert text to speech using the specified audio provider.
    Returns raw audio bytes with appropriate content type.
    """
    try:
        generator, mime_type, sample_rate = await audio_manager.speak(payload, stream=True)
        return StreamingResponse(
            generator,
            media_type=mime_type,
            headers={
                "X-Sample-Rate": str(sample_rate)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@audio_router.post("/transcribe")
async def audio_transcribe(
    audio_manager: AudioManagerType,
    provider: str = Form(...),
    api_key: str = Form(...),
    model: str = Form(...),
    file: UploadFile = File(...),
    base_url: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    gen_config: Optional[str] = Form(None)
) -> TranscribeResponse:
    """
    Transcribe audio to text using the specified audio provider.
    Returns the transcribed text, language, and usage information.
    """
    try:
        # Create AudioSettings from individual form fields
        audio_settings = AudioSettings(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url
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
            settings=audio_settings,
            file=file_tuple,
            language=language,
            gen_config=parsed_gen_config
        )
        
        # Call the transcribe method on the manager
        result: TranscribeResponse = await audio_manager.transcribe(transcribe_request)
        return result
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in gen_config: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))