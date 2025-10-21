from fastapi import APIRouter, Depends, Request, Response
from fastapi.exceptions import HTTPException
from typing import Annotated
from managers.audio import AudioManager
from models.audio.speak import SpeakRequest, SpeakResponse


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

