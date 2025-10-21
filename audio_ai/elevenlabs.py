from elevenlabs import AsyncElevenLabs
from elevenlabs import SpeechToTextConvertResponse
from audio_ai.base import AudioAIService
from models.audio.speak import SpeakRequest, SpeakResponse
from typing import Optional, Tuple
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
from models.audio.audio import AudioUsage
import logfire


class ElevenLabsAudioAIService(AudioAIService):
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        if base_url:
            self.client = AsyncElevenLabs(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncElevenLabs(api_key=api_key)

    
    def decode_output_format(self, output_format: str) -> Tuple[str, int]:
        """
        elevenlabs output format is like this: mp3_16000_16_stereo
        or ulaw_8000
        """
        media_type, postfix = output_format.split('_')
        sample_rate, _ = postfix.split("_")
        mime_type = self.mime_type_map.get(media_type)
        if mime_type:
            return mime_type, int(sample_rate)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    
    @logfire.instrument(span_name="ElevenLabs Speak", record_return=True)
    async def speak(self, request: SpeakRequest) -> SpeakResponse:
        config = request.gen_config or {}
        mime_type, sample_rate = self.decode_output_format(request.output_format)
        speech = await self.client.text_to_speech.convert(
            text=request.text,
            voice_id=request.voice,
            model_id=request.settings.model,
            output_format=request.output_format,
            **config
        )
        speech_data = bytearray()
        for chunk in speech:
            speech_data += chunk
        return SpeakResponse(audio=speech_data, content_type=mime_type, sample_rate=sample_rate)
    
    
    @logfire.instrument(span_name="ElevenLabs Transcribe", record_return=True)
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        config = request.gen_config or {}
        transcription: SpeechToTextConvertResponse = await self.client.speech_to_text.convert(
            file=request.file,
            language_code=request.language,
            model_id=request.settings.model,
            **config
        )
        print(transcription)
        return TranscribeResponse(
            text=transcription.text, 
            language=transcription.language_code, 
            usage=AudioUsage(
                input_tokens=0, 
                output_tokens=0
            )
        )
