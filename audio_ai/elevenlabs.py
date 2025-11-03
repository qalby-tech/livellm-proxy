from elevenlabs import AsyncElevenLabs
from elevenlabs import SpeechToTextConvertResponse
from audio_ai.base import AudioAIService
from models.audio.speak import SpeakRequest, SpeakResponse, SpeakStreamResponse
from typing import Tuple, AsyncIterator
from models.audio.transcribe import TranscribeRequest, TranscribeResponse
import logfire


class ElevenLabsAudioAIService(AudioAIService):
    def __init__(self, client: AsyncElevenLabs):
        self.client = client

    def decode_output_format(self, output_format: str) -> Tuple[str, int]:
        """
        elevenlabs output format is like this: mp3_16000_16_stereo
        or ulaw_8000
        """
        media_type, postfix = output_format.split('_', maxsplit=1)
        if "_" in postfix:
            sample_rate, _ = postfix.split("_", maxsplit=1)
        else:
            sample_rate = postfix
        sample_rate = int(sample_rate)
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
            model_id=request.model,
            output_format=request.output_format,
            **config
        )
        speech_data = bytearray()
        for chunk in speech:
            speech_data += chunk
        return SpeakResponse(audio=speech_data, content_type=mime_type, sample_rate=sample_rate)
    

    @logfire.instrument(span_name="ElevenLabs Stream Speak", record_return=True)
    async def stream_speak(self, request: SpeakRequest) -> SpeakStreamResponse:
        config = request.gen_config or {}
        mime_type, sample_rate = self.decode_output_format(request.output_format)
        
        async def _elevenlabs_stream_speak_generator() -> AsyncIterator[bytes]:
            async for chunk in self.client.text_to_speech.stream(
                text=request.text,
                voice_id=request.voice,
                model_id=request.model,
                output_format=request.output_format,
                **config
            ):
                yield chunk
        
        generator = _elevenlabs_stream_speak_generator()
        first_chunk = await generator.__anext__()
        async def _generator(first_chunk: bytes, generator: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
            yield first_chunk
            async for chunk in generator:
                yield chunk
        return _generator(first_chunk, generator), mime_type, sample_rate
    
    @logfire.instrument(span_name="ElevenLabs Transcribe", record_return=True)
    async def transcribe(self, request: TranscribeRequest) -> TranscribeResponse:
        config = request.gen_config or {}
        transcription: SpeechToTextConvertResponse = await self.client.speech_to_text.convert(
            file=request.file,
            language_code=request.language,
            model_id=request.model,
            **config
        )
        return TranscribeResponse(
            text=transcription.text, 
            language=transcription.language_code
        )
