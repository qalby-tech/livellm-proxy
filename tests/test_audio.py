"""Tests for audio endpoints."""

import pytest
from models.audio.speak import SpeakResponse
from models.audio.transcribe import TranscribeResponse


class TestAudioSpeak:
    """Tests for the /audio/speak endpoint."""

    def test_audio_speak_success(self, client, mock_audio_manager, audio_speak_payload, audio_headers):
        """Test successful text-to-speech conversion."""
        # Setup mock
        mock_audio_manager.speak.return_value = SpeakResponse(
            audio=b"fake_audio_data",
            content_type="audio/mpeg",
            sample_rate=24000
        )

        # Make request
        response = client.post(
            "/audio/speak",
            json=audio_speak_payload,
            headers=audio_headers
        )

        # Assertions
        assert response.status_code == 200
        assert response.content == b"fake_audio_data"
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.headers["x-sample-rate"] == "24000"
        
        # Verify manager was called
        mock_audio_manager.speak.assert_called_once()

    def test_audio_speak_with_base_url(self, client, mock_audio_manager, audio_speak_payload, audio_headers):
        """Test text-to-speech with custom base URL."""
        mock_audio_manager.speak.return_value = SpeakResponse(
            audio=b"audio_data",
            content_type="audio/mpeg",
            sample_rate=24000
        )

        headers = {**audio_headers, "X-Base-Url": "https://custom.api.com"}
        response = client.post(
            "/audio/speak",
            json=audio_speak_payload,
            headers=headers
        )

        assert response.status_code == 200
        mock_audio_manager.speak.assert_called_once()

    def test_audio_speak_missing_api_key(self, client, audio_speak_payload):
        """Test speak without API key header."""
        headers = {"X-Provider": "openai"}
        response = client.post(
            "/audio/speak",
            json=audio_speak_payload,
            headers=headers
        )
        
        assert response.status_code == 422

    def test_audio_speak_missing_provider(self, client, audio_speak_payload):
        """Test speak without provider header."""
        headers = {"X-Api-Key": "test-key"}
        response = client.post(
            "/audio/speak",
            json=audio_speak_payload,
            headers=headers
        )
        
        assert response.status_code == 422

    def test_audio_speak_invalid_provider(self, client, mock_audio_manager, audio_speak_payload):
        """Test speak with invalid provider."""
        mock_audio_manager.speak.side_effect = ValueError("Invalid provider")

        headers = {"X-Api-Key": "test-key", "X-Provider": "invalid_provider"}
        response = client.post(
            "/audio/speak",
            json=audio_speak_payload,
            headers=headers
        )
        
        assert response.status_code == 400
        assert "Invalid provider" in response.json()["detail"]

    def test_audio_speak_different_providers(self, client, mock_audio_manager, audio_speak_payload):
        """Test speak with different audio providers."""
        mock_audio_manager.speak.return_value = SpeakResponse(
            audio=b"audio_data",
            content_type="audio/mpeg",
            sample_rate=24000
        )

        for provider in ["openai", "elevenlabs"]:
            headers = {"X-Api-Key": "test-key", "X-Provider": provider}
            response = client.post(
                "/audio/speak",
                json=audio_speak_payload,
                headers=headers
            )
            assert response.status_code == 200

    def test_audio_speak_invalid_payload(self, client, audio_headers):
        """Test speak with invalid payload."""
        invalid_payload = {
            "model": "tts-1"
            # Missing required fields
        }
        
        response = client.post(
            "/audio/speak",
            json=invalid_payload,
            headers=audio_headers
        )
        
        assert response.status_code == 422

    def test_audio_speak_with_gen_config(self, client, mock_audio_manager, audio_headers):
        """Test speak with generation config."""
        mock_audio_manager.speak.return_value = SpeakResponse(
            audio=b"audio_data",
            content_type="audio/mpeg",
            sample_rate=24000
        )

        payload = {
            "model": "tts-1",
            "text": "Hello world",
            "voice": "alloy",
            "output_format": "mp3",
            "gen_config": {"speed": 1.5}
        }

        response = client.post(
            "/audio/speak",
            json=payload,
            headers=audio_headers
        )

        assert response.status_code == 200


class TestAudioSpeakStream:
    """Tests for the /audio/speak_stream endpoint."""

    def test_audio_speak_stream_success(self, client, mock_audio_manager, audio_speak_payload, audio_headers):
        """Test successful streaming text-to-speech."""
        # Setup mock async generator
        async def mock_stream():
            yield b"chunk1"
            yield b"chunk2"
            yield b"chunk3"
        
        mock_audio_manager.speak.return_value = (mock_stream(), "audio/mpeg", 24000)

        # Make request
        response = client.post(
            "/audio/speak_stream",
            json=audio_speak_payload,
            headers=audio_headers
        )

        # Assertions
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/mpeg"
        assert response.headers["x-sample-rate"] == "24000"
        
        # Verify manager was called with stream=True
        mock_audio_manager.speak.assert_called_once()
        call_kwargs = mock_audio_manager.speak.call_args[1]
        assert call_kwargs["stream"] is True

    def test_audio_speak_stream_with_base_url(self, client, mock_audio_manager, audio_speak_payload, audio_headers):
        """Test streaming with custom base URL."""
        async def mock_stream():
            yield b"audio_chunk"
        
        mock_audio_manager.speak.return_value = (mock_stream(), "audio/mpeg", 24000)

        headers = {**audio_headers, "X-Base-Url": "https://custom.api.com"}
        response = client.post(
            "/audio/speak_stream",
            json=audio_speak_payload,
            headers=headers
        )

        assert response.status_code == 200

    def test_audio_speak_stream_missing_headers(self, client, audio_speak_payload):
        """Test streaming without required headers."""
        response = client.post(
            "/audio/speak_stream",
            json=audio_speak_payload
        )
        
        assert response.status_code == 422

    def test_audio_speak_stream_invalid_provider(self, client, mock_audio_manager, audio_speak_payload):
        """Test streaming with invalid provider."""
        mock_audio_manager.speak.side_effect = ValueError("Invalid provider")

        headers = {"X-Api-Key": "test-key", "X-Provider": "invalid"}
        response = client.post(
            "/audio/speak_stream",
            json=audio_speak_payload,
            headers=headers
        )
        
        assert response.status_code == 400


class TestAudioTranscribe:
    """Tests for the /audio/transcribe endpoint."""

    def test_audio_transcribe_success(self, client, mock_audio_manager, audio_headers):
        """Test successful audio transcription."""
        # Setup mock
        mock_audio_manager.transcribe.return_value = TranscribeResponse(
            text="This is the transcribed text",
            language="en"
        )

        # Create fake audio file
        audio_content = b"fake_audio_content"
        files = {"file": ("test_audio.mp3", audio_content, "audio/mpeg")}
        data = {"model": "whisper-1"}

        # Make request
        response = client.post(
            "/audio/transcribe",
            files=files,
            data=data,
            headers=audio_headers
        )

        # Assertions
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "This is the transcribed text"
        assert data["language"] == "en"
        
        # Verify manager was called
        mock_audio_manager.transcribe.assert_called_once()

    def test_audio_transcribe_with_language(self, client, mock_audio_manager, audio_headers):
        """Test transcription with specified language."""
        mock_audio_manager.transcribe.return_value = TranscribeResponse(
            text="Bonjour le monde",
            language="fr"
        )

        files = {"file": ("test_audio.mp3", b"fake_audio", "audio/mpeg")}
        data = {"model": "whisper-1", "language": "fr"}

        response = client.post(
            "/audio/transcribe",
            files=files,
            data=data,
            headers=audio_headers
        )

        assert response.status_code == 200
        assert response.json()["language"] == "fr"

    def test_audio_transcribe_with_gen_config(self, client, mock_audio_manager, audio_headers):
        """Test transcription with generation config."""
        mock_audio_manager.transcribe.return_value = TranscribeResponse(
            text="Transcribed text",
            language="en"
        )

        files = {"file": ("test_audio.mp3", b"fake_audio", "audio/mpeg")}
        data = {
            "model": "whisper-1",
            "gen_config": '{"temperature": 0.5}'
        }

        response = client.post(
            "/audio/transcribe",
            files=files,
            data=data,
            headers=audio_headers
        )

        assert response.status_code == 200

    def test_audio_transcribe_invalid_gen_config(self, client, mock_audio_manager, audio_headers):
        """Test transcription with invalid JSON in gen_config."""
        files = {"file": ("test_audio.mp3", b"fake_audio", "audio/mpeg")}
        data = {
            "model": "whisper-1",
            "gen_config": 'invalid json'
        }

        response = client.post(
            "/audio/transcribe",
            files=files,
            data=data,
            headers=audio_headers
        )

        # The error message could be about invalid JSON or invalid provider validation
        assert response.status_code == 400

    def test_audio_transcribe_with_base_url(self, client, mock_audio_manager, audio_headers):
        """Test transcription with custom base URL."""
        mock_audio_manager.transcribe.return_value = TranscribeResponse(
            text="Transcribed text",
            language="en"
        )

        headers = {**audio_headers, "X-Base-Url": "https://custom.api.com"}
        files = {"file": ("test_audio.mp3", b"fake_audio", "audio/mpeg")}
        data = {"model": "whisper-1"}

        response = client.post(
            "/audio/transcribe",
            files=files,
            data=data,
            headers=headers
        )

        assert response.status_code == 200

    def test_audio_transcribe_missing_file(self, client, audio_headers):
        """Test transcription without file."""
        data = {"model": "whisper-1"}

        response = client.post(
            "/audio/transcribe",
            data=data,
            headers=audio_headers
        )

        assert response.status_code == 422

    def test_audio_transcribe_missing_model(self, client, audio_headers):
        """Test transcription without model."""
        files = {"file": ("test_audio.mp3", b"fake_audio", "audio/mpeg")}

        response = client.post(
            "/audio/transcribe",
            files=files,
            headers=audio_headers
        )

        assert response.status_code == 422

    def test_audio_transcribe_missing_headers(self, client):
        """Test transcription without required headers."""
        files = {"file": ("test_audio.mp3", b"fake_audio", "audio/mpeg")}
        data = {"model": "whisper-1"}

        response = client.post(
            "/audio/transcribe",
            files=files,
            data=data
        )

        assert response.status_code == 422

    def test_audio_transcribe_invalid_provider(self, client, mock_audio_manager):
        """Test transcription with invalid provider."""
        mock_audio_manager.transcribe.side_effect = ValueError("Invalid provider")

        headers = {"X-Api-Key": "test-key", "X-Provider": "invalid_provider"}
        files = {"file": ("test_audio.mp3", b"fake_audio", "audio/mpeg")}
        data = {"model": "whisper-1"}

        response = client.post(
            "/audio/transcribe",
            files=files,
            data=data,
            headers=headers
        )

        assert response.status_code == 400
        assert "Invalid provider" in response.json()["detail"]

    def test_audio_transcribe_different_file_types(self, client, mock_audio_manager, audio_headers):
        """Test transcription with different audio file types."""
        mock_audio_manager.transcribe.return_value = TranscribeResponse(
            text="Transcribed text",
            language="en"
        )

        file_types = [
            ("test.mp3", "audio/mpeg"),
            ("test.wav", "audio/wav"),
            ("test.m4a", "audio/m4a"),
            ("test.ogg", "audio/ogg")
        ]

        for filename, content_type in file_types:
            files = {"file": (filename, b"fake_audio", content_type)}
            data = {"model": "whisper-1"}

            response = client.post(
                "/audio/transcribe",
                files=files,
                data=data,
                headers=audio_headers
            )

            assert response.status_code == 200


class TestAudioErrorHandling:
    """Tests for error handling in audio endpoints."""

    def test_speak_internal_error(self, client, mock_audio_manager, audio_speak_payload, audio_headers):
        """Test handling of internal errors in speak endpoint."""
        mock_audio_manager.speak.side_effect = Exception("Internal service error")

        response = client.post(
            "/audio/speak",
            json=audio_speak_payload,
            headers=audio_headers
        )

        assert response.status_code == 500
        assert "Internal service error" in response.json()["detail"]

    def test_transcribe_internal_error(self, client, mock_audio_manager, audio_headers):
        """Test handling of internal errors in transcribe endpoint."""
        mock_audio_manager.transcribe.side_effect = Exception("Internal service error")

        files = {"file": ("test_audio.mp3", b"fake_audio", "audio/mpeg")}
        data = {"model": "whisper-1"}

        response = client.post(
            "/audio/transcribe",
            files=files,
            data=data,
            headers=audio_headers
        )

        assert response.status_code == 500
        assert "Internal service error" in response.json()["detail"]
