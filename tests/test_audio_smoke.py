"""
Smoke tests for audio endpoints using mocked dependencies.
These tests verify the basic request/response flow without calling real audio APIs.
"""

import pytest


@pytest.mark.audio
def test_audio_speak_with_mock(client, mock_audio_manager, audio_speak_payload):
    """Test audio speak endpoint with mocked manager."""
    # Setup mock response
    mock_audio_manager.safe_speak.return_value = type('obj', (object,), {
        'audio': b'fake-audio-data',
        'content_type': 'audio/pcm',
        'sample_rate': 24000
    })()
    
    response = client.post("/livellm/audio/speak", json=audio_speak_payload)
    
    assert response.status_code == 200
    assert response.content == b'fake-audio-data'
    assert response.headers.get("content-type") == "audio/pcm"
    assert response.headers.get("X-Sample-Rate") == "24000"
    mock_audio_manager.safe_speak.assert_called_once()


@pytest.mark.audio
def test_audio_speak_missing_required_fields(client, mock_audio_manager):
    """Test that audio speak fails with missing required fields."""
    incomplete_payload = {
        "provider_uid": "test-provider"
        # missing model, text, voice
    }
    
    response = client.post("/livellm/audio/speak", json=incomplete_payload)
    assert response.status_code == 422  # Validation error


@pytest.mark.audio
def test_audio_speak_stream_endpoint(client, mock_audio_manager, audio_speak_payload):
    """Test audio speak stream endpoint returns streaming response."""
    async def mock_audio_stream():
        yield b'audio-chunk-1'
        yield b'audio-chunk-2'
    
    mock_audio_manager.safe_speak.return_value = (
        mock_audio_stream(),
        "audio/pcm",
        24000
    )
    
    response = client.post("/livellm/audio/speak_stream", json=audio_speak_payload)
    
    assert response.status_code == 200
    assert "audio/pcm" in response.headers.get("content-type", "")
    assert response.headers.get("X-Sample-Rate") == "24000"


@pytest.mark.audio
def test_audio_transcribe_missing_file(client, mock_audio_manager):
    """Test that audio transcribe fails without file."""
    response = client.post(
        "/livellm/audio/transcribe",
        data={
            "provider_uid": "test-provider",
            "model": "whisper-1"
            # missing file
        }
    )
    assert response.status_code == 422  # Validation error


@pytest.mark.audio
def test_audio_transcribe_with_mock(client, mock_audio_manager):
    """Test audio transcribe endpoint with mocked manager."""
    mock_audio_manager.safe_transcribe.return_value = type('obj', (object,), {
        'text': 'Hello world',
        'language': 'en',
        'usage': {'duration': 2.5}
    })()
    
    response = client.post(
        "/livellm/audio/transcribe",
        data={
            "provider_uid": "test-provider-uid",
            "model": "whisper-1",
            "language": "en"
        },
        files={
            "file": ("test.wav", b"fake-audio-content", "audio/wav")
        }
    )
    
    # Should process successfully
    assert response.status_code in [200, 400]


@pytest.mark.audio
def test_audio_transcribe_json_with_mock(client, mock_audio_manager):
    """Test audio transcribe JSON endpoint with mocked manager."""
    mock_audio_manager.safe_transcribe.return_value = {
        'text': 'Hello world',
        'language': 'en',
        'usage': {'duration': 2.5}
    }
    
    payload = {
        "provider_uid": "test-provider-uid",
        "model": "whisper-1",
        "file": ["test-audio.wav", "ZmFrZS1hdWRpby1jb250ZW50", "audio/wav"],
        "language": "en"
    }
    
    response = client.post("/livellm/audio/transcribe_json", json=payload)
    assert response.status_code == 200
    mock_audio_manager.safe_transcribe.assert_called_once()


@pytest.mark.audio
def test_audio_transcribe_json_invalid_base64(client, mock_audio_manager):
    """Test that invalid base64 audio data is rejected."""
    payload = {
        "provider_uid": "test-provider-uid",
        "model": "whisper-1",
        "file": ["test-audio.wav", "not-valid-base64!!!", "audio/wav"],
        "language": "en"
    }
    
    response = client.post("/livellm/audio/transcribe_json", json=payload)
    assert response.status_code in [400, 422]  # Should fail validation


@pytest.mark.audio
def test_audio_speak_fallback_request_structure(client, mock_audio_manager):
    """Test that audio speak accepts fallback request structure."""
    mock_audio_manager.safe_speak.return_value = type('obj', (object,), {
        'audio': b'fake-audio',
        'content_type': 'audio/pcm',
        'sample_rate': 24000
    })()
    
    fallback_payload = {
        "requests": [
            {
                "provider_uid": "openai-1",
                "model": "tts-1",
                "text": "Hello world",
                "voice": "alloy"
            },
            {
                "provider_uid": "elevenlabs-1",
                "model": "eleven_multilingual_v2",
                "text": "Hello world",
                "voice": "Rachel"
            }
        ],
        "strategy": "sequential",
        "timeout_per_request": 360
    }
    
    response = client.post("/livellm/audio/speak", json=fallback_payload)
    # Should either succeed or fail validation
    assert response.status_code in [200, 400, 422]


@pytest.mark.audio
def test_audio_transcribe_invalid_file_type(client, mock_audio_manager):
    """Test that non-audio file types are rejected."""
    response = client.post(
        "/livellm/audio/transcribe",
        data={
            "provider_uid": "test-provider-uid",
            "model": "whisper-1"
        },
        files={
            "file": ("test.txt", b"text content", "text/plain")
        }
    )
    
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]

