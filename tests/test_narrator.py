import asyncio
import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evolora.config import Config
from evolora.observability.narrator import Narrator


@pytest.fixture
def mock_config(monkeypatch):
    cfg = Config(
        livekit_url="wss://test.livekit.cloud",
        livekit_api_key="test_key",
        livekit_api_secret="test_secret",
        minimax_api_key="minimax_test",
        artifact_dir="./test_artifacts",
    )
    import evolora.observability.narrator as narrator_mod
    monkeypatch.setattr(narrator_mod, "get_config", lambda: cfg)
    return cfg


def test_narrator_is_livekit_ready(mock_config):
    narrator = Narrator()
    # If rtc is mocked/available and config is set, it should be ready
    with patch("evolora.observability.narrator.rtc", True), \
         patch("evolora.observability.narrator.api", True):
        assert narrator.is_livekit_ready is True

    with patch("evolora.observability.narrator.rtc", None), \
         patch("evolora.observability.narrator.api", None):
        assert narrator.is_livekit_ready is False


@pytest.mark.asyncio
async def test_narrator_connect_and_disconnect(mock_config):
    narrator = Narrator()

    mock_room = MagicMock()
    mock_room.connect = MagicMock(return_value=asyncio.sleep(0))
    mock_room.disconnect = MagicMock(return_value=asyncio.sleep(0))

    mock_api = MagicMock()
    mock_api_token = MagicMock()
    mock_api_token.with_identity.return_value = mock_api_token
    mock_api_token.with_name.return_value = mock_api_token
    mock_api_token.with_grants.return_value = mock_api_token
    mock_api_token.to_jwt.return_value = "fake_jwt"
    mock_api.AccessToken.return_value = mock_api_token

    with patch("evolora.observability.narrator.rtc") as mock_rtc, \
         patch("evolora.observability.narrator.api", mock_api):
        mock_rtc.Room.return_value = mock_room
        # Test connection success
        success = await narrator.connect()
        assert success is True
        assert narrator._room == mock_room

        # Test disconnect
        await narrator.disconnect()
        assert narrator._room is None
        mock_room.disconnect.assert_called_once()


def test_narrator_save_snapshot(mock_config, tmp_path):
    mock_config.artifact_dir = str(tmp_path)
    narrator = Narrator()

    snapshot = narrator.save_snapshot(
        state="RUNNING",
        metrics_text="loss: 0.1",
        agent_logs="Line 1\nLine 2",
        examples_logs="Ex 1\nEx 2",
    )

    assert "State: RUNNING" in snapshot
    assert "loss: 0.1" in snapshot
    assert "Line 1" in snapshot
    assert "Ex 1" in snapshot

    log_file = tmp_path / "terminal_snapshots.log"
    assert log_file.exists()
    file_content = log_file.read_text(encoding="utf-8")
    assert snapshot in file_content


@pytest.mark.asyncio
async def test_narrator_generate_commentary_fallback(mock_config):
    mock_config.minimax_api_key = ""  # No minimax key to trigger fallbacks
    narrator = Narrator()

    commentary_train = await narrator.generate_commentary("TRAINING", "loss: 0.05", "step 10")
    assert "training" in commentary_train.lower()

    commentary_baseline = await narrator.generate_commentary("BASELINE", "baseline score", "done")
    assert "baseline" in commentary_baseline.lower()

    commentary_planning = await narrator.generate_commentary("PLANNING", "planning next step", "started")
    assert "planning" in commentary_planning.lower()


@pytest.mark.asyncio
async def test_narrator_generate_commentary_minimax(mock_config):
    narrator = Narrator()

    mock_choices = MagicMock()
    mock_choices.message.content = "EvoLoRA is optimizing parameters for the LoRA run."
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choices]

    mock_chat = MagicMock()
    mock_chat.completions.create = MagicMock(return_value=asyncio.sleep(0, result=mock_resp))

    mock_client = MagicMock()
    mock_client.chat = mock_chat

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        commentary = await narrator.generate_commentary("TRAINING", "loss: 0.1", "running")
        assert commentary == "EvoLoRA is optimizing parameters for the LoRA run."


@pytest.mark.asyncio
async def test_narrator_narrate_commentary(mock_config, tmp_path):
    mock_config.artifact_dir = str(tmp_path)
    narrator = Narrator()

    with patch("subprocess.run") as mock_run:
        # Test muted
        narrator.muted = True
        await narrator.narrate_commentary("Hello world")
        mock_run.assert_not_called()

        # Test unmuted and calling 'say' and 'afplay'
        narrator.muted = False
        await narrator.narrate_commentary("Hello world")

        # Let async executor finish playing
        await asyncio.sleep(0.1)

        # subprocess.run should be called for 'say' and 'afplay'
        assert mock_run.call_count >= 2
        calls = [call[0][0] for call in mock_run.call_args_list]
        assert any(c[0] == "say" for c in calls)
        assert any(c[0] == "afplay" for c in calls)


@pytest.mark.asyncio
async def test_narrator_dictate_fallback(mock_config):
    narrator = Narrator()
    # Dictate fallback when speech_recognition is missing
    with patch("evolora.observability.narrator.sr", False):
        res = await narrator.dictate()
        assert "SpeechRecognition package not installed" in res

    # Dictate fallback Google Web Speech API when key is absent
    mock_sr = MagicMock()
    mock_mic = MagicMock()
    mock_sr.Microphone.return_value = mock_mic
    mock_recognizer = MagicMock()
    mock_recognizer.recognize_google.return_value = "transcribed voice prompt"
    mock_sr.Recognizer.return_value = mock_recognizer

    with patch("evolora.observability.narrator.sr", mock_sr):
        res = await narrator.dictate()
        assert res == "transcribed voice prompt"
        mock_recognizer.recognize_google.assert_called_once()
