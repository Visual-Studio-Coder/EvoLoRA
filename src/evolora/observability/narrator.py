"""LiveKit and local narration helper, including terminal snapshots and dictation."""

from __future__ import annotations

import asyncio
import json
import subprocess
import wave
from datetime import datetime
from pathlib import Path

from evolora.config import get_config

# Lazy-loaded SpeechRecognition and LiveKit to handle missing dependencies gracefully
sr = None
rtc = None
api = None


def _init_imports():
    global sr, rtc, api
    if sr is None:
        try:
            import speech_recognition as sr_module
            sr = sr_module
        except ImportError:
            sr = False
    if rtc is None:
        try:
            from livekit import api as api_module
            from livekit import rtc as rtc_module
            rtc = rtc_module
            api = api_module
        except ImportError:
            rtc = False
            api = False


class Narrator:
    """Manages periodic terminal snapshots, narration commentary, and voice dictation."""

    def __init__(self) -> None:
        _init_imports()
        self.muted = False
        self._room = None
        self._last_state = ""
        self._last_agent_log = ""
        self._last_narrated_text = ""

    @property
    def is_livekit_ready(self) -> bool:
        return bool(rtc and api and get_config().livekit_available)

    async def connect(self) -> bool:
        """Connect to the LiveKit Room using configured credentials."""
        if not self.is_livekit_ready:
            return False

        cfg = get_config()
        try:
            token = (
                api.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
                .with_identity("evolora-narrator")
                .with_name("EvoLoRA Narrator")
                .with_grants(api.VideoGrants(room_join=True, room="evolora-session"))
                .to_jwt()
            )

            self._room = rtc.Room()
            await self._room.connect(cfg.livekit_url, token)
            return True
        except Exception as exc:
            # Safe boundary: log error and run local-only fallback
            print(f"[Narrator] LiveKit connection failed: {exc}")
            self._room = None
            return False

    async def disconnect(self) -> None:
        """Disconnect from the LiveKit room."""
        if self._room:
            try:
                await self._room.disconnect()
            except Exception:
                pass
            self._room = None

    def save_snapshot(
        self,
        state: str,
        metrics_text: str,
        agent_logs: str,
        examples_logs: str,
    ) -> str:
        """Saves a formatted snapshot of the terminal state to artifacts/terminal_snapshots.log."""
        cfg = get_config()
        log_dir = Path(cfg.artifact_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "terminal_snapshots.log"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        snapshot = (
            f"=== EVO LORA TERMINAL SNAPSHOT [{timestamp}] ===\n"
            f"State: {state}\n"
            f"Metrics:\n{metrics_text}\n"
            f"Last Agent Logs:\n{agent_logs}\n"
            f"Last Examples Logs:\n{examples_logs}\n"
            f"================================================\n\n"
        )

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(snapshot)

        return snapshot

    async def generate_commentary(
        self,
        state: str,
        metrics_text: str,
        new_logs: str,
    ) -> str:
        """Generates a 10-word commentary about what the agent is doing using MiniMax if available."""
        cfg = get_config()
        if cfg.minimax_available:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=cfg.minimax_api_key, base_url=cfg.minimax_base_url)
            
            payload = {
                "state": state,
                "metrics": metrics_text,
                "new_logs": new_logs,
            }
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful sports commentator narrating a machine learning "
                        "self-improvement run. Describe what the agent is doing right now. "
                        "Your response MUST be a short, single sentence of about 10 words. "
                        "Do not use any introductory tags or meta-text."
                    ),
                },
                {"role": "user", "content": json.dumps(payload)},
            ]
            try:
                resp = await client.chat.completions.create(
                    model=cfg.minimax_model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100,
                )
                commentary = (resp.choices[0].message.content or "").strip()
                if commentary:
                    return commentary
            except Exception:
                pass

        # Heuristic/Fallback commentary based on current state
        state_upper = state.upper()
        if "TRAINING" in state_upper:
            return "LoRA adapter training is actively running with low loss."
        elif "BASELINE" in state_upper:
            return "EvoLoRA self improvement is running baseline evaluations right now."
        elif "PLANNING" in state_upper:
            return "Strategy planning mode activated. MiniMax is designing hyperparameters."
        elif "JUDGE" in state_upper or "JUDGED" in state_upper:
            return "DigitalOcean judge is evaluating model outputs for compliance."
        elif "DONE" in state_upper or "BEST" in state_upper:
            return "Specialized LoRA adapter is fully trained and ready to use."
        elif "READY" in state_upper:
            return "EvoLoRA is idle and waiting to start the next run."
        return f"EvoLoRA agent is performing operations in state {state}."

    async def narrate_commentary(self, text: str) -> None:
        """Plays the commentary voice locally and streams/publishes it to the LiveKit room."""
        if self.muted or not text:
            return

        cfg = get_config()
        log_dir = Path(cfg.artifact_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        wav_path = log_dir / "narration.wav"

        # 1. Synthesize text to WAV using macOS 'say' command
        try:
            subprocess.run(
                ["say", "-o", str(wav_path), "--data-format=LEI16@22050", text],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"[Narrator] macOS synthesis failed: {exc}")
            return

        # 2. Play locally in background using macOS 'afplay'
        def play_local():
            try:
                subprocess.run(
                    ["afplay", str(wav_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

        # Run local playback in background thread
        asyncio.get_event_loop().run_in_executor(None, play_local)

        # 3. Stream to LiveKit Room if connected
        if self._room and self._room.isconnected:
            try:
                await self._stream_wav_to_livekit(str(wav_path))
            except Exception as exc:
                print(f"[Narrator] LiveKit streaming failed: {exc}")

    async def _stream_wav_to_livekit(self, wav_filepath: str) -> None:
        """Streams a WAV file's audio frames to the connected LiveKit room."""
        import numpy as np

        with wave.open(wav_filepath, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()

            source = rtc.AudioSource(sample_rate, channels)
            track = rtc.LocalAudioTrack.create_audio_track("narration-track", source)

            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            publication = await self._room.local_participant.publish_track(track, options)

            frame_duration_ms = 20
            num_samples = int(sample_rate * frame_duration_ms / 1000)

            try:
                while True:
                    frames = wav_file.readframes(num_samples)
                    if not frames:
                        break

                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    # Check if we got enough samples for this frame, pad with zeros if needed
                    expected_samples = num_samples * channels
                    if len(audio_data) < expected_samples:
                        pad_len = expected_samples - len(audio_data)
                        audio_data = np.pad(audio_data, (0, pad_len), "constant")

                    frame = rtc.AudioFrame(
                        data=audio_data.tobytes(),
                        sample_rate=sample_rate,
                        num_channels=channels,
                        samples_per_channel=num_samples,
                    )
                    await source.capture_frame(frame)
                    await asyncio.sleep(frame_duration_ms / 1000)
            finally:
                # Always clean up the track publication
                try:
                    await self._room.local_participant.unpublish_track(publication.sid)
                except Exception:
                    pass

    async def dictate(self, api_key: str = "") -> str:
        """Captures microphone audio locally and transcribes it to text."""
        _init_imports()
        if not sr:
            return "Error: SpeechRecognition package not installed or missing."

        def record_and_transcribe():
            r = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.8)
                    audio = r.listen(source, timeout=8, phrase_time_limit=12)
            except Exception as e:
                return f"Error opening microphone: {e}"

            try:
                if api_key:
                    return r.recognize_whisper_api(audio, api_key=api_key)
                else:
                    return r.recognize_google(audio)
            except Exception as e:
                return f"Transcription error: {e}"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, record_and_transcribe)
