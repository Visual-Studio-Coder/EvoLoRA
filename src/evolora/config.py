"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Config(BaseModel):
    # MiniMax
    minimax_api_key: str = Field(default="")
    minimax_model: str = Field(default="MiniMax-M2.7-highspeed")
    minimax_base_url: str = Field(default="https://api.minimax.io/v1")
    minimax_group_id: str = Field(default="")

    # Training
    training_backend: str = Field(default="mock")  # mock | unsloth | remote
    model_runner: str = Field(default="mock")       # mock | local | remote
    base_model_id: str = Field(default="microsoft/Phi-3-mini-128k-instruct")

    # Remote GPU config push
    ssh_host: str = Field(default="")
    ssh_user: str = Field(default="")
    ssh_port: int = Field(default=22)
    ssh_key_path: str = Field(default="")
    remote_config_path: str = Field(default="/workspace/config.json")
    remote_results_path: str = Field(default="/workspace/generations/results.json")

    # Persistence
    mongodb_uri: str = Field(default="")
    mongodb_db_name: str = Field(default="evolora")
    mongodb_runs_collection: str = Field(default="runs")
    mongodb_server_selection_timeout_ms: int = Field(default=3000)
    artifact_store: str = Field(default="local")
    artifact_dir: str = Field(default="./artifacts")

    # Loop control
    max_iterations: int = Field(default=3)
    target_score: float = Field(default=0.85)
    improvement_threshold: float = Field(default=0.01)
    patience: int = Field(default=2)
    # AUTO_APPROVE=true -> fully autonomous (no approval gates).
    auto_approve: bool = Field(default=False)

    # DigitalOcean
    digitalocean_inference_base_url: str = Field(
        default="https://inference.do-ai.run/v1/"
    )
    digital_ocean_model_access_key: str = Field(default="")
    digital_ocean_judge_model: str = Field(default="llama3.3-70b-instruct")
    digitalocean_token: str = Field(default="")

    # Voice (LiveKit Inference STT/TTS) — optional, fully decoupled from training.
    livekit_url: str = Field(default="")
    livekit_api_key: str = Field(default="")
    livekit_api_secret: str = Field(default="")
    voice_enabled: bool = Field(default=True)
    stt_model: str = Field(default="deepgram/nova-3")
    tts_model: str = Field(default="cartesia/sonic-3")
    tts_voice: str = Field(default="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")
    narrate_interval: float = Field(default=30.0)
    narrate_polish: bool = Field(default=True)  # rephrase template lines via MiniMax when available
    ptt_key: str = Field(default="f9")  # global push-to-talk key (pynput name)

    @property
    def mock_mode(self) -> bool:
        return self.training_backend == "mock"

    @property
    def minimax_available(self) -> bool:
        return bool(self.minimax_api_key)

    @property
    def mongo_available(self) -> bool:
        return bool(self.mongodb_uri)

    @property
    def digital_ocean_judge_available(self) -> bool:
        return bool(self.digital_ocean_model_access_key)

    @property
    def voice_available(self) -> bool:
        """Voice can run only when enabled and all LiveKit creds are present.

        Package import + audio-device availability are checked at runtime by the
        VoiceService; this just gates on config so the TUI can decide whether to try.
        """
        return bool(
            self.voice_enabled
            and self.livekit_url
            and self.livekit_api_key
            and self.livekit_api_secret
        )


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config(
        minimax_api_key=os.getenv("MINIMAX_API_KEY", ""),
        minimax_model=os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
        minimax_base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
        minimax_group_id=os.getenv("MINIMAX_GROUP_ID", ""),
        training_backend=os.getenv("TRAINING_BACKEND", "mock"),
        model_runner=os.getenv("MODEL_RUNNER", "mock"),
        ssh_host=os.getenv("SSH_HOST", ""),
        ssh_user=os.getenv("SSH_USER", ""),
        ssh_port=int(os.getenv("SSH_PORT", "22")),
        ssh_key_path=os.getenv("SSH_KEY_PATH", ""),
        remote_config_path=os.getenv("REMOTE_CONFIG_PATH", "/workspace/config.json"),
        remote_results_path=os.getenv(
            "REMOTE_RESULTS_PATH", "/workspace/generations/results.json"
        ),
        mongodb_uri=os.getenv("MONGODB_URI", ""),
        mongodb_db_name=os.getenv("MONGODB_DB_NAME", "evolora"),
        mongodb_runs_collection=os.getenv("MONGODB_RUNS_COLLECTION", "runs"),
        mongodb_server_selection_timeout_ms=int(
            os.getenv("MONGODB_SERVER_SELECTION_TIMEOUT_MS", "3000")
        ),
        artifact_store=os.getenv("ARTIFACT_STORE", "local"),
        artifact_dir=os.getenv("ARTIFACT_DIR", "./artifacts"),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "3")),
        target_score=float(os.getenv("TARGET_SCORE", "0.85")),
        improvement_threshold=float(os.getenv("IMPROVEMENT_THRESHOLD", "0.01")),
        patience=int(os.getenv("PATIENCE", "2")),
        digitalocean_inference_base_url=os.getenv(
            "DIGITAL_OCEAN_INFERENCE_BASE_URL", "https://inference.do-ai.run/v1/"
        ),
        digital_ocean_model_access_key=os.getenv("DIGITAL_OCEAN_MODEL_ACCESS_KEY", ""),
        digital_ocean_judge_model=os.getenv(
            "DIGITAL_OCEAN_JUDGE_MODEL", "llama3.3-70b-instruct"
        ),
        base_model_id=os.getenv("BASE_MODEL_ID", "microsoft/Phi-3-mini-128k-instruct"),
        digitalocean_token=os.getenv("DIGITALOCEAN_TOKEN", ""),
        auto_approve=os.getenv("AUTO_APPROVE", "false").strip().lower() in {"1", "true", "yes", "on"},
        livekit_url=os.getenv("LIVEKIT_URL", ""),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
        voice_enabled=os.getenv("VOICE_ENABLED", "true").strip().lower()
        not in {"0", "false", "off", "no"},
        stt_model=os.getenv("STT_MODEL", "deepgram/nova-3"),
        tts_model=os.getenv("TTS_MODEL", "cartesia/sonic-3"),
        tts_voice=os.getenv("TTS_VOICE", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        narrate_interval=float(os.getenv("NARRATE_INTERVAL", "30")),
        narrate_polish=os.getenv("NARRATE_POLISH", "true").strip().lower()
        not in {"0", "false", "off", "no"},
        ptt_key=os.getenv("PTT_KEY", "f9"),
    )
