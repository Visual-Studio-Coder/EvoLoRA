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

    # Persistence
    mongodb_uri: str = Field(default="")
    artifact_store: str = Field(default="local")
    artifact_dir: str = Field(default="./artifacts")

    # Loop control
    max_iterations: int = Field(default=3)
    target_score: float = Field(default=0.85)
    improvement_threshold: float = Field(default=0.01)
    patience: int = Field(default=2)

    # DigitalOcean
    digitalocean_inference_base_url: str = Field(
        default="https://inference.do-ai.run/v1/"
    )
    digital_ocean_model_access_key: str = Field(default="")
    digital_ocean_judge_model: str = Field(default="llama3.3-70b-instruct")
    digitalocean_token: str = Field(default="")

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


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config(
        minimax_api_key=os.getenv("MINIMAX_API_KEY", ""),
        minimax_model=os.getenv("MINIMAX_MODEL", "MiniMax-M2.7-highspeed"),
        minimax_base_url=os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io/v1"),
        minimax_group_id=os.getenv("MINIMAX_GROUP_ID", ""),
        training_backend=os.getenv("TRAINING_BACKEND", "mock"),
        model_runner=os.getenv("MODEL_RUNNER", "mock"),
        mongodb_uri=os.getenv("MONGODB_URI", ""),
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
    )
