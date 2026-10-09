from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "LECTURELENS_", "env_file": ".env", "env_file_encoding": "utf-8"}

    redis_url: str = Field(default="redis://localhost:6379")
    ollama_url: str = Field(default="http://localhost:11434")

    tesseract_path: str = Field(default="C:\\Program Files\\Tesseract-OCR\\tesseract.exe")

    output_dir: Path = Field(default=Path("./output"))
    router_config_path: Path = Field(default=Path("./router.yaml"))

    audio_source: str = Field(default="mic")
    audio_only: bool = Field(default=False)
    active_agents: list[str] = Field(
        default=["audio", "screen", "camera", "sync", "structuring"]
    )

    session_name: Optional[str] = Field(default=None)

    mic_sample_rate: int = Field(default=16000)
    mic_chunk_duration_seconds: float = Field(default=3.0)
    vad_energy_threshold: float = Field(default=0.01)

    whisper_model_size: str = Field(default="base")
    whisper_device: str = Field(default="cpu")

    screenshot_interval_seconds: float = Field(default=1.0)
    slide_diff_threshold: float = Field(default=0.05)
    monitor_index: int = Field(default=0)

    camera_device_index: int = Field(default=0)
    camera_poll_interval_seconds: float = Field(default=0.5)
    motion_threshold: float = Field(default=0.02)
    blur_variance_threshold: float = Field(default=100.0)
    motion_stillness_required_frames: int = Field(default=5)

    structuring_interval_seconds: int = Field(default=120)
    correlation_window_seconds: float = Field(default=5.0)
    sync_merge_window_seconds: float = Field(default=2.0)

    gateway_host: str = Field(default="0.0.0.0")
    gateway_port: int = Field(default=8765)
    gateway_api_key: Optional[str] = Field(default=None)

    thread_pool_max_workers: int = Field(default=4)
    process_pool_max_workers: int = Field(default=2)

    redis_consumer_batch_size: int = Field(default=10)
    redis_consumer_block_ms: int = Field(default=2000)

    llm_mode: str = Field(default="hybrid")

    def get_session_output_dir(self, session_id: str) -> Path:
        session_dir = self.output_dir / session_id
        return session_dir

    def create_session_directories(self, session_id: str) -> Path:
        session_dir = self.get_session_output_dir(session_id)

        raw_audio_dir = session_dir / "raw" / "audio"
        raw_slides_dir = session_dir / "raw" / "slides"
        raw_board_dir = session_dir / "raw" / "board"

        raw_audio_dir.mkdir(parents=True, exist_ok=True)
        raw_slides_dir.mkdir(parents=True, exist_ok=True)
        raw_board_dir.mkdir(parents=True, exist_ok=True)

        return session_dir
