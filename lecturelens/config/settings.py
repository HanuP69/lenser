from lecturelens.config.defaults import DEFAULT_EMBEDDING_MODEL, DEFAULT_LOCAL_MODEL
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from lecturelens.core.validation import sanitize_file_stem


class Settings(BaseSettings):
    model_config = {
        "env_prefix": "LECTURELENS_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    redis_url: str = Field(default="redis://localhost:6379")
    ollama_url: str = Field(default="http://localhost:11434")
    local_model: str = Field(default=DEFAULT_LOCAL_MODEL)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)

    tesseract_path: str = Field(default="C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    docker_ocr_url: Optional[str] = Field(default=None)

    output_dir: Path = Field(default=Path("./output"))
    router_config_path: Path = Field(default=Path("./router.yaml"))

    audio_source: str = Field(default="mic")
    audio_only: bool = Field(default=False)
    active_agents: list[str] = Field(
        default=["audio", "screen", "camera", "sync", "structuring"]
    )

    session_name: Optional[str] = Field(default=None)
    course_name: Optional[str] = Field(default=None)

    mic_sample_rate: int = Field(default=16000)
    mic_chunk_duration_seconds: float = Field(default=3.0)
    audio_max_chunk_seconds: float = Field(default=12.0)
    audio_silence_gap_seconds: float = Field(default=0.5)
    audio_block_seconds: float = Field(default=0.1)
    audio_queue_capacity: int = Field(default=20)
    audio_stall_restart_seconds: float = Field(default=5.0)
    vad_energy_threshold: float = Field(default=0.01)

    whisper_model_size: str = Field(default="base")
    whisper_device: str = Field(default="cpu")
    whisper_language: Optional[str] = Field(default=None)
    whisper_lock_language: bool = Field(default=False)
    whisper_beam_size: int = Field(default=3)
    whisper_min_avg_logprob: float = Field(default=-1.0)
    whisper_max_no_speech_prob: float = Field(default=0.6)

    screenshot_interval_seconds: float = Field(default=1.0)
    slide_diff_threshold: float = Field(default=0.004)
    vision_retry_delays_seconds: list[float] = Field(default_factory=lambda: [15.0, 45.0, 120.0])
    slide_pixel_delta: int = Field(default=25)
    slide_min_interval_seconds: float = Field(default=5.0)
    slide_thumbnail_width: int = Field(default=640)
    slide_revisit_hamming_distance: int = Field(default=3)
    monitor_index: int = Field(default=1)

    camera_device_index: int = Field(default=0)
    camera_poll_interval_seconds: float = Field(default=0.5)
    camera_warmup_frames: int = Field(default=5)
    motion_threshold: float = Field(default=0.02)
    blur_variance_threshold: float = Field(default=100.0)
    motion_stillness_required_frames: int = Field(default=5)
    board_change_threshold: float = Field(default=0.01)
    board_min_interval_seconds: float = Field(default=10.0)
    board_min_area_ratio: float = Field(default=0.2)
    board_occlusion_fraction: float = Field(default=0.35)
    board_occlusion_patience: int = Field(default=3)

    structuring_interval_seconds: int = Field(default=120)
    max_slice_characters: int = Field(default=12000)
    context_max_characters: int = Field(default=4000)
    correlation_window_seconds: float = Field(default=5.0)
    correlation_active_seconds: float = Field(default=600.0)
    correlation_lead_seconds: float = Field(default=3.0)
    sync_merge_window_seconds: float = Field(default=30.0)

    gateway_host: str = Field(default="127.0.0.1")
    gateway_port: int = Field(default=8765)
    gateway_api_key: Optional[str] = Field(default=None)
    gateway_max_message_bytes: int = Field(default=33554432)
    gateway_messages_per_second: float = Field(default=20.0)
    gateway_max_sessions: int = Field(default=8)

    thread_pool_max_workers: int = Field(default=4)
    process_pool_max_workers: int = Field(default=2)

    redis_consumer_batch_size: int = Field(default=10)
    redis_consumer_block_ms: int = Field(default=2000)

    llm_mode: str = Field(default="local")
    llm_timeout_seconds: float = Field(default=120.0)

    primary_provider: Optional[str] = Field(default=None)
    ollama_model: Optional[str] = Field(default=None)
    unorouter_model: Optional[str] = Field(default=None)
    unorouter_api_key: Optional[str] = Field(default=None)
    openrouter_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)

    def get_session_output_dir(self, session_id: str) -> Path:
        session_dir = self.output_dir / session_id
        return session_dir

    def get_course_database_path(self, course_name: Optional[str]) -> Path:
        stem = "_default"
        if course_name is not None and len(course_name.strip()) > 0:
            stem = sanitize_file_stem(course_name)
        return self.output_dir / "_courses" / f"{stem}.db"

    def create_session_directories(self, session_id: str) -> Path:
        session_dir = self.get_session_output_dir(session_id)

        raw_audio_dir = session_dir / "raw" / "audio"
        raw_slides_dir = session_dir / "raw" / "slides"
        raw_board_dir = session_dir / "raw" / "board"
        control_dir = session_dir / "control"
        heartbeat_dir = session_dir / "heartbeats"
        logs_dir = session_dir / "logs"

        raw_audio_dir.mkdir(parents=True, exist_ok=True)
        raw_slides_dir.mkdir(parents=True, exist_ok=True)
        raw_board_dir.mkdir(parents=True, exist_ok=True)
        control_dir.mkdir(parents=True, exist_ok=True)
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        return session_dir
