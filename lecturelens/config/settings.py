from lecturelens.config.defaults import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOCAL_MODEL,
    SYNC_CONSUMER_GROUP,
    SYNC_CONSUMER_PREFIX
)
from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings


def project_root() -> Path:
    """The folder of the project, where the .env file is."""
    return Path(__file__).resolve().parent.parent.parent


def resolve_output_dir(output_dir: Path) -> Path:
    """Where the sessions are written.

    A relative path is resolved against the folder of the project, not
    against the folder the command was started in. Otherwise the same
    command wrote its notes somewhere else every time it was started
    from a different folder, and the notes of a session could not be
    found again.
    """
    if output_dir.is_absolute():
        return output_dir
    return (project_root() / output_dir).resolve()


class Settings(BaseSettings):
    model_config = {
        "env_prefix": "LECTURELENS_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

    redis_url: str = Field(default="redis://127.0.0.1:6379")
    postgres_url: str = Field(default="postgresql://postgres:postgres@127.0.0.1:5433/lecturelens")
    ollama_url: str = Field(default="http://127.0.0.1:11434")
    local_model: str = Field(default=DEFAULT_LOCAL_MODEL)
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)

    output_dir: Path = Field(default=Path("./output"))
    router_config_path: Path = Field(default=Path("./router.yaml"))

    audio_source: str = Field(default="mic")
    audio_only: bool = Field(default=False)

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
    whisper_compute_type: Optional[str] = Field(default=None)
    whisper_preload_in_every_worker: bool = Field(default=True)
    whisper_hallucination_phrases: list[str] = Field(default_factory=list)

    screenshot_interval_seconds: float = Field(default=1.0)
    # The frames per second sampled from a video file in video mode. The
    # live capture takes one frame per interval; one per second is also
    # enough for a recording, because the change gate keeps the frames
    # that matter and drops the ones that repeat.
    video_frame_fps: float = Field(default=1.0)
    # Whether the text of the screen is read on the machine. The reader
    # runs on the processor and costs nothing, and it reads the text of a
    # slide as well as a model does over the network. The model is then
    # only asked for the parts the reader cannot do.
    screen_local_ocr: bool = Field(default=True)
    # Whether the parts of the screen are found: the text, the pictures
    # and the parts of the window around the lecture. The pictures of the
    # diagrams go into the notes, and the parts of the window are left
    # out of the text.
    screen_layout_model: bool = Field(default=True)
    slide_diff_threshold: float = Field(default=0.004)
    # The slide text is extracted by a vision model, so the number of slides
    # that can be transcribed is bounded by the provider's budget. The retries
    # are patient so a backlog of detected slides drains at the rate the
    # provider allows instead of being given up on. The slide images are
    # always saved, so nothing visual is lost either way.
    vision_retry_delays_seconds: list[float] = Field(
        default_factory=lambda: [10.0, 30.0, 60.0, 120.0, 180.0, 240.0]
    )
    slide_pixel_delta: int = Field(default=25)
    slide_min_interval_seconds: float = Field(default=5.0)
    slide_thumbnail_width: int = Field(default=640)
    slide_revisit_hamming_distance: int = Field(default=3)
    monitor_index: int = Field(default=1)
    # What is recorded. "screen" is the whole display, "window" is a single
    # window on the desktop. A window is the better choice, because the rest
    # of the desktop belongs to other work, and a video call that happens to
    # be on the same display would put the faces of the participants into
    # the notes instead of the slides.
    capture_target: str = Field(default="screen")
    # The window that is recorded. The handle is what the agent uses, the
    # query is what the user typed, and the title is kept so a window that
    # was closed can be found again.
    window_handle: Optional[int] = Field(default=None)
    window_title: Optional[str] = Field(default=None)
    window_query: Optional[str] = Field(default=None)

    low_disk_bytes: int = Field(default=2 * 1024 * 1024 * 1024)
    sync_fingerprint_history_limit: int = Field(default=5000)
    stream_max_length: int = Field(default=200000)
    reclaim_idle_ms: int = Field(default=30000)
    retry_key_ttl_seconds: int = Field(default=86400)
    redis_outbox_max_size: int = Field(default=1000)
    structuring_max_batches_per_run: int = Field(default=1000)
    qa_embed_batch_size: int = Field(default=32)
    postgres_query_limit: int = Field(default=1000)

    structuring_interval_seconds: int = Field(default=120)
    # Whether a second pass writes the mathematics of a section in
    # LaTeX. The model is asked for LaTeX, but it does not always do it,
    # and a formula that is left as plain text is not drawn by a viewer.
    structuring_latex_pass: bool = Field(default=True)
    # How much of the timeline goes into one request to the model. The
    # free providers put a limit on the length of a single message, and a
    # batch that is too long is refused with an error that leaves the
    # lecture without notes. The batch is also split in half when the
    # provider says that the message is too long.
    max_slice_characters: int = Field(default=6000)
    context_max_characters: int = Field(default=4000)
    correlation_window_seconds: float = Field(default=5.0)
    correlation_active_seconds: float = Field(default=600.0)
    correlation_lead_seconds: float = Field(default=3.0)
    sync_merge_window_seconds: float = Field(default=30.0)

    gateway_host: str = Field(default="127.0.0.1")
    gateway_port: int = Field(default=8765)
    gateway_api_key: Optional[str] = Field(default=None)
    gateway_require_api_key: bool = Field(default=True)
    gateway_allowed_origins: list[str] = Field(default_factory=list)
    gateway_max_message_bytes: int = Field(default=33554432)
    gateway_messages_per_second: float = Field(default=20.0)
    gateway_max_sessions: int = Field(default=8)
    gateway_capture_id_capacity: int = Field(default=5000)
    gateway_capture_id_ttl_seconds: float = Field(default=600.0)
    gateway_replay_scan_limit: int = Field(default=5000)

    thread_pool_max_workers: int = Field(default=4)
    process_pool_max_workers: int = Field(default=2)

    redis_consumer_batch_size: int = Field(default=10)
    redis_consumer_block_ms: int = Field(default=2000)
    redis_max_delivery_attempts: int = Field(default=5)
    redis_reclaim_interval_seconds: float = Field(default=15.0)
    redis_backlog_drain_budget_seconds: float = Field(default=60.0)
    redis_socket_connect_timeout_seconds: float = Field(default=5.0)
    redis_socket_timeout_seconds: float = Field(default=30.0)
    capture_spool_max_size: int = Field(default=500)
    capture_spool_append_timeout_seconds: float = Field(default=5.0)
    # What to do when the capture spool is full and stays full: "drop_oldest"
    # keeps the capture going and loses the oldest pending capture,
    # "raise" loses nothing but the capture itself fails.
    capture_spool_on_full: str = Field(default="drop_oldest")
    sync_consumer_group: str = Field(default=SYNC_CONSUMER_GROUP)
    sync_consumer_prefix: str = Field(default=SYNC_CONSUMER_PREFIX)

    # Cloud first, local fallback if there is one. In "hybrid" mode every
    # provider of the chain is allowed and the order of the chain decides:
    # the cloud providers are asked first, and the local model is only
    # asked when every cloud provider was refused. "local" asks the local
    # model alone, and "cloud" leaves the local model out entirely.
    llm_mode: str = Field(default="hybrid")
    llm_timeout_seconds: float = Field(default=120.0)

    primary_provider: Optional[str] = Field(default=None)
    ollama_model: Optional[str] = Field(default=None)
    unorouter_model: Optional[str] = Field(default=None)

    # The provider credentials are documented in .env.example without the
    # LECTURELENS_ prefix. pydantic-settings only reads prefixed names, so
    # the unprefixed spelling is accepted as well. Without this the keys in
    # .env were silently ignored and every cloud provider was skipped.
    unorouter_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LECTURELENS_UNOROUTER_API_KEY", "UNOROUTER_API_KEY")
    )
    unorouter_api_keys: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LECTURELENS_UNOROUTER_API_KEYS", "UNOROUTER_API_KEYS")
    )
    openrouter_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LECTURELENS_OPENROUTER_API_KEY", "OPENROUTER_API_KEY")
    )
    openrouter_api_keys: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LECTURELENS_OPENROUTER_API_KEYS", "OPENROUTER_API_KEYS")
    )
    google_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LECTURELENS_GOOGLE_API_KEY", "GOOGLE_API_KEY")
    )
    google_api_keys: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LECTURELENS_GOOGLE_API_KEYS", "GOOGLE_API_KEYS")
    )
    gemini_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LECTURELENS_GEMINI_API_KEY", "GEMINI_API_KEY")
    )
    gemini_api_keys: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("LECTURELENS_GEMINI_API_KEYS", "GEMINI_API_KEYS")
    )

    enable_drift_filter: bool = Field(default=True)
    drift_threshold: Optional[float] = Field(default=None)
    drift_similarity_threshold: float = Field(default=0.50)
    drift_consecutive_shift: int = Field(default=3)
    drift_min_content_words: int = Field(default=8)
    drift_embed_model: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def sync_drift_threshold(self) -> "Settings":
        if self.drift_threshold is not None:
            self.drift_similarity_threshold = self.drift_threshold
        if not self.drift_embed_model:
            # The drift filter used a separately configured model, so it
            # could silently use a different embedding space than the rest
            # of the pipeline.
            self.drift_embed_model = self.embedding_model
        self.output_dir = resolve_output_dir(self.output_dir)
        if not self.router_config_path.is_absolute():
            # The router file belongs to the project, so it is found
            # wherever the command was started.
            self.router_config_path = (project_root() / self.router_config_path).resolve()
        return self

    def get_session_output_dir(self, session_id: str) -> Path:
        session_dir = self.output_dir / session_id
        return session_dir

    def create_session_directories(self, session_id: str) -> Path:
        session_dir = self.get_session_output_dir(session_id)

        raw_audio_dir = session_dir / "raw" / "audio"
        raw_slides_dir = session_dir / "raw" / "slides"
        control_dir = session_dir / "control"
        heartbeat_dir = session_dir / "heartbeats"
        logs_dir = session_dir / "logs"

        raw_audio_dir.mkdir(parents=True, exist_ok=True)
        raw_slides_dir.mkdir(parents=True, exist_ok=True)
        control_dir.mkdir(parents=True, exist_ok=True)
        heartbeat_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        return session_dir
