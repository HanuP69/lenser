import asyncio
from collections import deque
import io
import queue
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import (
    AgentControl,
    RuntimeContext,
    configure_agent_pools,
    execute_agent_main,
    run_standard_agent
)
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_process_pool, run_in_thread_pool
from lecturelens.core.fileio import atomic_write_bytes, next_file_index
from lecturelens.core.importance import detect_importance_flags
from lecturelens.core.logging_setup import get_logger
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import TranscriptChunk

logger = get_logger("lecturelens.audio_agent")

ENGLISH_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "thank you very much",
    "you",
    "bye",
    "bye bye",
    "subtitles by the amara.org community",
    "please subscribe",
)

# Whisper hallucinates the sign off of the language it is transcribing, so an
# English only list does nothing for a Hindi or Hinglish lecture.
HINDI_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "\u0927\u0928\u094d\u092f\u0935\u093e\u0926",
    "\u0927\u0928\u094d\u092f\u0935\u093e\u0926 \u092c\u0939\u0941\u0924",
    "\u0927\u0928\u094d\u092f\u0935\u093e\u0926 \u092c\u0939\u0941\u0924 \u092c\u0939\u0941\u0924",
    "\u0905\u0917\u0932\u0947 \u0935\u0940\u0921\u093f\u092f\u094b",
    "\u0905\u0917\u0932\u0947 \u092d\u093e\u0917",
    "\u0938\u092c\u094d\u0938\u0915\u094d\u0930\u093e\u0907\u092c",
    "\u0938\u092c\u094d\u0938\u0915\u094d\u0930\u093e\u0907\u092c \u0915\u0930\u0947\u0902",
    "\u091a\u0948\u0928\u0932",
    "\u0932\u093e\u0907\u0915 \u0915\u0930\u0947\u0902",
    "\u092c\u0947\u0932 \u0906\u0907\u0915\u0928",
)

HINGLISH_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "dhanyavaad",
    "dhanyavad",
    "shukriya",
    "agle video",
    "agle part",
    "subscribe karein",
    "like karein",
    "bell icon dabayein",
)

HALLUCINATION_PHRASES_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "en": ENGLISH_HALLUCINATION_PHRASES,
    "hi": HINDI_HALLUCINATION_PHRASES,
    "hinglish": HINGLISH_HALLUCINATION_PHRASES,
}

# Whisper never reports "hinglish" as a language, and a Hinglish lecture is
# detected as "hi" or "en", so a list that is only selected by the detected
# language would never be used. All of the lists are applied instead; the
# phrases are specific enough for that.
HALLUCINATION_PHRASES: tuple[str, ...] = tuple(
    phrase
    for language_phrases in HALLUCINATION_PHRASES_BY_LANGUAGE.values()
    for phrase in language_phrases
)

_whisper_model_cache: dict[str, Any] = {}


@dataclass
class AudioChunk:
    samples: np.ndarray
    start_seconds: float
    end_seconds: float


@dataclass
class PendingTranscription:
    wav_bytes: bytes
    wav_path: str
    start_seconds: float
    end_seconds: float


def compute_rms_energy(audio_buffer: np.ndarray) -> float:
    if audio_buffer.size == 0:
        return 0.0
    squared = np.square(audio_buffer.astype(np.float64))
    return float(np.sqrt(np.mean(squared)))


def is_speech_detected(audio_buffer: np.ndarray, energy_threshold: float) -> bool:
    return compute_rms_energy(audio_buffer) >= energy_threshold


def convert_numpy_to_wav_bytes(audio_buffer: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(audio_buffer.astype(np.float32), -1.0, 1.0)
    int16_samples = (clipped * 32767.0).astype(np.int16)
    output_buffer = io.BytesIO()
    with wave.open(output_buffer, "wb") as wav_writer:
        wav_writer.setnchannels(1)
        wav_writer.setsampwidth(2)
        wav_writer.setframerate(sample_rate)
        wav_writer.writeframes(int16_samples.tobytes())
    return output_buffer.getvalue()


def normalize_phrase(text: str) -> str:
    lowered = text.lower().strip()
    # Combining marks are not word characters, so a plain [^\w\s.] filter
    # strips them. That changes Devanagari and every other Indic script, so
    # the marks are kept and a Hindi phrase still matches its own form.
    kept = "".join(
        character
        for character in lowered
        if character.isalnum()
        or character == "_"
        or character.isspace()
        or character == "."
        or unicodedata.category(character).startswith("M")
    )
    return kept.strip(" .")


def is_hallucinated_text(
    text: str,
    language: str = "en",
    extra_phrases: Sequence[str] = ()
) -> bool:
    """Whether the transcript is a Whisper hallucination.

    The phrases are matched for the language of the audio, so a Hindi or
    Hinglish lecture is filtered as well. Extra phrases can be configured.
    """
    normalized = normalize_phrase(text)
    if len(normalized) == 0:
        return True
    phrases = set(HALLUCINATION_PHRASES)
    language_key = (language or "en").strip().lower()
    phrases.update(HALLUCINATION_PHRASES_BY_LANGUAGE.get(language_key, ()))
    for phrase in extra_phrases:
        cleaned = normalize_phrase(str(phrase))
        if len(cleaned) > 0:
            phrases.add(cleaned)
    if normalized in phrases:
        return True
    words = normalized.split()
    if len(words) >= 8:
        distinct_words = set(words)
        if len(distinct_words) <= 2:
            return True
    return False


def save_audio_chunk(audio_bytes: bytes, output_dir: str, chunk_index: int) -> str:
    audio_dir = Path(output_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    file_path = audio_dir / f"chunk_{chunk_index:05d}.wav"
    atomic_write_bytes(file_path, audio_bytes)
    return str(file_path)


def load_whisper_model(model_size: str, device: str, compute_type: Optional[str] = None) -> Any:
    cache_key = f"{model_size}:{device}:{compute_type}"
    if cache_key not in _whisper_model_cache:
        from faster_whisper import WhisperModel
        resolved_compute = compute_type or ("int8" if device == "cpu" else "float16")
        _whisper_model_cache[cache_key] = WhisperModel(
            model_size, device=device, compute_type=resolved_compute
        )
    return _whisper_model_cache[cache_key]


def preload_whisper_model(model_size: str, device: str, compute_type: Optional[str] = None) -> bool:
    load_whisper_model(model_size, device, compute_type)
    return True


def transcribe_audio_chunk(
    audio_bytes: bytes,
    model_size: str,
    device: str = "cpu",
    language: Optional[str] = None,
    beam_size: int = 3,
    min_avg_logprob: float = -1.0,
    max_no_speech_prob: float = 0.6,
    compute_type: Optional[str] = None
) -> dict:
    model = load_whisper_model(model_size, device, compute_type)
    segments, info = model.transcribe(
        io.BytesIO(audio_bytes),
        beam_size=beam_size,
        language=language,
        vad_filter=True,
        condition_on_previous_text=False
    )

    kept_texts: list[str] = []
    probability_values: list[float] = []
    for segment in segments:
        if segment.no_speech_prob > max_no_speech_prob and segment.avg_logprob < min_avg_logprob:
            continue
        if segment.avg_logprob < (min_avg_logprob - 1.0):
            continue
        segment_text = segment.text.strip()
        if len(segment_text) == 0:
            continue
        kept_texts.append(segment_text)
        probability_values.append(float(np.exp(min(0.0, segment.avg_logprob))))

    confidence = 0.0
    if len(probability_values) > 0:
        confidence = float(sum(probability_values) / len(probability_values))

    result = {}
    result["text"] = " ".join(kept_texts).strip()
    result["confidence"] = max(0.0, min(1.0, confidence))
    result["language"] = str(info.language)
    result["language_probability"] = float(getattr(info, "language_probability", 0.0))
    return result


NOISE_HISTORY_BLOCKS: int = 600
NOISE_MIN_BLOCKS: int = 100
NOISE_UPDATE_EVERY_BLOCKS: int = 5
NOISE_FLOOR_PERCENTILE: float = 10.0
NOISE_MARGIN_FACTOR: float = 1.3
NOISE_MAX_THRESHOLD_FACTOR: float = 4.0


class SpeechSegmenter:
    def __init__(
        self,
        sample_rate: int,
        energy_threshold: float,
        min_chunk_seconds: float,
        max_chunk_seconds: float,
        silence_gap_seconds: float,
        pre_roll_blocks: int = 2
    ) -> None:
        self.sample_rate: int = sample_rate
        self.energy_threshold: float = energy_threshold
        self.noise_floor: float = 0.0
        self.adaptive: bool = True
        self.rms_history: deque = deque(maxlen=NOISE_HISTORY_BLOCKS)
        self.blocks_since_floor_update: int = 0
        self.min_chunk_seconds: float = min_chunk_seconds
        self.max_chunk_seconds: float = max_chunk_seconds
        self.silence_gap_seconds: float = silence_gap_seconds
        self.pre_roll_blocks: int = pre_roll_blocks
        self.origin_seconds: Optional[float] = None
        self.consumed_samples: int = 0
        self.pre_roll: list[tuple[int, np.ndarray]] = []
        self.active_blocks: list[np.ndarray] = []
        self.active_start_sample: int = 0
        self.active_sample_count: int = 0
        self.has_speech: bool = False
        self.silence_run_samples: int = 0
        self.dropped_samples: int = 0

    def note_dropped_samples(self, dropped_samples: int) -> None:
        """Record audio that the capture source dropped.

        The sample clock is derived from the number of delivered blocks, so
        dropped blocks shift every later timestamp. The dropped duration is
        accounted for when the next block is pushed.
        """
        if dropped_samples > 0:
            self.dropped_samples += int(dropped_samples)

    def update_noise_floor(self, rms: float) -> None:
        if not self.adaptive:
            return
        self.rms_history.append(rms)
        self.blocks_since_floor_update += 1
        if len(self.rms_history) < NOISE_MIN_BLOCKS or self.blocks_since_floor_update < NOISE_UPDATE_EVERY_BLOCKS:
            return
        self.blocks_since_floor_update = 0
        self.noise_floor = float(np.percentile(np.asarray(self.rms_history), NOISE_FLOOR_PERCENTILE))

    def effective_threshold(self) -> float:
        if not self.adaptive or self.noise_floor < self.energy_threshold:
            return self.energy_threshold
        return min(self.noise_floor * NOISE_MARGIN_FACTOR, self.energy_threshold * NOISE_MAX_THRESHOLD_FACTOR)

    def mark_discontinuity(self) -> list[AudioChunk]:
        emitted = self.flush()
        self.origin_seconds = None
        self.consumed_samples = 0
        self.pre_roll = []
        return emitted

    def build_chunk(self) -> AudioChunk:
        samples = np.concatenate(self.active_blocks)
        origin = self.origin_seconds if self.origin_seconds is not None else 0.0
        start_seconds = origin + (self.active_start_sample / float(self.sample_rate))
        end_seconds = start_seconds + (samples.size / float(self.sample_rate))
        chunk = AudioChunk(samples=samples, start_seconds=start_seconds, end_seconds=end_seconds)
        self.active_blocks = []
        self.active_sample_count = 0
        self.has_speech = False
        self.silence_run_samples = 0
        return chunk

    def push(self, block: np.ndarray, stream_elapsed_seconds: float) -> list[AudioChunk]:
        emitted: list[AudioChunk] = []
        if block.size == 0:
            return emitted
        if self.origin_seconds is None:
            self.origin_seconds = stream_elapsed_seconds - (block.size / float(self.sample_rate))
        elif self.dropped_samples > 0:
            # Blocks that were dropped by the capture source break the
            # assumption that the sample clock is gap free, so the origin has
            # to be moved forward by the dropped duration. Otherwise every
            # later timestamp lags the real time by the total dropped
            # duration and the error grows for the rest of the session.
            dropped_seconds = self.dropped_samples / float(self.sample_rate)
            self.origin_seconds += dropped_seconds
            self.dropped_samples = 0

        block_start_sample = self.consumed_samples
        self.consumed_samples += block.size
        rms = compute_rms_energy(block)
        self.update_noise_floor(rms)
        speech_present = rms >= self.effective_threshold()

        if len(self.active_blocks) == 0:
            if not speech_present:
                self.pre_roll.append((block_start_sample, block))
                while len(self.pre_roll) > self.pre_roll_blocks:
                    self.pre_roll.pop(0)
                return emitted
            if len(self.pre_roll) > 0:
                self.active_start_sample = self.pre_roll[0][0]
                for _, pre_roll_block in self.pre_roll:
                    self.active_blocks.append(pre_roll_block)
                    self.active_sample_count += pre_roll_block.size
            else:
                self.active_start_sample = block_start_sample
            self.pre_roll = []

        self.active_blocks.append(block)
        self.active_sample_count += block.size
        if speech_present:
            self.has_speech = True
            self.silence_run_samples = 0
        else:
            self.silence_run_samples += block.size

        active_seconds = self.active_sample_count / float(self.sample_rate)
        silence_seconds = self.silence_run_samples / float(self.sample_rate)
        reached_max = active_seconds >= self.max_chunk_seconds
        natural_boundary = (
            active_seconds >= self.min_chunk_seconds
            and silence_seconds >= self.silence_gap_seconds
        )
        if reached_max or natural_boundary:
            if self.has_speech:
                emitted.append(self.build_chunk())
            else:
                self.active_blocks = []
                self.active_sample_count = 0
                self.silence_run_samples = 0
        return emitted

    def flush(self) -> list[AudioChunk]:
        emitted: list[AudioChunk] = []
        if len(self.active_blocks) > 0 and self.has_speech:
            emitted.append(self.build_chunk())
        self.active_blocks = []
        self.active_sample_count = 0
        self.has_speech = False
        self.silence_run_samples = 0
        return emitted


class MicBlockSource:
    def __init__(self, sample_rate: int, block_seconds: float, max_queue_blocks: int = 600) -> None:
        self.sample_rate: int = sample_rate
        self.block_frames: int = max(1, int(sample_rate * block_seconds))
        self.block_queue: queue.Queue = queue.Queue(maxsize=max_queue_blocks)
        self.stream: Any = None
        self.dropped_blocks: int = 0

    def callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        try:
            self.block_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped_blocks += 1

    @property
    def dropped_samples(self) -> int:
        return self.dropped_blocks * self.block_frames

    def start(self) -> None:
        import sounddevice as sounddevice_module
        self.stream = sounddevice_module.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_frames,
            callback=self.callback
        )
        self.stream.start()

    def read_block(self, timeout_seconds: float) -> Optional[np.ndarray]:
        try:
            return self.block_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as error:
                logger.warning("mic stream close failed: %s", error)
            self.stream = None

    def restart(self) -> None:
        self.stop()
        self.start()


def find_loopback_microphone() -> tuple[Optional[Any], Optional[str]]:
    import sys
    try:
        import soundcard
    except Exception as err:
        return None, f"soundcard library unavailable: {err}"
    if sys.platform == "darwin":
        loopbacks = [
            m for m in soundcard.all_microphones(include_loopback=True)
            if getattr(m, "isloopback", False)
            or "blackhole" in str(getattr(m, "name", "")).lower()
            or "soundflower" in str(getattr(m, "name", "")).lower()
        ]
        if loopbacks:
            return loopbacks[0], None
        return None, "System audio capture on macOS requires a virtual loopback device (such as BlackHole or Soundflower)"
    try:
        mic = soundcard.get_microphone(id=str(soundcard.default_speaker().id), include_loopback=True)
        return mic, None
    except Exception:
        loopbacks = [m for m in soundcard.all_microphones(include_loopback=True) if getattr(m, "isloopback", False)]
        if loopbacks:
            return loopbacks[0], None
    return None, "No loopback audio device found for system audio capture"


class SystemBlockSource:
    def __init__(
        self,
        sample_rate: int,
        block_seconds: float,
        max_queue_blocks: int = 600,
        strict: bool = False
    ) -> None:
        self.sample_rate: int = sample_rate
        self.block_frames: int = max(1, int(sample_rate * block_seconds))
        self.block_queue: queue.Queue = queue.Queue(maxsize=max_queue_blocks)
        self.thread: Optional[threading.Thread] = None
        self.is_running: bool = False
        self.dropped_blocks: int = 0
        self.strict: bool = strict
        self.mic: Any = None
        self.device_error: Optional[str] = None

    def _record_loop(self) -> None:
        try:
            if self.mic is None:
                frame_interval = float(self.block_frames) / float(self.sample_rate)
                while self.is_running:
                    time.sleep(frame_interval)
                    silent_block = np.zeros(self.block_frames, dtype=np.float32)
                    try:
                        self.block_queue.put_nowait(silent_block)
                    except queue.Full:
                        self.dropped_blocks += 1
                return

            with self.mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                while self.is_running:
                    data = recorder.record(numframes=self.block_frames)
                    if data is not None and len(data) > 0:
                        block = data[:, 0].astype(np.float32)
                        try:
                            self.block_queue.put_nowait(block)
                        except queue.Full:
                            self.dropped_blocks += 1
        except Exception as error:
            logger.error("System audio capture error: %s", error)

    def start(self) -> None:
        self.mic, self.device_error = find_loopback_microphone()
        if self.mic is None:
            if self.strict:
                raise RuntimeError(self.device_error or "No loopback audio device found for system audio capture")
            logger.warning("No loopback audio device found for system audio capture; capturing silence")
        self.is_running = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def read_block(self, timeout_seconds: float) -> Optional[np.ndarray]:
        try:
            return self.block_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            return None

    @property
    def dropped_samples(self) -> int:
        return self.dropped_blocks * self.block_frames

    def stop(self) -> None:
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def restart(self) -> None:
        self.stop()
        self.start()


class MixedBlockSource:
    def __init__(self, sample_rate: int, block_seconds: float) -> None:
        self.mic_source = MicBlockSource(sample_rate, block_seconds)
        self.system_source = SystemBlockSource(sample_rate, block_seconds, strict=False)
        self.dropped_blocks: int = 0

    def start(self) -> None:
        self.mic_source.start()
        self.system_source.start()

    def read_block(self, timeout_seconds: float) -> Optional[np.ndarray]:
        while self.system_source.block_queue.qsize() > 5:
            try:
                self.system_source.block_queue.get_nowait()
                self.dropped_blocks += 1
            except queue.Empty:
                break
        while self.mic_source.block_queue.qsize() > 5:
            try:
                self.mic_source.block_queue.get_nowait()
                self.dropped_blocks += 1
            except queue.Empty:
                break

        mic_block = self.mic_source.read_block(timeout_seconds)
        sys_block = self.system_source.read_block(0.02)

        if mic_block is None and sys_block is None:
            return None
        if mic_block is None:
            return sys_block
        if sys_block is None:
            return mic_block

        min_len = min(len(mic_block), len(sys_block))
        mixed = np.clip(mic_block[:min_len] + sys_block[:min_len], -1.0, 1.0).astype(np.float32)
        return mixed

    def stop(self) -> None:
        self.mic_source.stop()
        self.system_source.stop()

    @property
    def dropped_samples(self) -> int:
        return (
            self.dropped_blocks * self.mic_source.block_frames
            + self.mic_source.dropped_samples
            + self.system_source.dropped_samples
        )

    def restart(self) -> None:
        self.mic_source.restart()
        self.system_source.restart()


def build_block_source(settings: Settings) -> Any:
    sample_rate = settings.mic_sample_rate
    block_seconds = settings.audio_block_seconds
    audio_source = settings.audio_source or "mic"
    if audio_source == "system":
        return SystemBlockSource(sample_rate, block_seconds, strict=True)
    elif audio_source == "both":
        return MixedBlockSource(sample_rate, block_seconds)
    return MicBlockSource(sample_rate, block_seconds)


class AudioAgent:
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        session_clock: SessionClock,
        session_id: str,
        block_source: Any = None
    ) -> None:
        self.settings: Settings = settings
        self.redis_bus: RedisBus = redis_bus
        self.session_clock: SessionClock = session_clock
        self.session_id: str = session_id
        self.block_source: Any = block_source
        self.is_running: bool = False

        session_dir = settings.create_session_directories(session_id)
        self.raw_audio_dir: Path = session_dir / "raw" / "audio"
        self.stream_name: str = f"lecturelens:{session_id}:stream:transcript"
        self.control: AgentControl = AgentControl(session_dir, "audio_agent")
        self.chunk_counter: int = next_file_index(self.raw_audio_dir, "chunk_", ".wav") - 1
        self.locked_language: Optional[str] = settings.whisper_language
        self.transcription_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.audio_queue_capacity)
        self.spillover_queue: list[PendingTranscription] = []
        self.dropped_transcriptions: int = 0

    async def preload_model(self) -> None:
        try:
            await run_in_process_pool(
                preload_whisper_model,
                self.settings.whisper_model_size,
                self.settings.whisper_device,
                self.settings.whisper_compute_type
            )
        except Exception as error:
            logger.error("whisper preload failed: %s", error)

    async def run(self) -> None:
        self.is_running = True
        source = self.block_source
        if source is None:
            source = build_block_source(self.settings)
        self.block_source = source
        await run_in_thread_pool(source.start)
        preload_task = asyncio.create_task(self.preload_model())
        transcription_task = asyncio.create_task(self.transcription_loop())
        self.control.write_heartbeat("live")
        try:
            await self.capture_loop(source)
        finally:
            self.is_running = False
            try:
                # The queue can be full and the transcription task can be
                # gone, in which case a plain put() waits for space that
                # never arrives and the agent hangs on shutdown.
                await asyncio.wait_for(self.transcription_queue.put(None), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("transcription queue was full at shutdown; forcing the stop signal")
                try:
                    self.transcription_queue.put_nowait(None)
                except Exception:
                    pass
            try:
                await transcription_task
            except asyncio.CancelledError:
                pass
            except Exception as error:
                logger.error("transcription task failed during shutdown: %s", error)
            preload_task.cancel()
            await asyncio.gather(preload_task, return_exceptions=True)
            try:
                # The capture device has to be closed even when the
                # transcription task failed, otherwise the microphone stream
                # and the recording thread leak.
                await run_in_thread_pool(source.stop)
            except Exception as error:
                logger.error("could not stop the audio capture device: %s", error)

    async def capture_loop(self, source: Any) -> None:
        segmenter = SpeechSegmenter(
            sample_rate=self.settings.mic_sample_rate,
            energy_threshold=self.settings.vad_energy_threshold,
            min_chunk_seconds=self.settings.mic_chunk_duration_seconds,
            max_chunk_seconds=self.settings.audio_max_chunk_seconds,
            silence_gap_seconds=self.settings.audio_silence_gap_seconds
        )
        stalled_seconds = 0.0
        was_paused = False
        read_timeout = 0.5
        last_dropped_samples = 0
        while self.is_running and not self.control.stop_requested():
            block = await run_in_thread_pool(source.read_block, read_timeout)

            dropped_samples = int(getattr(source, "dropped_samples", 0) or 0)
            if dropped_samples > last_dropped_samples:
                newly_dropped = dropped_samples - last_dropped_samples
                last_dropped_samples = dropped_samples
                segmenter.note_dropped_samples(newly_dropped)
                # The counter used to be incremented but never read, so the
                # operator never learned that audio was being lost.
                self.control.set_stat("dropped_audio_samples", dropped_samples)
                logger.warning(
                    "audio capture dropped %d samples (%.1fs); the transcript "
                    "timestamps are shifted to compensate",
                    newly_dropped, newly_dropped / float(self.settings.mic_sample_rate)
                )

            if self.control.is_paused():
                if not was_paused:
                    for chunk in segmenter.mark_discontinuity():
                        await self.handle_chunk(chunk)
                    was_paused = True
                continue
            if was_paused:
                was_paused = False
                segmenter.mark_discontinuity()

            if block is None:
                stalled_seconds += read_timeout
                if stalled_seconds >= self.settings.audio_stall_restart_seconds:
                    logger.warning("audio source stalled, restarting capture device")
                    try:
                        await run_in_thread_pool(source.restart)
                    except Exception as error:
                        logger.error("audio source restart failed: %s", error)
                    for chunk in segmenter.mark_discontinuity():
                        await self.handle_chunk(chunk)
                    stalled_seconds = 0.0
                continue

            stalled_seconds = 0.0
            elapsed_now = self.session_clock.get_elapsed_seconds()
            for chunk in segmenter.push(block, elapsed_now):
                await self.handle_chunk(chunk)

        for chunk in segmenter.flush():
            await self.handle_chunk(chunk)

    async def handle_chunk(self, chunk: AudioChunk) -> None:
        wav_bytes = convert_numpy_to_wav_bytes(chunk.samples, self.settings.mic_sample_rate)
        self.chunk_counter += 1
        chunk_index = self.chunk_counter
        wav_path = await run_in_thread_pool(
            save_audio_chunk, wav_bytes, str(self.raw_audio_dir), chunk_index
        )
        pending = PendingTranscription(
            wav_bytes=wav_bytes,
            wav_path=wav_path,
            start_seconds=chunk.start_seconds,
            end_seconds=chunk.end_seconds
        )
        try:
            self.transcription_queue.put_nowait(pending)
        except asyncio.QueueFull:
            if len(self.spillover_queue) < 100:
                self.spillover_queue.append(pending)
                logger.warning("transcription backlog full, buffering to spillover queue: %s", wav_path)
            else:
                self.dropped_transcriptions += 1
                self.control.set_stat("dropped_transcriptions", self.dropped_transcriptions)
                logger.error("transcription backlog and spillover full (%d), dropped: %s", len(self.spillover_queue), wav_path)

    async def transcription_loop(self) -> None:
        while True:
            if self.spillover_queue and not self.transcription_queue.full():
                try:
                    self.transcription_queue.put_nowait(self.spillover_queue.pop(0))
                except asyncio.QueueFull:
                    pass
            pending = await self.transcription_queue.get()
            if pending is None:
                while not self.transcription_queue.empty():
                    rem = self.transcription_queue.get_nowait()
                    if rem is not None:
                        try:
                            await self.transcribe_and_publish(rem)
                        except Exception as error:
                            logger.error("transcription failed for remaining %s: %s", rem.wav_path, error)
                while self.spillover_queue:
                    spill_pending = self.spillover_queue.pop(0)
                    try:
                        await self.transcribe_and_publish(spill_pending)
                    except Exception as error:
                        logger.error("transcription failed for spillover %s: %s", spill_pending.wav_path, error)
                return
            try:
                await self.transcribe_and_publish(pending)
            except Exception as error:
                logger.error("transcription failed for %s: %s", pending.wav_path, error)
                self.control.increment_stat("transcription_errors")

    async def transcribe_and_publish(self, pending: PendingTranscription) -> Optional[TranscriptChunk]:
        result = await run_in_process_pool(
            transcribe_audio_chunk,
            pending.wav_bytes,
            self.settings.whisper_model_size,
            self.settings.whisper_device,
            self.locked_language,
            self.settings.whisper_beam_size,
            self.settings.whisper_min_avg_logprob,
            self.settings.whisper_max_no_speech_prob,
            self.settings.whisper_compute_type
        )
        transcript_text = str(result.get("text", "")).strip()
        detected_language = str(result.get("language", "en"))
        if is_hallucinated_text(
            transcript_text,
            language=detected_language,
            extra_phrases=self.settings.whisper_hallucination_phrases
        ):
            return None

        language_probability = float(result.get("language_probability", 0.0))
        if self.locked_language is None:
            # The language was detected again for every chunk, so a Hinglish
            # lecture could flip between languages mid sentence. A confident
            # detection is remembered and reused. whisper_lock_language
            # additionally sticks a low confidence detection.
            threshold = 0.0 if self.settings.whisper_lock_language else 0.8
            if language_probability >= threshold:
                self.locked_language = detected_language
                logger.info("language locked to %s", detected_language)

        importance_flags = detect_importance_flags(transcript_text)
        chunk_event = TranscriptChunk(
            session_id=self.session_id,
            timestamp_epoch=self.session_clock.epoch_for_elapsed(pending.start_seconds),
            start_seconds=pending.start_seconds,
            end_seconds=pending.end_seconds,
            text=transcript_text,
            confidence=float(result.get("confidence", 0.0)),
            language=detected_language,
            audio_source=self.settings.audio_source,
            speaker_id=None,
            importance_flags=importance_flags,
            raw_audio_path=pending.wav_path
        )
        published_id = await self.redis_bus.publish_resilient(self.stream_name, chunk_event)
        if published_id is not None:
            self.control.increment_stat("transcripts_published")
        else:
            self.control.increment_stat("transcripts_buffered")
        return chunk_event

    async def ingest_wav_bytes(
        self,
        wav_bytes: bytes,
        capture_epoch: Optional[float] = None
    ) -> Optional[TranscriptChunk]:
        self.chunk_counter += 1
        wav_path = await run_in_thread_pool(
            save_audio_chunk, wav_bytes, str(self.raw_audio_dir), self.chunk_counter
        )
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_reader:
            duration_seconds = wav_reader.getnframes() / float(wav_reader.getframerate())
        if capture_epoch is not None:
            end_seconds = self.session_clock.elapsed_for_epoch(capture_epoch)
        else:
            end_seconds = self.session_clock.get_elapsed_seconds()
        start_seconds = max(0.0, end_seconds - duration_seconds)
        pending = PendingTranscription(
            wav_bytes=wav_bytes,
            wav_path=wav_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds
        )
        return await self.transcribe_and_publish(pending)

    def stop(self) -> None:
        self.is_running = False


async def main() -> None:
    def build_agent(context: RuntimeContext) -> AudioAgent:
        # Every process pool worker preloads the Whisper model, so the model
        # is not loaded twice or reloaded on the first transcription.
        configure_agent_pools(context.settings)
        return AudioAgent(
            settings=context.settings,
            redis_bus=context.redis_bus,
            session_clock=context.session_clock,
            session_id=context.session_id
        )

    await run_standard_agent(build_agent)


if __name__ == "__main__":
    execute_agent_main(main)
