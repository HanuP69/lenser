import asyncio
from collections import deque
import io
import queue
import sys
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from lecturelens.agents.base_agent import BaseAgent
from lecturelens.config.settings import Settings
from lecturelens.core.agent_runtime import (
    AgentControl, RuntimeContext, configure_agent_pools,
    execute_agent_main, run_standard_agent
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
    "thank you", "thanks for watching", "thank you for watching", "thank you very much",
    "you", "bye", "bye bye", "subtitles by the amara.org community", "please subscribe",
)
HINDI_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "\u0927\u0928\u094d\u092f\u0935\u093e\u0926", "\u0927\u0928\u094d\u092f\u0935\u093e\u0926 \u092c\u0939\u0941\u0924",
    "\u0927\u0928\u094d\u092f\u0935\u093e\u0926 \u092c\u0939\u0941\u0924 \u092c\u0939\u0941\u0924", "\u0905\u0917\u0932\u0947 \u0935\u0940\u0921\u093f\u092f\u094b",
    "\u0905\u0917\u0932\u0947 \u092d\u093e\u0917", "\u0938\u092c\u094d\u0938\u0915\u094d\u0930\u093e\u0907\u092c",
    "\u0938\u092c\u094d\u0938\u0915\u094d\u0930\u093e\u0907\u092c \u0915\u0930\u0947\u0902", "\u091a\u0948\u0928\u0932",
    "\u0932\u093e\u0907\u0915 \u0915\u0930\u0947\u0902", "\u092c\u0947\u0932 \u0906\u0907\u0915\u0928",
)
HINGLISH_HALLUCINATION_PHRASES: tuple[str, ...] = (
    "dhanyavaad", "dhanyavad", "shukriya", "agle video", "agle part",
    "subscribe karein", "like karein", "bell icon dabayein",
)
HALLUCINATION_PHRASES_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "en": ENGLISH_HALLUCINATION_PHRASES, "hi": HINDI_HALLUCINATION_PHRASES, "hinglish": HINGLISH_HALLUCINATION_PHRASES,
}
HALLUCINATION_PHRASES: tuple[str, ...] = tuple(p for l in HALLUCINATION_PHRASES_BY_LANGUAGE.values() for p in l)
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
    return float(np.sqrt(np.mean(np.square(audio_buffer.astype(np.float64))))) if audio_buffer.size > 0 else 0.0


def convert_numpy_to_wav_bytes(audio_buffer: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(audio_buffer.astype(np.float32), -1.0, 1.0)
    int16_samples = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(int16_samples.tobytes())
    return buf.getvalue()


def normalize_phrase(text: str) -> str:
    lowered = text.lower().strip()
    return "".join(c for c in lowered if c.isalnum() or c in ("_", " ", ".") or unicodedata.category(c).startswith("M")).strip(" .")


def is_hallucinated_text(text: str, language: str = "en", extra_phrases: Sequence[str] = ()) -> bool:
    normalized = normalize_phrase(text)
    if not normalized:
        return True
    phrases = set(HALLUCINATION_PHRASES)
    phrases.update(HALLUCINATION_PHRASES_BY_LANGUAGE.get((language or "en").strip().lower(), ()))
    for p in extra_phrases:
        cl = normalize_phrase(str(p))
        if cl:
            phrases.add(cl)
    if normalized in phrases:
        return True
    words = normalized.split()
    if len(words) >= 6:
        distinct = set(words)
        if len(distinct) <= 2 or (len(words) >= 8 and len(distinct) / float(len(words)) < 0.35):
            return True
        digits = sum(1 for c in normalized if c.isdigit())
        alnums = sum(1 for c in normalized if c.isalnum())
        if alnums >= 10 and (digits / float(alnums)) > 0.60:
            return True
        for w_sz in (2, 3):
            if len(words) >= w_sz * 3:
                ngrams = [tuple(words[i:i + w_sz]) for i in range(0, len(words) - w_sz + 1, w_sz)]
                if len(ngrams) >= 3 and len(set(ngrams)) == 1:
                    return True
    return False


def save_audio_chunk(audio_bytes: bytes, output_dir: str, chunk_index: int) -> str:
    path = Path(output_dir) / f"chunk_{chunk_index:05d}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, audio_bytes)
    return str(path)


def load_whisper_model(model_size: str, device: str, compute_type: Optional[str] = None) -> Any:
    key = f"{model_size}:{device}:{compute_type}"
    if key not in _whisper_model_cache:
        from faster_whisper import WhisperModel
        resolved = compute_type or ("int8" if device == "cpu" else "float16")
        _whisper_model_cache[key] = WhisperModel(model_size, device=device, compute_type=resolved)
    return _whisper_model_cache[key]


def preload_whisper_model(model_size: str, device: str, compute_type: Optional[str] = None) -> bool:
    load_whisper_model(model_size, device, compute_type)
    return True


def transcribe_audio_chunk(
    audio_bytes: bytes, model_size: str, device: str = "cpu", language: Optional[str] = None,
    beam_size: int = 3, min_avg_logprob: float = -1.0, max_no_speech_prob: float = 0.6,
    compute_type: Optional[str] = None, repetition_penalty: float = 1.2, no_repeat_ngram_size: int = 3,
    initial_prompt: Optional[str] = None, vad_min_silence_duration_ms: int = 500, vad_speech_pad_ms: int = 200
) -> dict:
    model = load_whisper_model(model_size, device, compute_type)
    opts: dict[str, Any] = {
        "beam_size": beam_size, "language": language, "vad_filter": True,
        "condition_on_previous_text": False, "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "vad_parameters": dict(min_silence_duration_ms=vad_min_silence_duration_ms, speech_pad_ms=vad_speech_pad_ms)
    }
    if initial_prompt:
        opts["initial_prompt"] = initial_prompt
    segments, info = model.transcribe(io.BytesIO(audio_bytes), **opts)
    kept_texts, probs = [], []
    for s in segments:
        if (s.no_speech_prob > max_no_speech_prob and s.avg_logprob < min_avg_logprob) or s.avg_logprob < (min_avg_logprob - 1.0):
            continue
        text = s.text.strip()
        if text:
            kept_texts.append(text)
            probs.append(float(np.exp(min(0.0, s.avg_logprob))))
    return {
        "text": " ".join(kept_texts).strip(),
        "confidence": max(0.0, min(1.0, float(sum(probs) / len(probs)))) if probs else 0.0,
        "language": str(info.language),
        "language_probability": float(getattr(info, "language_probability", 0.0))
    }


NOISE_HISTORY_BLOCKS = 600
NOISE_MIN_BLOCKS = 100
NOISE_UPDATE_EVERY_BLOCKS = 5
NOISE_FLOOR_PERCENTILE = 10.0
NOISE_MARGIN_FACTOR = 1.3
NOISE_MAX_THRESHOLD_FACTOR = 4.0


class SpeechSegmenter:
    def __init__(
        self, sample_rate: int, energy_threshold: float, min_chunk_seconds: float,
        max_chunk_seconds: float, silence_gap_seconds: float, pre_roll_blocks: int = 2
    ) -> None:
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.noise_floor = 0.0
        self.adaptive = True
        self.rms_history: deque = deque(maxlen=NOISE_HISTORY_BLOCKS)
        self.blocks_since_floor_update = 0
        self.min_chunk_seconds = min_chunk_seconds
        self.max_chunk_seconds = max_chunk_seconds
        self.silence_gap_seconds = silence_gap_seconds
        self.pre_roll_blocks = pre_roll_blocks
        self.origin_seconds: Optional[float] = None
        self.consumed_samples = 0
        self.pre_roll: list[tuple[int, np.ndarray]] = []
        self.active_blocks: list[np.ndarray] = []
        self.active_start_sample = 0
        self.active_sample_count = 0
        self.has_speech = False
        self.silence_run_samples = 0
        self.dropped_samples = 0

    def note_dropped_samples(self, dropped_samples: int) -> None:
        if dropped_samples > 0:
            self.dropped_samples += int(dropped_samples)

    def update_noise_floor(self, rms: float) -> None:
        if not self.adaptive:
            return
        self.rms_history.append(rms)
        self.blocks_since_floor_update += 1
        if len(self.rms_history) >= NOISE_MIN_BLOCKS and self.blocks_since_floor_update >= NOISE_UPDATE_EVERY_BLOCKS:
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
        chunk = AudioChunk(samples=samples, start_seconds=start_seconds, end_seconds=start_seconds + (samples.size / float(self.sample_rate)))
        self.active_blocks, self.active_sample_count, self.has_speech, self.silence_run_samples = [], 0, False, 0
        return chunk

    def push(self, block: np.ndarray, stream_elapsed_seconds: float) -> list[AudioChunk]:
        if block.size == 0:
            return []
        if self.origin_seconds is None:
            self.origin_seconds = stream_elapsed_seconds - (block.size / float(self.sample_rate))
        elif self.dropped_samples > 0:
            self.origin_seconds += self.dropped_samples / float(self.sample_rate)
            self.dropped_samples = 0

        block_start = self.consumed_samples
        self.consumed_samples += block.size
        rms = compute_rms_energy(block)
        self.update_noise_floor(rms)
        speech_present = rms >= self.effective_threshold()

        if not self.active_blocks:
            if not speech_present:
                self.pre_roll.append((block_start, block))
                while len(self.pre_roll) > self.pre_roll_blocks:
                    self.pre_roll.pop(0)
                return []
            if self.pre_roll:
                self.active_start_sample = self.pre_roll[0][0]
                for _, prb in self.pre_roll:
                    self.active_blocks.append(prb)
                    self.active_sample_count += prb.size
            else:
                self.active_start_sample = block_start
            self.pre_roll = []

        self.active_blocks.append(block)
        self.active_sample_count += block.size
        if speech_present:
            self.has_speech, self.silence_run_samples = True, 0
        else:
            self.silence_run_samples += block.size

        active_sec = self.active_sample_count / float(self.sample_rate)
        silence_sec = self.silence_run_samples / float(self.sample_rate)
        if active_sec >= self.max_chunk_seconds or (active_sec >= self.min_chunk_seconds and silence_sec >= self.silence_gap_seconds):
            if self.has_speech:
                return [self.build_chunk()]
            self.active_blocks, self.active_sample_count, self.silence_run_samples = [], 0, 0
        return []

    def flush(self) -> list[AudioChunk]:
        emitted = [self.build_chunk()] if self.active_blocks and self.has_speech else []
        self.active_blocks, self.active_sample_count, self.has_speech, self.silence_run_samples = [], 0, False, 0
        return emitted


class BaseBlockSource:
    def __init__(self, sample_rate: int, block_seconds: float, max_queue_blocks: int = 600) -> None:
        self.sample_rate = sample_rate
        self.block_frames = max(1, int(sample_rate * block_seconds))
        self.block_queue: queue.Queue = queue.Queue(maxsize=max_queue_blocks)
        self.dropped_blocks = 0

    def read_block(self, timeout_seconds: float) -> Optional[np.ndarray]:
        try:
            return self.block_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            return None

    @property
    def dropped_samples(self) -> int:
        return self.dropped_blocks * self.block_frames

    def restart(self) -> None:
        self.stop()
        self.start()

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class MicBlockSource(BaseBlockSource):
    def __init__(self, sample_rate: int, block_seconds: float, max_queue_blocks: int = 600) -> None:
        super().__init__(sample_rate, block_seconds, max_queue_blocks)
        self.stream: Any = None

    def callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        try:
            self.block_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.dropped_blocks += 1

    def start(self) -> None:
        import sounddevice as sd
        self.stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=self.block_frames, callback=self.callback
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.warning("mic stream close failed: %s", e)
            self.stream = None


def initialize_com_in_current_thread() -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        return ctypes.WinDLL("ole32").CoInitializeEx(None, 0x2) >= 0
    except Exception:
        return False


def uninitialize_com_in_current_thread(initialized_here: bool) -> None:
    if initialized_here and sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.WinDLL("ole32").CoUninitialize()
        except Exception:
            pass


def find_loopback_microphone() -> tuple[Optional[Any], Optional[str]]:
    try:
        import soundcard
    except Exception as err:
        return None, f"soundcard library unavailable: {err}"
    if sys.platform == "darwin":
        loopbacks = [
            m for m in soundcard.all_microphones(include_loopback=True)
            if getattr(m, "isloopback", False) or "blackhole" in str(getattr(m, "name", "")).lower() or "soundflower" in str(getattr(m, "name", "")).lower()
        ]
        return (loopbacks[0], None) if loopbacks else (None, "System audio capture on macOS requires a virtual loopback device (such as BlackHole or Soundflower)")
    try:
        return soundcard.get_microphone(id=str(soundcard.default_speaker().id), include_loopback=True), None
    except Exception:
        loopbacks = [m for m in soundcard.all_microphones(include_loopback=True) if getattr(m, "isloopback", False)]
        return (loopbacks[0], None) if loopbacks else (None, "No loopback audio device found for system audio capture")


class SystemBlockSource(BaseBlockSource):
    def __init__(self, sample_rate: int, block_seconds: float, max_queue_blocks: int = 600, strict: bool = False) -> None:
        super().__init__(sample_rate, block_seconds, max_queue_blocks)
        self.strict = strict
        self.thread: Optional[threading.Thread] = None
        self.is_running = False
        self.mic: Any = None
        self.device_error: Optional[str] = None

    def _record_loop(self) -> None:
        com_init = False
        try:
            if self.mic is None:
                interval = float(self.block_frames) / float(self.sample_rate)
                while self.is_running:
                    time.sleep(interval)
                    try:
                        self.block_queue.put_nowait(np.zeros(self.block_frames, dtype=np.float32))
                    except queue.Full:
                        self.dropped_blocks += 1
                return
            com_init = initialize_com_in_current_thread()
            with self.mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                while self.is_running:
                    data = recorder.record(numframes=self.block_frames)
                    if data is not None and len(data) > 0:
                        try:
                            self.block_queue.put_nowait(data[:, 0].astype(np.float32))
                        except queue.Full:
                            self.dropped_blocks += 1
        except Exception as error:
            logger.error("System audio capture error: %s", error, exc_info=True)
        finally:
            uninitialize_com_in_current_thread(com_init)

    def start(self) -> None:
        self.mic, self.device_error = find_loopback_microphone()
        if self.mic is None:
            if self.strict:
                raise RuntimeError(self.device_error or "No loopback audio device found for system audio capture")
            logger.warning("No loopback audio device found for system audio capture; capturing silence")
        self.is_running = True
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None


class MixedBlockSource:
    def __init__(self, sample_rate: int, block_seconds: float) -> None:
        self.mic_source = MicBlockSource(sample_rate, block_seconds)
        self.system_source = SystemBlockSource(sample_rate, block_seconds, strict=False)
        self.dropped_blocks = 0

    def start(self) -> None:
        self.mic_source.start()
        self.system_source.start()

    def read_block(self, timeout_seconds: float) -> Optional[np.ndarray]:
        for src in (self.system_source, self.mic_source):
            while src.block_queue.qsize() > 5:
                try:
                    src.block_queue.get_nowait()
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
        return np.clip(mic_block[:min_len] + sys_block[:min_len], -1.0, 1.0).astype(np.float32)

    def stop(self) -> None:
        self.mic_source.stop()
        self.system_source.stop()

    @property
    def dropped_samples(self) -> int:
        return self.dropped_blocks * self.mic_source.block_frames + self.mic_source.dropped_samples + self.system_source.dropped_samples

    def restart(self) -> None:
        self.mic_source.restart()
        self.system_source.restart()


def build_block_source(settings: Settings) -> Any:
    sr, bs = settings.mic_sample_rate, settings.audio_block_seconds
    src = settings.audio_source or "mic"
    if src == "system":
        return SystemBlockSource(sr, bs, strict=True)
    if src == "both":
        return MixedBlockSource(sr, bs)
    return MicBlockSource(sr, bs)


class AudioAgent(BaseAgent):
    def __init__(
        self, settings: Settings, redis_bus: RedisBus, session_clock: SessionClock,
        session_id: str, block_source: Any = None
    ) -> None:
        super().__init__(settings, redis_bus, session_clock, session_id, "audio_agent")
        self.block_source: Any = block_source
        self.raw_audio_dir: Path = self.session_dir / "raw" / "audio"
        self.stream_name: str = f"lecturelens:{session_id}:stream:transcript"
        self.chunk_counter: int = next_file_index(self.raw_audio_dir, "chunk_", ".wav") - 1
        self.locked_language: Optional[str] = settings.whisper_language
        self.transcription_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.audio_queue_capacity)
        self.spillover_queue: list[PendingTranscription] = []
        self.dropped_transcriptions: int = 0

    async def preload_model(self) -> None:
        try:
            await run_in_process_pool(
                preload_whisper_model, self.settings.whisper_model_size,
                self.settings.whisper_device, self.settings.whisper_compute_type
            )
        except Exception as error:
            logger.error("whisper preload failed: %s", error)

    async def run(self) -> None:
        self.is_running = True
        source = self.block_source or build_block_source(self.settings)
        self.block_source = source
        await run_in_thread_pool(source.start)
        preload_task = asyncio.create_task(self.preload_model())
        transcription_task = asyncio.create_task(self.transcription_loop())
        self.control.write_heartbeat("live")
        try:
            await self.capture_loop(source)
        finally:
            self.is_running = False
            if not transcription_task.done():
                deadline = time.monotonic() + 15.0
                delivered = False
                while time.monotonic() < deadline and not transcription_task.done():
                    try:
                        self.transcription_queue.put_nowait(None)
                        delivered = True
                        break
                    except asyncio.QueueFull:
                        await asyncio.sleep(0.2)
                if not delivered and not transcription_task.done():
                    transcription_task.cancel()
            try:
                await transcription_task
            except (asyncio.CancelledError, Exception):
                pass
            preload_task.cancel()
            await asyncio.gather(preload_task, return_exceptions=True)
            try:
                await run_in_thread_pool(source.stop)
            except Exception as error:
                logger.error("could not stop audio capture device: %s", error)

    async def capture_loop(self, source: Any) -> None:
        segmenter = SpeechSegmenter(
            sample_rate=self.settings.mic_sample_rate, energy_threshold=self.settings.vad_energy_threshold,
            min_chunk_seconds=self.settings.mic_chunk_duration_seconds,
            max_chunk_seconds=self.settings.audio_max_chunk_seconds,
            silence_gap_seconds=self.settings.audio_silence_gap_seconds
        )
        stalled_seconds, was_paused, read_timeout, last_dropped = 0.0, False, 0.5, 0
        while self.is_running and not self.control.stop_requested():
            block = await run_in_thread_pool(source.read_block, read_timeout)
            dropped = int(getattr(source, "dropped_samples", 0) or 0)
            if dropped > last_dropped:
                new_dropped = dropped - last_dropped
                last_dropped = dropped
                segmenter.note_dropped_samples(new_dropped)
                self.control.set_stat("dropped_audio_samples", dropped)
                logger.warning("audio capture dropped %d samples", new_dropped)

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
                    logger.warning("audio source stalled, restarting")
                    try:
                        await run_in_thread_pool(source.restart)
                    except Exception as error:
                        logger.error("audio restart failed: %s", error)
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
        wav_path = await run_in_thread_pool(save_audio_chunk, wav_bytes, str(self.raw_audio_dir), self.chunk_counter)
        pending = PendingTranscription(wav_bytes=wav_bytes, wav_path=wav_path, start_seconds=chunk.start_seconds, end_seconds=chunk.end_seconds)
        try:
            self.transcription_queue.put_nowait(pending)
        except asyncio.QueueFull:
            if len(self.spillover_queue) < 100:
                self.spillover_queue.append(pending)
            else:
                self.dropped_transcriptions += 1
                self.control.set_stat("dropped_transcriptions", self.dropped_transcriptions)

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
                            logger.error("transcription failed: %s", error)
                while self.spillover_queue:
                    try:
                        await self.transcribe_and_publish(self.spillover_queue.pop(0))
                    except Exception as error:
                        logger.error("spillover transcription failed: %s", error)
                return
            try:
                await self.transcribe_and_publish(pending)
            except Exception as error:
                logger.error("transcription failed: %s", error)
                self.control.increment_stat("transcription_errors")

    async def transcribe_and_publish(self, pending: PendingTranscription) -> Optional[TranscriptChunk]:
        result = await run_in_process_pool(
            transcribe_audio_chunk, pending.wav_bytes, self.settings.whisper_model_size,
            self.settings.whisper_device, self.locked_language, self.settings.whisper_beam_size,
            self.settings.whisper_min_avg_logprob, self.settings.whisper_max_no_speech_prob,
            self.settings.whisper_compute_type, self.settings.whisper_repetition_penalty,
            self.settings.whisper_no_repeat_ngram_size, self.settings.whisper_initial_prompt,
            self.settings.whisper_vad_min_silence_duration_ms, self.settings.whisper_vad_speech_pad_ms
        )
        text = str(result.get("text", "")).strip()
        lang = str(result.get("language", "en"))
        if is_hallucinated_text(text, language=lang, extra_phrases=self.settings.whisper_hallucination_phrases):
            return None

        lang_prob = float(result.get("language_probability", 0.0))
        if self.locked_language is None:
            threshold = 0.0 if self.settings.whisper_lock_language else 0.8
            if lang_prob >= threshold:
                self.locked_language = lang

        chunk_event = TranscriptChunk(
            session_id=self.session_id,
            timestamp_epoch=self.session_clock.epoch_for_elapsed(pending.start_seconds),
            start_seconds=pending.start_seconds,
            end_seconds=pending.end_seconds,
            text=text,
            confidence=float(result.get("confidence", 0.0)),
            language=lang,
            audio_source=self.settings.audio_source,
            speaker_id=None,
            importance_flags=detect_importance_flags(text),
            raw_audio_path=pending.wav_path
        )
        pub_id = await self.redis_bus.publish_resilient(self.stream_name, chunk_event)
        self.control.increment_stat("transcripts_published" if pub_id is not None else "transcripts_buffered")
        return chunk_event

    async def ingest_wav_bytes(self, wav_bytes: bytes, capture_epoch: Optional[float] = None) -> Optional[TranscriptChunk]:
        self.chunk_counter += 1
        wav_path = await run_in_thread_pool(save_audio_chunk, wav_bytes, str(self.raw_audio_dir), self.chunk_counter)
        with wave.open(io.BytesIO(wav_bytes), "rb") as r:
            duration = r.getnframes() / float(r.getframerate())
        end_sec = self.session_clock.elapsed_for_epoch(capture_epoch) if capture_epoch is not None else self.session_clock.get_elapsed_seconds()
        return await self.transcribe_and_publish(
            PendingTranscription(wav_bytes=wav_bytes, wav_path=wav_path, start_seconds=max(0.0, end_sec - duration), end_seconds=end_sec)
        )

    def stop(self) -> None:
        self.is_running = False


async def main() -> None:
    def build_agent(context: RuntimeContext) -> AudioAgent:
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
