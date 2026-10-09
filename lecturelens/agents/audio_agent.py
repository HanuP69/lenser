import asyncio
import io
import os
import sys
import wave
from pathlib import Path
from typing import Any, Optional
import numpy as np
from lecturelens.config.settings import Settings
from lecturelens.core.clock import SessionClock
from lecturelens.core.concurrency import run_in_thread_pool, run_in_process_pool
from lecturelens.core.importance import detect_importance_flags
from lecturelens.core.redis_bus import RedisBus
from lecturelens.schemas.events import TranscriptChunk

_cached_whisper_model: Any = None


def get_or_create_whisper_model(model_size: str, device: str) -> Any:
    global _cached_whisper_model
    if _cached_whisper_model is None:
        from faster_whisper import WhisperModel
        _cached_whisper_model = WhisperModel(model_size, device=device, compute_type="int8")
    return _cached_whisper_model


def transcribe_audio_chunk_worker(
    audio_bytes: bytes,
    model_size: str,
    device: str
) -> dict[str, Any]:
    model = get_or_create_whisper_model(model_size, device)
    byte_stream = io.BytesIO(audio_bytes)
    segments, info = model.transcribe(byte_stream, beam_size=5)

    transcribed_text_parts: list[str] = []
    total_confidence = 0.0
    segment_count = 0

    for segment in segments:
        transcribed_text_parts.append(segment.text.strip())
        segment_count += 1
        probability = getattr(segment, "avg_logprob", -1.0)
        confidence = float(np.exp(probability)) if probability < 0.0 else 0.8
        total_confidence += min(max(confidence, 0.0), 1.0)

    combined_text = " ".join(transcribed_text_parts)
    average_confidence = 0.8
    if segment_count > 0:
        average_confidence = total_confidence / segment_count

    return {
        "text": combined_text,
        "confidence": average_confidence,
        "language": getattr(info, "language", "en")
    }


def capture_mic_chunk(duration_seconds: float, sample_rate: int) -> np.ndarray:
    import sounddevice as sd
    number_of_frames = int(duration_seconds * sample_rate)
    recording = sd.rec(
        frames=number_of_frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return recording.flatten()


def capture_system_audio_chunk(duration_seconds: float, sample_rate: int) -> np.ndarray:
    import soundcard as sc
    speaker_name = sc.default_speaker().name
    loopback_mic = sc.get_microphone(id=speaker_name, include_loopback=True)
    number_of_frames = int(duration_seconds * sample_rate)
    with loopback_mic.recorder(samplerate=sample_rate, channels=1) as recorder:
        recorded_data = recorder.record(numframes=number_of_frames)
    return recorded_data.flatten()


def compute_rms_energy(audio_buffer: np.ndarray) -> float:
    if len(audio_buffer) == 0:
        return 0.0
    mean_square = np.mean(np.square(audio_buffer))
    root_mean_square = float(np.sqrt(mean_square))
    return root_mean_square


def is_speech_detected(audio_buffer: np.ndarray, energy_threshold: float) -> bool:
    energy = compute_rms_energy(audio_buffer)
    if energy >= energy_threshold:
        return True
    return False


def convert_numpy_to_wav_bytes(audio_buffer: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(audio_buffer, -1.0, 1.0)
    int16_samples = (clipped * 32767).astype(np.int16)
    byte_io = io.BytesIO()
    with wave.open(byte_io, "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(int16_samples.tobytes())
    return byte_io.getvalue()


def save_wav_file(wav_bytes: bytes, file_path: Path) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as file_handle:
        file_handle.write(wav_bytes)


class AudioAgent:
    def __init__(
        self,
        settings: Settings,
        redis_bus: RedisBus,
        session_clock: SessionClock,
        session_id: str
    ) -> None:
        self.settings: Settings = settings
        self.redis_bus: RedisBus = redis_bus
        self.session_clock: SessionClock = session_clock
        self.session_id: str = session_id
        self.stream_name: str = f"lecturelens:{session_id}:stream:transcript"
        self.chunk_counter: int = 0
        self.is_running: bool = False

    async def capture_single_chunk(self) -> np.ndarray:
        if self.settings.audio_source == "system":
            audio_buffer = await run_in_thread_pool(
                capture_system_audio_chunk,
                self.settings.mic_chunk_duration_seconds,
                self.settings.mic_sample_rate
            )
            return audio_buffer
        if self.settings.audio_source == "both":
            mic_task = run_in_thread_pool(
                capture_mic_chunk,
                self.settings.mic_chunk_duration_seconds,
                self.settings.mic_sample_rate
            )
            system_task = run_in_thread_pool(
                capture_system_audio_chunk,
                self.settings.mic_chunk_duration_seconds,
                self.settings.mic_sample_rate
            )
            mic_buffer, system_buffer = await asyncio.gather(mic_task, system_task)
            min_length = min(len(mic_buffer), len(system_buffer))
            combined = (mic_buffer[:min_length] + system_buffer[:min_length]) * 0.5
            return combined

        audio_buffer = await run_in_thread_pool(
            capture_mic_chunk,
            self.settings.mic_chunk_duration_seconds,
            self.settings.mic_sample_rate
        )
        return audio_buffer

    async def run(self) -> None:
        self.is_running = True
        session_dir = self.settings.create_session_directories(self.session_id)
        raw_audio_dir = session_dir / "raw" / "audio"

        while self.is_running:
            chunk_start_seconds = self.session_clock.get_elapsed_seconds()
            chunk_timestamp_epoch = self.session_clock.get_timestamp()

            try:
                audio_buffer = await self.capture_single_chunk()
            except Exception:
                await asyncio.sleep(0.5)
                continue

            chunk_end_seconds = self.session_clock.get_elapsed_seconds()

            speech_present = is_speech_detected(
                audio_buffer,
                self.settings.vad_energy_threshold
            )
            if not speech_present:
                await asyncio.sleep(0.01)
                continue

            self.chunk_counter += 1
            wav_bytes = convert_numpy_to_wav_bytes(
                audio_buffer,
                self.settings.mic_sample_rate
            )

            chunk_file_name = f"chunk_{self.chunk_counter:05d}.wav"
            chunk_file_path = raw_audio_dir / chunk_file_name
            save_wav_file(wav_bytes, chunk_file_path)

            try:
                transcription_result = await run_in_process_pool(
                    transcribe_audio_chunk_worker,
                    wav_bytes,
                    self.settings.whisper_model_size,
                    self.settings.whisper_device
                )
            except Exception:
                continue

            text = transcription_result.get("text", "").strip()
            if len(text) == 0:
                continue

            confidence = transcription_result.get("confidence", 0.8)
            language = transcription_result.get("language", "en")
            importance_flags = detect_importance_flags(text)

            chunk_event = TranscriptChunk(
                session_id=self.session_id,
                timestamp_epoch=chunk_timestamp_epoch,
                start_seconds=chunk_start_seconds,
                end_seconds=chunk_end_seconds,
                text=text,
                confidence=confidence,
                language=language,
                audio_source=self.settings.audio_source,
                importance_flags=importance_flags,
                raw_audio_path=str(chunk_file_path)
            )

            await self.redis_bus.publish(self.stream_name, chunk_event)

    def stop(self) -> None:
        self.is_running = False


async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=None)
    parser.add_argument("--audio-source", default=None)
    args, _ = parser.parse_known_args()

    settings = Settings()
    if args.session is not None and len(args.session) > 0:
        settings.session_name = args.session
    if args.audio_source is not None and len(args.audio_source) > 0:
        settings.audio_source = args.audio_source

    session_id = settings.session_name or "default_session"
    clock_file = settings.get_session_output_dir(session_id) / "clock.json"
    if clock_file.exists():
        session_clock = SessionClock.load_from_file(clock_file)
    else:
        session_clock = SessionClock()

    redis_bus = RedisBus(settings.redis_url)
    agent = AudioAgent(
        settings=settings,
        redis_bus=redis_bus,
        session_clock=session_clock,
        session_id=session_id
    )
    try:
        await agent.run()
    finally:
        await redis_bus.close()


if __name__ == "__main__":
    asyncio.run(main())
