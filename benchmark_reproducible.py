import glob
import io
import json
import math
import os
import struct
import time
import wave
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

from lecturelens.agents.audio_agent import transcribe_audio_chunk
from lecturelens.core.image_utils import (
    SlideChangeGate,
    compute_difference_hash,
    hamming_distance
)
from lecturelens.agents.screen_agent import build_gray_thumbnail
from lecturelens.core.markdown_formatter import format_clean_markdown

print("=" * 60)
print("       LECTURELENS EMPIRICAL REPRODUCIBLE BENCHMARK")
print("=" * 60)

def resolve_audio_sample() -> tuple[bytes, float, bool]:
    candidate_patterns = [
        "output/**/raw/audio/*.wav",
    ]
    for pattern in candidate_patterns:
        matches = glob.glob(pattern, recursive=True)
        if len(matches) > 0:
            target_path = Path(matches[0])
            try:
                with wave.open(str(target_path), "rb") as wf:
                    duration = wf.getnframes() / float(wf.getframerate())
                return target_path.read_bytes(), duration, False
            except Exception:
                pass

    sample_rate = 16000
    duration = 5.0
    num_samples = int(sample_rate * duration)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(num_samples):
            sample = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate))
            wf.writeframes(struct.pack("<h", sample))
    return buffer.getvalue(), duration, True

def resolve_slide_sample() -> tuple[Image.Image, bool]:
    candidate_patterns = [
        "output/**/raw/slides/*.png",
    ]
    for pattern in candidate_patterns:
        matches = glob.glob(pattern, recursive=True)
        if len(matches) > 0:
            try:
                return Image.open(matches[0]), False
            except Exception:
                pass

    img = Image.new("RGB", (1920, 1080), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 1820, 980], outline=(0, 0, 0), width=4)
    draw.text((200, 200), "LectureLens Empirical Slide Benchmark", fill=(0, 0, 0))
    return img, True

audio_bytes, audio_duration_sec, audio_is_synthetic = resolve_audio_sample()
print(f"\n[1] AUDIO TRANSCRIPTION (ASR) BENCHMARK")
print(f"    Audio Duration: {audio_duration_sec:.2f} seconds")
if audio_is_synthetic:
    print("    WARNING: no recorded audio found in ./output/**/raw/audio. Using a synthetic 440 Hz tone.")
    print("             Whisper's VAD discards a pure tone, so these timings are NOT representative of speech.")
else:
    print("    Source: real recorded chunk from ./output")

try:
    _ = transcribe_audio_chunk(audio_bytes, "base", device="cpu")

    cpu_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = transcribe_audio_chunk(audio_bytes, "base", device="cpu")
        cpu_times.append(time.perf_counter() - t0)

    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    cpu_rtf = avg_cpu_time / audio_duration_sec
    print(f"    CPU (INT8): {avg_cpu_time*1000:.1f} ms | RTF: {cpu_rtf:.4f} (Speed: {audio_duration_sec/avg_cpu_time:.1f}x real-time)")

    try:
        import torch
        if torch.cuda.is_available():
            _ = transcribe_audio_chunk(audio_bytes, "base", device="cuda")
            cuda_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                _ = transcribe_audio_chunk(audio_bytes, "base", device="cuda")
                cuda_times.append(time.perf_counter() - t0)
            avg_cuda_time = sum(cuda_times) / len(cuda_times)
            cuda_rtf = avg_cuda_time / audio_duration_sec
            print(f"    GPU (CUDA FP16): {avg_cuda_time*1000:.1f} ms | RTF: {cuda_rtf:.4f} (Speed: {audio_duration_sec/avg_cuda_time:.1f}x real-time)")
            print(f"    GPU vs CPU Speedup: {avg_cpu_time / avg_cuda_time:.1f}x")
    except Exception:
        pass
except Exception as asr_error:
    print(f"    ASR benchmark skipped: {type(asr_error).__name__}: {asr_error}")

print(f"\n[2] SCREEN CHANGE DETECTION (VISION) BENCHMARK")
slide_img, slide_is_synthetic = resolve_slide_sample()
width, height = slide_img.size
print(f"    Slide Resolution: {width}x{height}")
if slide_is_synthetic:
    print("    WARNING: no recorded slide found in ./output/**/raw/slides. Using a synthetic white slide.")
else:
    print("    Source: real captured slide from ./output")

img_arr1 = np.array(slide_img.convert("RGB"))
img_arr2 = np.copy(img_arr1)
# The synthetic change has to be large enough to cross the slide change gate
# threshold. The small 50x50 square was below it, so the "changed"
# benchmarks actually measured the unchanged path and the numbers were
# meaningless.
_changed_fraction = 0.12
_region_height = max(1, int(img_arr1.shape[0] * (_changed_fraction ** 0.5)))
_region_width = max(1, int(img_arr1.shape[1] * (_changed_fraction ** 0.5)))
img_arr2[:_region_height, :_region_width] = 0

naive_times = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = np.mean(np.abs(img_arr1.astype(np.int16) - img_arr2.astype(np.int16))) > 5.0
    naive_times.append(time.perf_counter() - t0)
avg_naive_ms = (sum(naive_times) / len(naive_times)) * 1000.0

thumb_base = build_gray_thumbnail(slide_img, 640)
alt_slide_img = Image.fromarray(img_arr2)
thumb_changed = build_gray_thumbnail(alt_slide_img, 640)

# Verify that the synthetic change is actually detected by the gate, so the
# "changed" measurements below are real.
_gate_probe = SlideChangeGate(change_threshold=0.004, pixel_delta=25, min_interval_seconds=0.0)
_gate_probe.evaluate(thumb_base, 0.0)
_detected = False
for _probe_index in range(1, 6):
    if _gate_probe.evaluate(thumb_changed, float(_probe_index)):
        _detected = True
        break
if not _detected:
    print("    WARNING: the synthetic change is below the gate threshold; "
          "the 'changed' measurements below do not exercise the change path.")

gate_settled = SlideChangeGate(change_threshold=0.004, pixel_delta=25, min_interval_seconds=0.0)
gate_settled.evaluate(thumb_base, 0.0)
settled_times = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = gate_settled.evaluate(thumb_base, time.monotonic())
    settled_times.append(time.perf_counter() - t0)
avg_settled_ms = (sum(settled_times) / len(settled_times)) * 1000.0

gate_changing = SlideChangeGate(change_threshold=0.004, pixel_delta=25, min_interval_seconds=0.0)
changing_times = []
for i in range(50):
    current_thumb = thumb_changed if (i % 2 == 1) else thumb_base
    t0 = time.perf_counter()
    _ = gate_changing.evaluate(current_thumb, time.monotonic())
    changing_times.append(time.perf_counter() - t0)
avg_changing_ms = (sum(changing_times) / len(changing_times)) * 1000.0

thumbnail_times = []
for _ in range(20):
    t0 = time.perf_counter()
    _ = build_gray_thumbnail(slide_img, 640)
    thumbnail_times.append(time.perf_counter() - t0)
avg_thumbnail_ms = (sum(thumbnail_times) / len(thumbnail_times)) * 1000.0
total_settled_ms = avg_thumbnail_ms + avg_settled_ms
total_changing_ms = avg_thumbnail_ms + avg_changing_ms

revisit_hash_times = []
thumb_pre = build_gray_thumbnail(slide_img, 640)
for _ in range(200):
    t0 = time.perf_counter()
    _ = compute_difference_hash(thumb_pre)
    revisit_hash_times.append(time.perf_counter() - t0)
avg_revisit_hash_us = (sum(revisit_hash_times) / len(revisit_hash_times)) * 1_000_000.0

print(f"    Method 1: Naive 1080p RGB Matrix Difference:    {avg_naive_ms:.2f} ms")
print(f"    Thumbnail build (full frame -> 640px gray):     {avg_thumbnail_ms:.2f} ms (paid once per captured frame)")
print(f"    Method 2a: SlideChangeGate.evaluate Settled:    {avg_settled_ms:.2f} ms")
print(f"    Method 2b: SlideChangeGate.evaluate Changed:    {avg_changing_ms:.2f} ms")
print(f"    Method 2 total per frame (thumbnail + settled): {total_settled_ms:.2f} ms")
print(f"    Method 2 total per frame (thumbnail + changed): {total_changing_ms:.2f} ms")
print(f"    Revisit Cache dHash Kernel:                     {avg_revisit_hash_us:.1f} us")
print(f"    Gating Speedup vs Full Frame Diff (Settled):    {avg_naive_ms / total_settled_ms:.1f}x faster")
print(f"    Gating Speedup vs Full Frame Diff (Changed):    {avg_naive_ms / total_changing_ms:.1f}x faster")

found_session = False
output_dir = Path("./output")
if output_dir.exists():
    for session_dir in sorted(output_dir.iterdir(), reverse=True):
        slides_dir = session_dir / "raw" / "slides"
        if slides_dir.exists():
            n_slides = len(list(slides_dir.glob("*.png")))
            session_sec = 0
            meta_file = session_dir / "session_meta.json"
            clock_file = session_dir / "clock.json"
            timeline_file = session_dir / "timeline.jsonl"
            if meta_file.exists():
                try:
                    m_data = json.loads(meta_file.read_text(encoding="utf-8"))
                    dur = sum(float(r.get("duration_seconds", 0.0)) for r in m_data.get("runs", []))
                    if dur > 0:
                        session_sec = int(dur)
                except Exception:
                    pass
            if session_sec == 0 and clock_file.exists():
                try:
                    c_data = json.loads(clock_file.read_text(encoding="utf-8"))
                    dur = float(c_data.get("accumulated_seconds", 0.0))
                    if dur > 0:
                        session_sec = int(dur)
                except Exception:
                    pass
            if session_sec == 0 and timeline_file.exists():
                try:
                    lines = timeline_file.read_text(encoding="utf-8").strip().splitlines()
                    if len(lines) >= 2:
                        first_t = float(json.loads(lines[0]).get("timestamp_epoch", 0.0))
                        last_t = float(json.loads(lines[-1]).get("timestamp_epoch", 0.0))
                        if last_t > first_t:
                            session_sec = int(last_t - first_t)
                except Exception:
                    pass
            print(f"\n[3] SESSION COMPUTE REDUCTION (Recorded session: {session_dir.name})")
            if session_sec == 0:
                print("    Session duration could not be determined from session_meta.json, clock.json or timeline.jsonl.")
                print(f"    LectureLens gated invocations: {n_slides} calls (no baseline comparison without a duration).")
            else:
                total_potential_frames = session_sec
                reduction = ((total_potential_frames - n_slides) / total_potential_frames) * 100.0
                print(f"    Session Duration:               {session_sec} seconds (~{session_sec/60:.1f} minutes)")
                print(f"    OCR-every-second baseline:      {total_potential_frames} calls (worst-case strawman, not a real competitor)")
                print(f"    LectureLens Gated Invocations:  {n_slides} calls")
                print(f"    Reduction vs that baseline:     {reduction:.1f}%")
            found_session = True
            break

if not found_session:
    print("\n[3] SESSION COMPUTE REDUCTION")
    print("    No recorded session directory found in ./output. Run a lecture with `lecturelens start` to inspect session savings.")

print(f"\n[4] MARKDOWN POST-PROCESSING & SANITIZATION BENCHMARK")
dirty_markdown = """01:51:40
Drafting Section 1: Algorithmic Complexity analysis.
Markdown headings? Yes.
Ignore the DP slide? Yes.
#### Computational Complexity & Invariant Analysis

The partition step in quicksort guarantees that the pivot element is placed in its final position.
- $A = 2$
- $B = 4$
$$
T(n) = 2T(n/2) + O(n)
$$
"""
clean_times = []
for _ in range(200):
    t0 = time.perf_counter()
    _ = format_clean_markdown(dirty_markdown)
    clean_times.append(time.perf_counter() - t0)

avg_clean_us = (sum(clean_times) / len(clean_times)) * 1_000_000.0
print(f"    Cleaning & Markdown Formatting Latency: {avg_clean_us:.1f} us ({avg_clean_us/1000.0:.3f} ms)")

print("=" * 60)
