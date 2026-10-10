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
    compute_difference_hash,
    hamming_distance
)
from lecturelens.agents.screen_agent import build_gray_thumbnail
from lecturelens.core.markdown_formatter import format_clean_markdown

print("=" * 60)
print("       LECTURELENS EMPIRICAL REPRODUCIBLE BENCHMARK")
print("=" * 60)

def resolve_audio_sample() -> tuple[bytes, float]:
    candidate_patterns = [
        "output/**/raw/audio/*.wav",
        "C:/Users/enup1/output/**/raw/audio/*.wav"
    ]
    for pattern in candidate_patterns:
        matches = glob.glob(pattern, recursive=True)
        if len(matches) > 0:
            target_path = Path(matches[0])
            with wave.open(str(target_path), "rb") as wf:
                duration = wf.getnframes() / float(wf.getframerate())
            return target_path.read_bytes(), duration

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
    return buffer.getvalue(), duration

def resolve_slide_sample() -> Image.Image:
    candidate_patterns = [
        "output/**/raw/slides/*.png",
        "C:/Users/enup1/output/**/raw/slides/*.png"
    ]
    for pattern in candidate_patterns:
        matches = glob.glob(pattern, recursive=True)
        if len(matches) > 0:
            return Image.open(matches[0])

    img = Image.new("RGB", (1920, 1080), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 1820, 980], outline=(0, 0, 0), width=4)
    draw.text((200, 200), "LectureLens Empirical Slide Benchmark", fill=(0, 0, 0))
    return img

audio_bytes, audio_duration_sec = resolve_audio_sample()
print(f"\n[1] AUDIO TRANSCRIPTION (ASR) BENCHMARK")
print(f"    Audio Duration: {audio_duration_sec:.2f} seconds")

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

print(f"\n[2] SCREEN CHANGE DETECTION (VISION) BENCHMARK")
slide_img = resolve_slide_sample()
width, height = slide_img.size
print(f"    Slide Resolution: {width}x{height}")

img_arr1 = np.array(slide_img.convert("RGB"))
img_arr2 = np.copy(img_arr1)
img_arr2[100:150, 100:150] = 0

naive_times = []
for _ in range(50):
    t0 = time.perf_counter()
    _ = np.mean(np.abs(img_arr1.astype(np.int16) - img_arr2.astype(np.int16))) > 5.0
    naive_times.append(time.perf_counter() - t0)
avg_naive_ms = (sum(naive_times) / len(naive_times)) * 1000.0

gating_times = []
for _ in range(50):
    t0 = time.perf_counter()
    thumb1 = build_gray_thumbnail(slide_img, 640)
    hash1 = compute_difference_hash(thumb1)
    thumb2 = build_gray_thumbnail(Image.fromarray(img_arr2), 640)
    hash2 = compute_difference_hash(thumb2)
    _ = hamming_distance(hash1, hash2)
    gating_times.append(time.perf_counter() - t0)
avg_gate_ms = (sum(gating_times) / len(gating_times)) * 1000.0

hash_only_times = []
thumb_pre = build_gray_thumbnail(slide_img, 640)
for _ in range(200):
    t0 = time.perf_counter()
    _ = compute_difference_hash(thumb_pre)
    hash_only_times.append(time.perf_counter() - t0)
avg_hash_only_us = (sum(hash_only_times) / len(hash_only_times)) * 1_000_000.0

print(f"    Method 1: Naive 1080p RGB Matrix Difference: {avg_naive_ms:.2f} ms")
print(f"    Method 2: LectureLens Thumbnail + dHash:      {avg_gate_ms:.2f} ms")
print(f"    dHash Hash Kernel Only:                      {avg_hash_only_us:.1f} us (microseconds)")
print(f"    Gating Speedup vs Full Frame Diff:           {avg_naive_ms / avg_gate_ms:.1f}x faster")

candidate_sessions = [
    Path(r"C:\Users\enup1\output\2026-10-11_014623"),
    Path("./output/2026-10-11_014623")
]
for session_dir in candidate_sessions:
    if (session_dir / "raw" / "slides").exists():
        n_slides = len(list((session_dir / "raw" / "slides").glob("*.png")))
        session_sec = 600
        clock_file = session_dir / "clock.json"
        if clock_file.exists():
            c_data = json.loads(clock_file.read_text(encoding="utf-8"))
            session_sec = max(60, int(c_data.get("accumulated_seconds", 600)))
        total_potential_frames = session_sec
        reduction = ((total_potential_frames - n_slides) / total_potential_frames) * 100.0
        print(f"\n[3] SESSION COMPUTE REDUCTION (Real Lecture: {session_dir.name})")
        print(f"    Session Duration:               {session_sec} seconds (~{session_sec/60:.1f} minutes)")
        print(f"    Naive 1 FPS OCR Invocations:    {total_potential_frames} calls")
        print(f"    LectureLens Gated Invocations:  {n_slides} calls")
        print(f"    Empirical Compute Reduction:    {reduction:.1f}%")
        break

print(f"\n[4] MARKDOWN AST SANITIZATION BENCHMARK")
dirty_markdown = """01:51:40
Expert academic note taker. O. Henry's "Hearts and Hands".
Sub-topic 1: The Reinforcement of the Persona.
Heading 1: The Persona
Drafting Section 1: Detail text here.
Refining Section 1: Better academic words.
Markdown headings? Yes.
No Roman numerals? Yes.
Ignore the DP slide? Yes.
## The Reinforcement of the False Persona

The second man accompanying Mr. Easton inadvertently serves as a catalyst for the misdirection.
- $A = 2$
- $B = 4$
$$
x^2 + 2x + 1 = 0
$$
"""
clean_times = []
for _ in range(200):
    t0 = time.perf_counter()
    _ = format_clean_markdown(dirty_markdown)
    clean_times.append(time.perf_counter() - t0)

avg_clean_us = (sum(clean_times) / len(clean_times)) * 1_000_000.0
print(f"    Cleaning & AST Formatting Latency: {avg_clean_us:.1f} us (0.0{int(avg_clean_us/10)} ms)")

print("=" * 60)
