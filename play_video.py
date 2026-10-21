"""Play a local video file the way a browser would: audio to the default
output device (so the loopback hears it) and the frames fullscreen on the
primary monitor (so the screen capture sees them).

    python play_video.py <file> [--seconds N] [--start N]
"""
import argparse
import sys
import time

import av
import numpy as np
import sounddevice as sd

import cv2

WINDOW = "lecturelens-playback"
PRIMARY = (0, 0, 1920, 1080)


def decode_audio(path: str, start: float, seconds: float):
    """Decode the audio track into one array, resampled for the device.

    The frame timestamps are left alone. Clearing them stops the resampler
    from working out how many samples to produce, and it pads the output
    instead, which plays the audio in slow motion.
    """
    container = av.open(path)
    stream = next((s for s in container.streams if s.type == "audio"), None)
    if stream is None:
        container.close()
        return None, 0
    device_rate = int(sd.query_devices(sd.default.device[1], "output")["default_samplerate"])
    # "fltp" is planar float32, so the array comes out as (channels, samples).
    # "flt" is packed, and transposing that flattened the interleaved stereo
    # into mono, which played the left and right channels alternately at
    # double speed.
    resampler = av.AudioResampler(format="fltp", layout="stereo", rate=device_rate)
    chunks = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            array = resampled.to_ndarray()
            if array.ndim == 2:
                # planar: (channels, samples) -> interleaved (samples, channels)
                array = array.T
            chunks.append(np.ascontiguousarray(array, dtype=np.float32))
    container.close()
    if not chunks:
        return None, device_rate
    audio = np.concatenate(chunks)
    # The decoder can hand out samples outside the valid range.
    np.clip(audio, -1.0, 1.0, out=audio)
    start_index = int(start * device_rate)
    return audio[start_index:start_index + int(seconds * device_rate)], device_rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--start", type=float, default=0.0)
    args = parser.parse_args()

    audio, device_rate = decode_audio(args.path, args.start, args.seconds)
    if audio is None:
        print("no audio track", file=sys.stderr)
        return 1
    print(f"decoded {len(audio) / device_rate:.1f}s of audio at {device_rate}Hz", flush=True)

    container = av.open(args.path)
    video_stream = next((s for s in container.streams if s.type == "video"), None)
    if video_stream is None:
        print("no video track", file=sys.stderr)
        return 1
    container.seek(int(args.start / float(video_stream.time_base)), stream=video_stream)

    cv2.namedWindow(WINDOW, cv2.WINDOW_FULLSCREEN)
    try:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass
    try:
        cv2.moveWindow(WINDOW, PRIMARY[0], PRIMARY[1])
        cv2.resizeWindow(WINDOW, PRIMARY[2], PRIMARY[3])
    except Exception:
        pass
    # Keep the window on top of everything else, otherwise the screen capture
    # records whatever is in front of it.
    try:
        import ctypes

        hwnd = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.user32.SetWindowPos(
            ctypes.windll.user32.GetParent(hwnd) or hwnd, 0, 0, 0, 0, 0x0001 | 0x0002
        )
    except Exception:
        pass

    started = time.monotonic()
    print("playing", flush=True)
    sd.play(audio, samplerate=device_rate, blocking=False)

    frame_index = 0
    for frame in container.decode(video_stream):
        elapsed = time.monotonic() - started
        if elapsed > args.seconds:
            break
        image = frame.to_ndarray(format="bgr24")
        cv2.imshow(WINDOW, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        frame_index += 1
        # Keep the video in step with the audio.
        target = frame_index / float(video_stream.average_rate or 30)
        delay = target - elapsed
        if delay > 0:
            time.sleep(delay)

    cv2.destroyAllWindows()
    sd.stop()
    container.close()
    print(f"played {frame_index} frames, {time.monotonic() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
