import os
import tempfile
import time
from pathlib import Path


def atomic_write_bytes(file_path: Path, data: bytes) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(file_path.parent),
        prefix=file_path.name + ".",
        suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as file_handle:
            file_handle.write(data)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        replace_succeeded = False
        last_error = None
        for attempt in range(5):
            try:
                os.replace(temporary_name, str(file_path))
                replace_succeeded = True
                break
            except PermissionError as error:
                last_error = error
                time.sleep(0.02)
        if not replace_succeeded and last_error is not None:
            raise last_error
    except BaseException:
        if os.path.exists(temporary_name):
            try:
                os.remove(temporary_name)
            except OSError:
                pass
        raise


def atomic_write_text(file_path: Path, text: str) -> None:
    atomic_write_bytes(file_path, text.encode("utf-8"))


def append_text_durably(file_path: Path, text: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8", newline="\n") as file_handle:
        file_handle.write(text)
        file_handle.flush()
        os.fsync(file_handle.fileno())


def next_file_index(directory: Path, prefix: str, suffix: str) -> int:
    if not directory.exists():
        return 1
    highest_index = 0
    for entry in directory.iterdir():
        entry_name = entry.name
        if not entry_name.startswith(prefix):
            continue
        if not entry_name.endswith(suffix):
            continue
        digits = entry_name[len(prefix):len(entry_name) - len(suffix)]
        if digits.isdigit():
            parsed_index = int(digits)
            if parsed_index > highest_index:
                highest_index = parsed_index
    return highest_index + 1
