"""Keeping a choice in the .env file.

Some choices are made once and then reused, for example the window that
is recorded. Writing them into the .env file means the question is only
asked the first time.
"""

import os
from pathlib import Path
from typing import Optional

from lecturelens.core.logging_setup import get_logger

logger = get_logger("lecturelens.env_file")


def find_env_file(start: Optional[Path] = None) -> Optional[Path]:
    """The .env file of the project."""
    candidates = []
    if start is not None:
        candidates.append(Path(start) / ".env")
    candidates.append(Path.cwd() / ".env")
    project_root = Path(__file__).resolve().parent.parent.parent
    candidates.append(project_root / ".env")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def set_env_value(name: str, value: str, env_path: Optional[Path] = None) -> bool:
    """Set a value in the .env file, keeping the rest of it as it is.

    Returns whether the value was stored.
    """
    target = env_path if env_path is not None else find_env_file()
    if target is None:
        target = Path.cwd() / ".env"

    lines: list[str] = []
    if target.is_file():
        try:
            lines = target.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            logger.warning("the .env file could not be read: %s", error)
            return False

    entry = f"{name}={value}"
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.split("=", 1)[0].strip() == name:
            lines[index] = entry
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(entry)

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as error:
        logger.warning("the .env file could not be written: %s", error)
        return False
    return True


def remove_env_value(name: str, env_path: Optional[Path] = None) -> bool:
    """Remove a value from the .env file."""
    target = env_path if env_path is not None else find_env_file()
    if target is None or not target.is_file():
        return False
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        logger.warning("the .env file could not be read: %s", error)
        return False

    kept: list[str] = []
    removed = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped.split("=", 1)[0].strip() == name:
            removed = True
            continue
        kept.append(line)
    if not removed:
        return False
    try:
        target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    except OSError as error:
        logger.warning("the .env file could not be written: %s", error)
        return False
    return True


def read_env_value(name: str, env_path: Optional[Path] = None) -> Optional[str]:
    """Read a value from the .env file."""
    target = env_path if env_path is not None else find_env_file()
    if target is None or not target.is_file():
        return None
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.split("=", 1)[0].strip() == name:
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"').strip("'")
    return None


def env_file_location() -> str:
    """Where the .env file is, for the messages."""
    target = find_env_file()
    if target is None:
        return str(Path.cwd() / ".env")
    return str(target)


def current_working_directory() -> str:
    return os.getcwd()
