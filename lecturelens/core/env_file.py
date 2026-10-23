"""Keeping a choice in the .env file.

Some choices are made once and then reused, for example the window that
is recorded. Writing them into the .env file means the question is only
asked the first time.
"""

from pathlib import Path
from typing import Optional

import dotenv

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
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        success, _, _ = dotenv.set_key(str(target), name, value, quote_mode="auto", encoding="utf-8")
        return bool(success)
    except OSError as error:
        logger.warning("the .env file could not be written: %s", error)
        return False


def remove_env_value(name: str, env_path: Optional[Path] = None) -> bool:
    """Remove a value from the .env file."""
    target = env_path if env_path is not None else find_env_file()
    if target is None or not target.is_file():
        return False
    try:
        success, _ = dotenv.unset_key(str(target), name, quote_mode="always", encoding="utf-8")
        return bool(success)
    except OSError as error:
        logger.warning("the .env file could not be written: %s", error)
        return False


def read_env_value(name: str, env_path: Optional[Path] = None) -> Optional[str]:
    """Read a value from the .env file."""
    target = env_path if env_path is not None else find_env_file()
    if target is None or not target.is_file():
        return None
    try:
        return dotenv.get_key(str(target), name, encoding="utf-8")
    except OSError:
        return None


def env_file_location() -> str:
    """Where the .env file is, for the messages."""
    target = find_env_file()
    if target is None:
        return str(Path.cwd() / ".env")
    return str(target)
