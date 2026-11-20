import logging
import os
import sys

_configured: bool = False


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    level_name = os.environ.get("LECTURELENS_LOG_LEVEL", "INFO").upper()
    level_value = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level_value,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
