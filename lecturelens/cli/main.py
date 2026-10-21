"""The command line of LectureLens.

The commands live in the modules beside this one. This file is the
door that the console script goes through.
"""

from lecturelens.cli import commands
from lecturelens.cli.app import app, console, logger

__all__ = ["app", "commands", "console", "logger"]


if __name__ == "__main__":
    app()
