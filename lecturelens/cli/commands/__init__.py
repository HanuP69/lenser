"""The commands of the command line.

Importing the package brings every command in, so that they are
registered on the application.
"""

from lecturelens.cli.commands import knowledge, remote, session, windows

__all__ = ["knowledge", "remote", "session", "windows"]
