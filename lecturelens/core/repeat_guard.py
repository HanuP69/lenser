"""Saying a fault once instead of three hundred times.

A fault that happens every two seconds for ten minutes is one line of
news and three hundred lines of noise. The guard says the first time,
then keeps a count of how often it happened, so the news is not lost and
the noise is not made.
"""

import logging
from typing import Any, Optional


class RepeatGuard:
    """A fault that is said once and then counted.

    The use is where a fault is expected but should not be silent. The
    first time it happens it is said out loud. After that it is counted,
    and said again every so often so that a fault that never stops is
    not forgotten.
    """

    def __init__(
        self,
        name: str,
        logger: Optional[logging.Logger] = None,
        first_at: int = 1,
        every_at: int = 50
    ) -> None:
        self.name = name
        self.logger = logger
        self.first_at = max(1, first_at)
        self.every_at = max(1, every_at)
        self.count = 0

    def saw(self, error: BaseException) -> bool:
        """Note that the fault happened.

        Returns True when it should be said out loud.
        """
        self.count += 1
        if self.count == self.first_at or self.count % self.every_at == 0:
            if self.logger is not None:
                self.logger.warning(
                    "%s (%d times now): %s", self.name, self.count, error
                )
            return True
        return False

    def said(self) -> int:
        """How many times the fault happened."""
        return self.count

    def reset(self) -> None:
        """The fault is over. The next one is said out loud again."""
        self.count = 0


class Guards:
    """A place to keep the guards of a piece of work.

    The guards are made once, when the thing is made, and then used
    where the fault is caught. That way the count of a fault is not
    thrown away with the variable it belonged to.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger
        self._guards: dict[str, RepeatGuard] = {}

    def guard(self, name: str) -> RepeatGuard:
        """The guard of a fault, made the first time it is asked for."""
        if name not in self._guards:
            self._guards[name] = RepeatGuard(name, self.logger)
        return self._guards[name]

    def saw(self, name: str, error: BaseException) -> bool:
        """Note a fault and say it when it should be said."""
        return self.guard(name).saw(error)

    def said(self, name: str) -> int:
        """How many times a fault happened."""
        return self.guard(name).said()

    def reset(self, name: str) -> None:
        """A fault is over."""
        self.guard(name).reset()

    def counts(self) -> dict[str, int]:
        """How often every fault happened."""
        return {name: guard.count for name, guard in self._guards.items()}

    def report(self) -> dict[str, Any]:
        """The counts, for the heartbeat of the agent."""
        return self.counts()
