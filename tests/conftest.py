"""Keep the tests independent of the machine they run on.

The settings are read from the .env file of the project and from the
environment. A developer who puts a key, a language or a window in
there changes the behaviour of the tests, and a test that fails on one
machine and passes on another is worse than no test at all. It also put
the key of the developer into the output of a failing test once.

The .env file and the LECTURELENS_ variables are therefore switched off
for the whole test run. A test that wants a setting sets it itself.
"""

import os

import pytest

from lecturelens.config.settings import Settings

# The variables that switch the tests on are not removed. The failure
# injection tests need a redis to kill a worker in, and the build has
# one, so removing the variable made them skip there as well.
KEPT_ENVIRONMENT_VARIABLES = ("LECTURELENS_INTEGRATION_REDIS",)

# The .env file of the project is not read while the tests run. This has
# to happen while this file is imported, which is before the test modules
# are collected, because a test module may build the settings when it is
# imported.
Settings.model_config["env_file"] = None

def _clear_environment() -> None:
    """Remove the settings of the machine from the environment.

    The router loads the .env file into the environment, so the values
    come back after they were removed here. They are removed again
    before every test.
    """
    for name in list(os.environ):
        if not name.startswith("LECTURELENS_"):
            continue
        # The variables that switch the tests on are kept. The failure
        # injection tests only run when the machine has a redis to kill
        # a worker in, and the build has one.
        if name in KEPT_ENVIRONMENT_VARIABLES:
            continue
        os.environ.pop(name, None)


_clear_environment()


@pytest.fixture(autouse=True)
def isolated_settings() -> None:
    """Every test gets settings that do not come from the machine."""
    _clear_environment()
    yield
    _clear_environment()
