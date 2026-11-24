"""Launches Isaac Sim (headless) once for the test session: isaaclab imports need a running app.

Without Isaac Lab, set UNIENV_ISAACSIM_NO_SIM=1 to run only the tests that do not need it (test_adaptor.py is not
collected); otherwise a missing Isaac Lab is an error, so a broken CI environment cannot pass with no Isaac tests."""

import os
import sys

NO_SIM = os.environ.get("UNIENV_ISAACSIM_NO_SIM") == "1"

if not NO_SIM:
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app
else:
    collect_ignore = ["test_adaptor.py"]


def pytest_report_header(config):
    return None if not NO_SIM else "UNIENV_ISAACSIM_NO_SIM=1: test_adaptor.py (Isaac Sim tests) is not collected"


def pytest_sessionfinish(session, exitstatus):
    session.config.exit_code = int(exitstatus)


def pytest_unconfigure(config):
    sys.stdout.flush()  # os._exit skips it: the summary would be lost when piped
    os._exit(getattr(config, "exit_code", 0))  # after pytest's summary; Kit's own shutdown can hang
