"""Launches Isaac Sim (headless) once for the test session: isaaclab imports need a running app."""

import os

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app


def pytest_sessionfinish(session, exitstatus):
    session.config.exit_code = int(exitstatus)


def pytest_unconfigure(config):
    os._exit(getattr(config, "exit_code", 0))  # after pytest's summary; Kit's own shutdown can hang
