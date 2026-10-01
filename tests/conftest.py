"""Shared pytest configuration: headless matplotlib for the GUI smoke tests."""

import matplotlib

matplotlib.use("Agg")
