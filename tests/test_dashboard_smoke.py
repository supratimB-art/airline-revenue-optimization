"""Exercise the dashboard's default execution path without launching a server."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path


class _FakeContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        if name in {
            "metric", "subheader", "write", "caption", "markdown", "success",
            "warning", "error", "info", "latex", "dataframe", "plotly_chart",
            "download_button", "header", "title",
        }:
            return lambda *args, **kwargs: None
        if name in {"slider", "number_input", "select_slider"}:
            return _widget_default(name)
        raise AttributeError(name)


def _widget_default(name):
    def widget(*args, **kwargs):
        if "value" in kwargs:
            return kwargs["value"]
        if name == "select_slider" and "options" in kwargs:
            return kwargs["options"][0]
        if len(args) >= 4:
            return args[3]
        return 0
    return widget


class _FakeStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.sidebar = _FakeContext()

    def __getattr__(self, name):
        if name in {
            "set_page_config", "title", "caption", "markdown", "header", "subheader",
            "write", "error", "info", "metric", "warning", "success", "latex",
            "download_button", "dataframe", "plotly_chart",
        }:
            return lambda *args, **kwargs: None
        if name in {"slider", "number_input", "select_slider"}:
            return _widget_default(name)
        raise AttributeError(name)

    def columns(self, spec, *args, **kwargs):
        count = spec if isinstance(spec, int) else len(spec)
        return [_FakeContext() for _ in range(count)]

    def tabs(self, labels):
        return [_FakeContext() for _ in labels]

    def stop(self):
        raise RuntimeError("dashboard unexpectedly stopped on default parameters")


def test_dashboard_default_path_builds_all_charts(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setitem(sys.modules, "streamlit", _FakeStreamlit())
    # Real Plotly is retained so invalid chart-column references fail this test.
    runpy.run_path(str(root / "app" / "dashboard.py"), run_name="__main__")
