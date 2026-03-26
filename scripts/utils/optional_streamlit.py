"""Optional Streamlit compatibility for legacy helper modules."""

from __future__ import annotations


class _StreamlitStub:
    """Import-safe stub for modules that still contain legacy Streamlit render code."""

    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"Streamlit is not installed. Legacy UI method `st.{name}` is unavailable."
        )


try:
    import streamlit as st  # type: ignore
except Exception:
    st = _StreamlitStub()
