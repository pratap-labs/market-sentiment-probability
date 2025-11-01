"""Token persistence helpers for Kite Connect authentication."""

import json
import streamlit as st
from datetime import datetime
from pathlib import Path as _Path
from typing import Optional, Dict


def _token_file_path() -> _Path:
    """Return the path to the persistent token file.

    Use the current working directory (where Streamlit is usually launched) as the
    project root. This avoids issues when __file__ resolution differs under
    different runtimes (tests, packaging, or import paths).
    """
    try:
        root = _Path.cwd()
    except Exception:
        root = _Path(__file__).resolve().parents[2]
    return root / ".kite_token.json"


def save_kite_token(api_key: str, access_token: str) -> None:
    """Save Kite API credentials to persistent token file."""
    p = _token_file_path()
    payload = {
        "kite_api_key": api_key,
        "kite_access_token": access_token,
        "saved_at": datetime.utcnow().isoformat() + "Z"
    }
    try:
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        # Try to restrict permissions to owner-only where possible
        try:
            p.chmod(0o600)
        except Exception:
            pass
        print(f"Saved Kite token to {p}")
    except Exception as e:
        print(f"Failed to save Kite token: {e}")
        try:
            st.warning(f"Failed to save Kite token to {p}: {e}")
        except Exception:
            pass


def load_kite_token() -> Optional[Dict[str, str]]:
    """Load Kite API credentials from persistent token file."""
    p = _token_file_path()
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Failed to read Kite token file {p}: {e}")
        try:
            st.warning(f"Failed to read saved Kite token: {e}")
        except Exception:
            pass
        return None


def clear_kite_token_file() -> None:
    """Delete the persistent token file."""
    p = _token_file_path()
    try:
        if p.exists():
            p.unlink()
            print(f"Removed saved Kite token file {p}")
    except Exception as e:
        print(f"Failed to remove Kite token file: {e}")
