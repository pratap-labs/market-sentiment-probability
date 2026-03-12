from __future__ import annotations

import logging
import os
import time
import sys
import subprocess
import calendar
import math
import json
import io
import zipfile
import numpy as np
from datetime import date
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scripts.utils import (
    calculate_market_regime,
    calculate_portfolio_greeks,
    calculate_stress_pnl,
    calculate_var,
    enrich_position_with_greeks,
)
from scripts.data import NSEDataFetcher
from scripts.data import train_flow_probability_models as flow_train
from scripts.utils.greeks import calculate_implied_volatility
from scripts.utils.stress_testing import compute_var_es_metrics, get_weighted_scenarios, classify_bucket
from scripts.utils.forward_risk import run_advanced_forward_risk
from views.tabs.overview_tab import (
    get_alignment_status,
    get_market_signal,
    get_portfolio_health_status,
)
from views.tabs import risk_analysis_tab as ra_tab
from views.tabs import stress_testing_tab as st_tab
from views.tabs import trade_selector_tab as ts_tab

from views.tabs import historical_performance_tab as hp_tab
from views.tabs.risk_buckets_tab import (
    SimulationConfig,
    assign_buckets,
    aggregate_buckets,
    aggregate_portfolio,
    build_trades_from_saved_groups,
    build_trades_using_existing_grouping,
    compute_trade_risk,
    _build_trade_scenario_table,
    _compute_trade_payoff_curve,
    _compute_expiry_payoff_curve,
    simulate_forward_pnl,
    _load_saved_groups,
    _position_label,
    _zone_rules_table,
    _build_bucket_history,
)
import streamlit as st

try:
    from kiteconnect import KiteConnect
except Exception:
    KiteConnect = None

CACHE_DIR = ROOT / "database" / "derivatives_cache"
KITE_CREDS_FILE = CACHE_DIR / "kite_credentials.json"
POSITIONS_CACHE_FILE = CACHE_DIR / "positions_cache.json"
EQUITIES_CACHE_FILE = CACHE_DIR / "equities_holdings.json"
INDIANAPI_CACHE_FILE = ROOT / "data" / "indianapi_stock_cache.json"
EQUITY_HISTORY_CACHE_DIR = CACHE_DIR / "equity_history_cache"
LONG_TERM_FINAL_COMPOSITE_CACHE_FILE = CACHE_DIR / "long_term_final_composite_scores.json"
LONG_TERM_UNIVERSE_FILE = ROOT / "data" / "long_term_universe.json"
NIFTY50_WEIGHTS_FILE = ROOT / "data" / "nifty50_weights.json"
MANUAL_BUCKET_FILE = ROOT / "database" / "manual_bucket_overrides.json"
RISK_BUCKET_SETTINGS_FILE = ROOT / "database" / "risk_bucket_settings.json"
FRONTEND_DIST = ROOT / "dist"
KITE_TOKEN_TTL = timedelta(hours=12)
DEFAULT_REDIRECT_URL = "http://localhost:8000/auth/callback"
DEFAULT_FRONTEND_URL = "http://localhost:5173/login?auth=success"

logger = logging.getLogger("gammashield.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
# FastAPI/bare mode does not run Streamlit's script context; silence noisy warnings.
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

app = FastAPI(title="GammaShield API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _log_frontend_static_state(stage: str) -> None:
    index_file = FRONTEND_DIST / "index.html"
    client_assets_dir = FRONTEND_DIST / "client-assets"
    logger.info(
        "[static:%s] cwd=%s root=%s dist=%s dist_exists=%s index_exists=%s client_assets_exists=%s",
        stage,
        Path.cwd(),
        ROOT,
        FRONTEND_DIST,
        FRONTEND_DIST.exists(),
        index_file.exists(),
        client_assets_dir.exists(),
    )
    if client_assets_dir.exists():
        try:
            files = sorted(p.name for p in client_assets_dir.iterdir() if p.is_file())
            logger.info(
                "[static:%s] client-assets file_count=%s sample=%s",
                stage,
                len(files),
                files[:8],
            )
        except Exception as exc:
            logger.exception("[static:%s] failed to inspect client-assets: %s", stage, exc)


@app.on_event("startup")
async def startup_diagnostics() -> None:
    logger.info("[startup] GammaShield API starting")
    logger.info("[startup] python=%s", sys.version.replace("\n", " "))
    logger.info("[startup] sys.path[0:5]=%s", sys.path[:5])
    _log_frontend_static_state("startup")


class KiteHeaders(BaseModel):
    api_key: str
    access_token: str


class TradeGroupPayload(BaseModel):
    name: str
    legs: List[str]


class BucketOverridePayload(BaseModel):
    trade_id: str
    bucket: str


class RiskBucketSettingsPayload(BaseModel):
    settings: Dict[str, Any]


class PortfolioSimPayload(BaseModel):
    position_ids: List[str] = []
    virtual_positions: Optional[List[Dict[str, Any]]] = None


class VirtualPositionsPayload(BaseModel):
    virtual_positions: List[Dict[str, Any]]


def _log_advanced_simulation_snapshot(
    advanced_sim: Optional[Dict[str, Any]],
    current_spot: float,
    positions_count: int,
) -> None:
    if not isinstance(advanced_sim, dict):
        return
    cfg = advanced_sim.get("config") or {}
    logger.info(
        (
            "[ADV_SIM] version=%s spot=%.2f positions=%d horizon_days=%s n_paths=%s dt=%s seed=%s "
            "r=%s q=%s repricing_anchor=%s engines=%s pnl_modes=%s iv_rules=%s repricing_models=%s evt=%s surface=%s"
        ),
        str(advanced_sim.get("model_version", "unknown")),
        float(current_spot or 0.0),
        int(positions_count),
        cfg.get("horizon_days"),
        cfg.get("n_paths"),
        cfg.get("dt"),
        cfg.get("seed"),
        cfg.get("risk_free_rate"),
        cfg.get("dividend_yield"),
        cfg.get("repricing_anchor"),
        cfg.get("engines"),
        cfg.get("pnl_modes"),
        cfg.get("iv_rules"),
        cfg.get("repricing_models"),
        cfg.get("use_evt_overlay"),
        cfg.get("simulate_surface"),
    )

    engines = advanced_sim.get("engines") or {}
    if not isinstance(engines, dict):
        return
    for engine_name, engine_block in engines.items():
        if str(engine_name) == "gbm":
            continue
        modes = (engine_block or {}).get("modes") if isinstance(engine_block, dict) else None
        if not isinstance(modes, dict):
            continue
        for mode_name, mode_block in modes.items():
            mode_base = str((mode_block or {}).get("mode_base", mode_name))
            if mode_base != "repricing":
                continue
            kpis = (mode_block or {}).get("kpis") if isinstance(mode_block, dict) else None
            sample = (mode_block or {}).get("terminal_pnl_sample") if isinstance(mode_block, dict) else None
            if not isinstance(kpis, dict):
                continue
            quantiles = kpis.get("quantiles") or {}
            logger.info(
                (
                    "[ADV_SIM][PNL] engine=%s mode=%s mean=%s median=%s p95=%s p5=%s p1=%s "
                    "var95=%s var99=%s es95=%s es99=%s prob_loss=%s prob_breach_total=%s worst=%s"
                ),
                str(engine_name),
                f"{mode_name}[iv={str((mode_block or {}).get('iv_rule', ''))},pricing={str((mode_block or {}).get('pricing_model', ''))}]",
                kpis.get("mean"),
                kpis.get("median"),
                quantiles.get("p95") if isinstance(quantiles, dict) else None,
                quantiles.get("p5") if isinstance(quantiles, dict) else None,
                quantiles.get("p1") if isinstance(quantiles, dict) else None,
                kpis.get("var95"),
                kpis.get("var99"),
                kpis.get("es95"),
                kpis.get("es99"),
                kpis.get("prob_loss"),
                kpis.get("prob_breach_total"),
                kpis.get("worst_path_pnl"),
            )
            if isinstance(sample, list) and sample:
                try:
                    arr = np.asarray(sample, dtype=float)
                    neg_count = int(np.sum(arr < 0))
                    logger.info(
                        (
                            "[ADV_SIM][SAMPLE] engine=%s mode=%s n=%d neg_count=%d neg_pct=%.6f "
                            "min=%s p1=%s p5=%s median=%s p95=%s max=%s"
                        ),
                        str(engine_name),
                        f"{mode_name}[iv={str((mode_block or {}).get('iv_rule', ''))},pricing={str((mode_block or {}).get('pricing_model', ''))}]",
                        int(arr.size),
                        neg_count,
                        float(neg_count / arr.size) if arr.size else 0.0,
                        float(np.min(arr)),
                        float(np.percentile(arr, 1)),
                        float(np.percentile(arr, 5)),
                        float(np.percentile(arr, 50)),
                        float(np.percentile(arr, 95)),
                        float(np.max(arr)),
                    )
                except Exception:
                    logger.exception(
                        "[ADV_SIM][SAMPLE] Failed to summarize terminal_pnl_sample for engine=%s mode=%s",
                        str(engine_name),
                        str(mode_name),
                    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _append_csv_dedup(path: Path, new_df: pd.DataFrame, dedup_cols: List[str]) -> pd.DataFrame:
    if new_df.empty:
        return _read_csv(path)
    old_df = _read_csv(path)
    if old_df.empty:
        out = new_df.copy()
    else:
        out = pd.concat([old_df, new_df], ignore_index=True)
    keys = [c for c in dedup_cols if c in out.columns]
    if keys:
        out = out.drop_duplicates(subset=keys, keep="last")
    else:
        out = out.drop_duplicates(keep="last")

    # Keep cache deterministic for charting: each expiry should be chronological.
    sort_cols = [c for c in ["expiry_date", "expiry", "date", "timestamp_order"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _load_cache(name: str) -> pd.DataFrame:
    print(f"Loading cache for {name} from {CACHE_DIR / f'{name}.csv'}")
    return _read_csv(CACHE_DIR / f"{name}.csv")


def _filter_latest_snapshot(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    if out.empty:
        return df
    latest = out[date_col].max()
    return out[out[date_col] == latest].copy()


def _df_to_records(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    trimmed = df.tail(limit)
    rows = trimmed.to_dict(orient="records")
    sanitized: List[Dict[str, Any]] = []
    for row in rows:
        clean: Dict[str, Any] = {}
        for key, val in row.items():
            try:
                if pd.isna(val):
                    clean[key] = None
                    continue
            except Exception:
                pass
            if val is None or (isinstance(val, float) and not math.isfinite(val)):
                clean[key] = None
                continue
            if isinstance(val, (np.floating, np.integer)):
                v = val.item()
                if isinstance(v, float) and not math.isfinite(v):
                    clean[key] = None
                else:
                    clean[key] = v
                continue
            if isinstance(val, (pd.Timestamp, datetime, date)):
                try:
                    clean[key] = val.isoformat()
                except Exception:
                    clean[key] = str(val)
                continue
            clean[key] = val
        sanitized.append(clean)
    return sanitized


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = value
    except Exception:
        return


_load_env_file(ROOT / ".env")


def _load_kite_credentials_file() -> Dict[str, Optional[str]]:
    if not KITE_CREDS_FILE.exists():
        return {"api_key": None, "access_token": None, "saved_at": None}
    try:
        import json

        data = json.loads(KITE_CREDS_FILE.read_text())
        return {
            "api_key": data.get("api_key"),
            "access_token": data.get("access_token"),
            "saved_at": data.get("saved_at"),
        }
    except Exception:
        return {"api_key": None, "access_token": None, "saved_at": None}


def _load_positions_cache() -> List[Dict[str, Any]]:
    if not POSITIONS_CACHE_FILE.exists():
        return []
    try:
        import json

        payload = json.loads(POSITIONS_CACHE_FILE.read_text())
        return payload.get("positions", []) or []
    except Exception:
        return []


def _save_positions_cache(positions: List[Dict[str, Any]]) -> None:
    try:
        import json

        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "positions": positions,
        }
        POSITIONS_CACHE_FILE.write_text(json.dumps(payload))
    except Exception:
        return


def _fetch_positions_with_cache(kite: KiteConnect) -> List[Dict[str, Any]]:
    try:
        positions = kite.positions().get("net", []) or []
    except Exception:
        positions = []
    if positions:
        _save_positions_cache(positions)
        return positions
    return _load_positions_cache()


def _position_id(p: Dict[str, Any]) -> str:
    token = p.get("instrument_token") or p.get("token")
    if token is not None:
        return str(token)
    sym = p.get("tradingsymbol") or p.get("trading_symbol") or p.get("symbol") or ""
    expiry = p.get("expiry") or p.get("expiry_date") or ""
    strike = p.get("strike") or p.get("strike_price") or ""
    opt = p.get("option_type") or p.get("instrument_type") or ""
    return f"{sym}|{expiry}|{strike}|{opt}"


def _load_equities_cache() -> List[Dict[str, Any]]:
    if not EQUITIES_CACHE_FILE.exists():
        return []
    try:
        import json

        payload = json.loads(EQUITIES_CACHE_FILE.read_text())
        return payload.get("holdings", []) or []
    except Exception:
        return []


def _save_equities_cache(holdings: List[Dict[str, Any]]) -> None:
    try:
        import json

        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "holdings": holdings,
        }
        EQUITIES_CACHE_FILE.write_text(json.dumps(payload))
    except Exception:
        return


def _equity_history_cache_path(symbol: str) -> Path:
    safe = "".join(c for c in symbol if c.isalnum() or c in ("_", "-")).strip("-_")
    return EQUITY_HISTORY_CACHE_DIR / f"{safe}.csv"


def _load_equity_history_cache(symbol: str, max_age_hours: int = 24) -> pd.DataFrame:
    path = _equity_history_cache_path(symbol)
    if not path.exists():
        return pd.DataFrame()
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if datetime.now(timezone.utc) - mtime > timedelta(hours=max_age_hours):
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _save_equity_history_cache(symbol: str, df: pd.DataFrame) -> None:
    try:
        EQUITY_HISTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(_equity_history_cache_path(symbol), index=False)
    except Exception:
        return


def _load_long_term_final_composite_cache(max_age_hours: int = 12) -> Optional[List[Dict[str, Any]]]:
    path = LONG_TERM_FINAL_COMPOSITE_CACHE_FILE
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        saved_at_raw = payload.get("saved_at")
        rows = payload.get("rows")
        if not isinstance(saved_at_raw, str) or not isinstance(rows, list):
            return None
        saved_at = datetime.fromisoformat(saved_at_raw)
        if datetime.now(timezone.utc) - saved_at > timedelta(hours=max_age_hours):
            return None
        out: List[Dict[str, Any]] = []
        for r in rows:
            if isinstance(r, dict):
                out.append(r)
        return out
    except Exception:
        return None


def _save_long_term_final_composite_cache(rows: List[Dict[str, Any]]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "rows": rows,
        }
        LONG_TERM_FINAL_COMPOSITE_CACHE_FILE.write_text(json.dumps(payload))
    except Exception:
        return


def _load_indianapi_cache(max_age_days: int = 30) -> Dict[str, Dict[str, Any]]:
    if not INDIANAPI_CACHE_FILE.exists():
        return {}
    try:
        import json

        payload = json.loads(INDIANAPI_CACHE_FILE.read_text())
        data = payload.get("data", {}) or {}
        # Support both legacy {symbol: payload} and new {symbol: {saved_at, payload}} formats.
        cleaned: Dict[str, Dict[str, Any]] = {}
        now = datetime.now(timezone.utc)
        for sym, entry in data.items():
            if isinstance(entry, dict) and "payload" in entry and "saved_at" in entry:
                try:
                    ts = datetime.fromisoformat(str(entry.get("saved_at")))
                    if now - ts > timedelta(days=max_age_days):
                        continue
                except Exception:
                    continue
                cleaned[str(sym).upper()] = entry.get("payload", {}) or {}
            elif isinstance(entry, dict):
                # Legacy entry without per-symbol timestamp; accept but treat as stale if file is old.
                cleaned[str(sym).upper()] = entry
        return cleaned
    except Exception:
        return {}


def _save_indianapi_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    try:
        import json

        payload = {"saved_at": datetime.now(timezone.utc).isoformat(), "data": cache}
        INDIANAPI_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        INDIANAPI_CACHE_FILE.write_text(json.dumps(payload))
    except Exception:
        return


def _indianapi_headers(debug: bool = False) -> Dict[str, str]:
    token = os.getenv("INDIANAPI_KEY") or os.getenv("INDIAN_API_KEY") or os.getenv("INDIANAPI_TOKEN")
    headers = {"accept": "application/json"}
    if token:
        headers["X-Api-Key"] = token
    if debug:
        masked = None
        if token:
            masked = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "***"
        headers["X-Debug-Api-Key"] = masked or "MISSING"
    return headers


def _indianapi_base_url() -> str:
    return os.getenv("INDIANAPI_BASE_URL", "https://stock.indianapi.in").rstrip("/")


def _indianapi_get(path: str, params: Optional[Dict[str, Any]] = None, debug: bool = False) -> Dict[str, Any]:
    url = f"{_indianapi_base_url()}/{path.lstrip('/')}"
    try:
        resp = requests.get(url, params=params or {}, headers=_indianapi_headers(debug=debug), timeout=20)
        if resp.status_code >= 400:
            return {"error": f"HTTP {resp.status_code}", "detail": resp.text}
        return resp.json() if resp.content else {}
    except Exception as exc:
        return {"error": "request_failed", "detail": str(exc)}


def _load_long_term_universe() -> Dict[str, List[str]]:
    if not LONG_TERM_UNIVERSE_FILE.exists():
        return {"nifty100": [], "midcap150": []}
    try:
        import json

        payload = json.loads(LONG_TERM_UNIVERSE_FILE.read_text())
        if isinstance(payload, dict):
            return {
                "nifty100": [str(x) for x in payload.get("nifty100", [])],
                "midcap150": [str(x) for x in payload.get("midcap150", [])],
            }
    except Exception:
        return {"nifty100": [], "midcap150": []}
    return {"nifty100": [], "midcap150": []}


def _load_nifty50_weights() -> List[Dict[str, Any]]:
    if not NIFTY50_WEIGHTS_FILE.exists():
        return []
    try:
        payload = json.loads(NIFTY50_WEIGHTS_FILE.read_text())
        if not isinstance(payload, list):
            return []
        out: List[Dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol") or "").upper()
            company = str(row.get("company") or "").strip()
            try:
                weight_pct = float(row.get("weight_pct"))
            except Exception:
                continue
            if not sym or not company or not math.isfinite(weight_pct) or weight_pct <= 0:
                continue
            out.append(
                {
                    "symbol": sym,
                    "company": company,
                    "weight_pct": float(weight_pct),
                    "weight": float(weight_pct / 100.0),
                }
            )
        out = sorted(out, key=lambda r: (-float(r["weight_pct"]), str(r["symbol"])))
        return out
    except Exception:
        return []


def _json_safe(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        f = float(val)
        return f if math.isfinite(f) else None
    if isinstance(val, (np.ndarray,)):
        return [_json_safe(x) for x in val.tolist()]
    if isinstance(val, (list, tuple, set)):
        return [_json_safe(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    if isinstance(val, (pd.Timestamp, datetime, date)):
        try:
            return val.isoformat()
        except Exception:
            return str(val)
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return val


def _get_nested(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur.get(part)
    return cur


def _pick_first_num(d: Dict[str, Any], paths: List[str]) -> Optional[float]:
    for p in paths:
        val = _get_nested(d, p)
        try:
            num = float(val)
            if math.isfinite(num):
                return num
        except Exception:
            continue
    return None


def _normalize_key(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _pick_from_keymetrics(vendor: Dict[str, Any], keywords: List[str]) -> Optional[float]:
    km = vendor.get("keyMetrics")
    if not isinstance(km, dict):
        return None
    target = [_normalize_key(k) for k in keywords]
    for _, items in km.items():
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            key = it.get("key")
            if not isinstance(key, str):
                continue
            norm = _normalize_key(key)
            if all(k in norm for k in target):
                try:
                    num = float(it.get("value"))
                    if math.isfinite(num):
                        return num
                except Exception:
                    continue
    return None


def _format_inr_scale(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    return value / 1e7


def _pick_from_keymetrics_exact(vendor: Dict[str, Any], allowed: List[str], categories: Optional[List[str]] = None) -> Optional[float]:
    km = vendor.get("keyMetrics")
    if not isinstance(km, dict):
        return None
    allowed_norm = set(_normalize_key(a) for a in allowed)
    for cat, items in km.items():
        if categories and cat not in categories:
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            key = it.get("key")
            if not isinstance(key, str):
                continue
            if _normalize_key(key) in allowed_norm:
                try:
                    num = float(it.get("value"))
                    if math.isfinite(num):
                        return num
                except Exception:
                    continue
    return None


def _pick_from_keymetrics_contains(vendor: Dict[str, Any], keywords: List[str], categories: Optional[List[str]] = None) -> Optional[float]:
    km = vendor.get("keyMetrics")
    if not isinstance(km, dict):
        return None
    target = [_normalize_key(k) for k in keywords]
    for cat, items in km.items():
        if categories and cat not in categories:
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            key = it.get("key")
            if not isinstance(key, str):
                continue
            norm = _normalize_key(key)
            if all(k in norm for k in target):
                try:
                    num = float(it.get("value"))
                    if math.isfinite(num):
                        return num
                except Exception:
                    continue
    return None


def _pick_keymetrics_key(vendor: Dict[str, Any], target_key: str, categories: Optional[List[str]] = None) -> Optional[float]:
    km = vendor.get("keyMetrics")
    if not isinstance(km, dict):
        return None
    target = _normalize_key(target_key)
    for cat, items in km.items():
        if categories and cat not in categories:
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            key = it.get("key")
            if not isinstance(key, str):
                continue
            if _normalize_key(key) == target:
                try:
                    num = float(it.get("value"))
                    if math.isfinite(num):
                        return num
                except Exception:
                    return None
    return None


def _pick_keymetrics_display(vendor: Dict[str, Any], target_display: str, categories: Optional[List[str]] = None) -> Optional[float]:
    km = vendor.get("keyMetrics")
    if not isinstance(km, dict):
        return None
    target = _normalize_key(target_display)
    for cat, items in km.items():
        if categories and cat not in categories:
            continue
        if not isinstance(items, list):
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            disp = it.get("displayName")
            if not isinstance(disp, str):
                continue
            if _normalize_key(disp) == target:
                try:
                    num = float(it.get("value"))
                    if math.isfinite(num):
                        return num
                except Exception:
                    return None
    return None


def _load_manual_bucket_overrides() -> Dict[str, str]:
    if not MANUAL_BUCKET_FILE.exists():
        return {}
    try:
        import json

        data = json.loads(MANUAL_BUCKET_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


def _save_manual_bucket_overrides(overrides: Dict[str, str]) -> None:
    try:
        import json

        MANUAL_BUCKET_FILE.parent.mkdir(parents=True, exist_ok=True)
        MANUAL_BUCKET_FILE.write_text(json.dumps(overrides, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        return


def _default_risk_bucket_settings() -> Dict[str, Any]:
    return {
        "alloc_low": 50.0,
        "alloc_med": 30.0,
        "alloc_high": 20.0,
        "portfolio_es_limit": 4.0,
        "bucket_es_limit_low": 2.0,
        "bucket_es_limit_med": 3.0,
        "bucket_es_limit_high": 5.0,
        "trade_low_max": 1.0,
        "trade_med_max": 2.0,
        "sim_days": 10,
        "sim_paths": 2000,
        "iv_mode": "IV Flat",
        "iv_shock": 2.0,
        "gate_tail_ratio_watch": 30.0,
        "gate_tail_ratio_fail": 60.0,
        "gate_prob_loss_watch": 40.0,
        "gate_prob_loss_fail": 50.0,
        "gate_portfolio_breach_prob": 15.0,
        "gate_bucket_breach_prob_low": 5.0,
        "gate_bucket_breach_prob_med": 10.0,
        "gate_bucket_breach_prob_high": 15.0,
        "gate_p1_breach_fail_days": 2,
        "gate_portfolio_p10_fail_days": 4,
        "gate_bucket_p10_fail_days_med": 6,
        "gate_bucket_p10_fail_days_high": 3,
    }


def _load_risk_bucket_settings() -> Dict[str, Any]:
    defaults = _default_risk_bucket_settings()
    if not RISK_BUCKET_SETTINGS_FILE.exists():
        return defaults
    try:
        import json

        data = json.loads(RISK_BUCKET_SETTINGS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            merged = {**defaults, **data}
            return merged
    except Exception:
        return defaults
    return defaults


def _save_risk_bucket_settings(settings: Dict[str, Any]) -> None:
    try:
        import json

        RISK_BUCKET_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RISK_BUCKET_SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=True, indent=2), encoding="utf-8")
    except Exception:
        return


def _fetch_holdings_with_cache(kite: KiteConnect, use_cache: bool) -> List[Dict[str, Any]]:
    if use_cache:
        cached = _load_equities_cache()
        if cached:
            return cached
    try:
        holdings = kite.holdings() or []
    except Exception:
        holdings = []
    if holdings:
        _save_equities_cache(holdings)
        return holdings
    return _load_equities_cache()


def _is_equity_cash(pos: Dict[str, object]) -> bool:
    exchange = str(pos.get("exchange", "")).upper()
    segment = str(pos.get("segment", "")).upper()
    return "NSE" in exchange or "BSE" in exchange or "NSE" in segment or "BSE" in segment


def _fetch_equity_history(
    kite: KiteConnect,
    instrument_token: int,
    from_date: datetime,
    to_date: datetime,
) -> pd.DataFrame:
    try:
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval="day",
        )
        return pd.DataFrame(data or [])
    except Exception:
        return pd.DataFrame()


def _fetch_equity_history_cached(
    kite: KiteConnect,
    symbol: str,
    instrument_token: int,
    from_date: datetime,
    to_date: datetime,
    max_age_hours: int = 24,
    min_cache_days: int = 370,
) -> pd.DataFrame:
    cached = _load_equity_history_cache(symbol, max_age_hours=max_age_hours)
    if not cached.empty:
        # Reuse cache only when it actually spans the requested window.
        if "date" in cached.columns:
            cached_dates = pd.to_datetime(cached["date"], errors="coerce").dropna()
            if not cached_dates.empty:
                cached_start = cached_dates.min().date()
                cached_end = cached_dates.max().date()
                requested_start = pd.to_datetime(from_date).date()
                requested_end = pd.to_datetime(to_date).date()
                if cached_start <= requested_start and cached_end >= requested_end:
                    return cached
            else:
                return cached
        else:
            return cached
    requested_start = pd.to_datetime(from_date).date()
    requested_end = pd.to_datetime(to_date).date()
    min_start = requested_end - timedelta(days=max(1, int(min_cache_days)))
    fetch_start = min(requested_start, min_start)
    df = _fetch_equity_history(kite, instrument_token, fetch_start, requested_end)
    if not df.empty:
        _save_equity_history_cache(symbol, df)
    return df


def _compute_drawdown_from_peak(
    kite: KiteConnect, pos: Dict[str, object], lookback_days: int
) -> Tuple[Optional[float], Optional[str]]:
    instrument_token = pos.get("instrument_token")
    if not instrument_token:
        return None, "N/A (missing token)"
    today = datetime.now().date()
    entry_date = pos.get("entry_date")
    if entry_date:
        try:
            from_date = pd.to_datetime(entry_date).date()
        except Exception:
            from_date = today - timedelta(days=int(lookback_days))
    else:
        from_date = today - timedelta(days=int(lookback_days))
    to_date = today
    df = _fetch_equity_history(
        kite,
        int(instrument_token),
        datetime.combine(from_date, datetime.min.time()),
        datetime.combine(to_date, datetime.min.time()),
    )
    if df.empty or "close" not in df.columns:
        return None, "N/A (needs price history)"
    peak = df["close"].max()
    ltp = pos.get("last_price") or pos.get("ltp") or pos.get("mark_price")
    if ltp is None or not peak:
        return None, "N/A"
    drawdown_pct = (float(ltp) - float(peak)) / float(peak) * 100.0
    return drawdown_pct, None


def _compute_time_under_water_days(
    kite: KiteConnect, pos: Dict[str, object], lookback_days: int
) -> Tuple[Optional[object], bool, Optional[str]]:
    avg_cost = pos.get("average_price") or pos.get("avg_price") or pos.get("cost_price")
    if not avg_cost:
        return None, False, "N/A (missing cost)"
    instrument_token = pos.get("instrument_token")
    if not instrument_token:
        return None, False, "N/A (missing token)"
    today = datetime.now().date()
    to_date = today
    from_date = today - timedelta(days=int(lookback_days))
    df = _fetch_equity_history(
        kite,
        int(instrument_token),
        datetime.combine(from_date, datetime.min.time()),
        datetime.combine(to_date, datetime.min.time()),
    )
    if df.empty or "close" not in df.columns or "date" not in df.columns:
        return None, False, "N/A (history unavailable)"
    df = df.dropna(subset=["close", "date"]).copy()
    if df.empty:
        return None, False, "N/A (history unavailable)"
    df["date"] = pd.to_datetime(df["date"])
    last_date = df["date"].max()
    breakeven = df[df["close"] >= float(avg_cost)]
    if breakeven.empty:
        return f">{lookback_days}d", True, None
    last_breakeven = breakeven["date"].max()
    return (last_date - last_breakeven).days, False, None


def _equity_es99_per_symbol(
    equities: List[Dict[str, object]],
    lookback_days: int,
) -> Dict[str, Optional[float]]:
    iv_regime, _ = ra_tab.get_iv_regime(35)
    scenarios = get_weighted_scenarios(iv_regime)
    bucket_probs, _, _ = ra_tab.compute_historical_bucket_probabilities(
        lookback=int(lookback_days),
        smoothing_enabled=False,
        smoothing_span=63,
    )
    if not bucket_probs:
        bucket_probs = ra_tab.DEFAULT_BUCKET_PROBS.copy()
    bucket_counts: Dict[str, int] = {}
    for scenario in scenarios:
        bucket = classify_bucket(
            {"type": scenario.category.upper(), "dS_pct": scenario.ds_pct, "dIV_pts": scenario.div_pts}
        )
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    scenario_probs: List[Tuple[float, float]] = []
    for scenario in scenarios:
        bucket = classify_bucket(
            {"type": scenario.category.upper(), "dS_pct": scenario.ds_pct, "dIV_pts": scenario.div_pts}
        )
        bucket_prob = bucket_probs.get(bucket, 0.0)
        count = bucket_counts.get(bucket, 0)
        prob = bucket_prob / count if count and bucket_prob > 0 else 0.0
        scenario_probs.append((scenario.ds_pct, prob))
    es99_map: Dict[str, Optional[float]] = {}
    for pos in equities:
        symbol = pos.get("tradingsymbol") or pos.get("symbol") or pos.get("instrument") or "—"
        qty = float(pos.get("quantity") or pos.get("qty") or 0.0)
        avg_cost = pos.get("average_price") or pos.get("avg_price") or pos.get("cost_price")
        ltp = pos.get("last_price") or pos.get("ltp") or pos.get("mark_price") or avg_cost or 0.0
        nav = float(avg_cost) * qty if avg_cost else float(ltp) * qty
        if nav <= 0 or qty == 0:
            es99_map[symbol] = None
            continue
        losses = []
        for ds_pct, prob in scenario_probs:
            pnl = qty * float(ltp) * (ds_pct / 100.0)
            loss_pct = (-pnl / nav * 100.0) if pnl < 0 else 0.0
            losses.append({"loss_pct": loss_pct, "prob": prob})
        metrics = compute_var_es_metrics(losses, nav)
        es99_map[symbol] = float(metrics.get("ES99Value", 0.0))
    return es99_map


def _is_token_expired(saved_at: Optional[str]) -> bool:
    if not saved_at:
        return True
    try:
        saved_dt = datetime.fromisoformat(saved_at)
    except ValueError:
        return True
    if saved_dt.tzinfo is None:
        saved_dt = saved_dt.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - saved_dt >= KITE_TOKEN_TTL


def _save_kite_credentials_file(api_key: str, access_token: str) -> None:
    try:
        import json

        payload = {
            "api_key": api_key,
            "access_token": access_token,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        KITE_CREDS_FILE.write_text(json.dumps(payload))
    except Exception as exc:
        logger.error("failed to save kite credentials: %s", exc)


def _get_kite_client(api_key: str, access_token: str):
    if KiteConnect is None:
        raise HTTPException(status_code=500, detail="kiteconnect not available")
    if not api_key or not access_token:
        raise HTTPException(status_code=401, detail="Missing Kite credentials")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def _get_credentials(request: Request) -> KiteHeaders:
    api_key = request.headers.get("x-kite-api-key")
    access_token = request.headers.get("x-kite-access-token")

    if not api_key or not access_token:
        api_key = api_key or os.getenv("KITE_API_KEY")
        access_token = access_token or os.getenv("KITE_ACCESS_TOKEN")

    if not api_key or not access_token:
        cached = _load_kite_credentials_file()
        if cached.get("api_key") and cached.get("access_token") and not _is_token_expired(cached.get("saved_at")):
            api_key = api_key or cached.get("api_key")
            access_token = access_token or cached.get("access_token")

    if not api_key or not access_token:
        raise HTTPException(status_code=401, detail="Kite credentials required")
    return KiteHeaders(api_key=api_key, access_token=access_token)


def _prepare_rb_state(options_df: pd.DataFrame, nifty_df: pd.DataFrame) -> None:
    st.session_state["options_df_cache"] = options_df
    st.session_state["nifty_df_cache"] = nifty_df
    st.session_state.setdefault("tba_sim_days", 10)
    st.session_state.setdefault("tba_sim_paths", 2000)
    st.session_state.setdefault("tba_iv_mode", "IV Flat")
    st.session_state.setdefault("tba_iv_shock", 2.0)
    st.session_state.setdefault("tba_alloc_low", 50.0)
    st.session_state.setdefault("tba_alloc_med", 30.0)
    st.session_state.setdefault("tba_alloc_high", 20.0)
    st.session_state.setdefault("tba_bucket_es_limit_low", 2.0)
    st.session_state.setdefault("tba_bucket_es_limit_med", 3.0)
    st.session_state.setdefault("tba_bucket_es_limit_high", 5.0)
    st.session_state.setdefault("tba_trade_low_max", 1.0)
    st.session_state.setdefault("tba_trade_med_max", 2.0)
    st.session_state.setdefault("tba_spot_input", 0.0)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.time()
    response = await call_next(request)
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["x-request-id"] = request_id
    return response


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat() + "Z"}


@app.get("/auth/login")
def auth_login():
    if KiteConnect is None:
        raise HTTPException(status_code=500, detail="kiteconnect not available")
    api_key = os.getenv("KITE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="KITE_API_KEY missing")
    redirect_uri = os.getenv("KITE_REDIRECT_URL", DEFAULT_REDIRECT_URL)
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    if "redirect_uri=" not in login_url:
        if "?" in login_url:
            login_url = f"{login_url}&redirect_uri={redirect_uri}"
        else:
            login_url = f"{login_url}?redirect_uri={redirect_uri}"
    return RedirectResponse(login_url)


@app.get("/auth/callback")
def auth_callback(request_token: str, status: Optional[str] = None):
    if KiteConnect is None:
        raise HTTPException(status_code=500, detail="kiteconnect not available")
    api_key = os.getenv("KITE_API_KEY")
    api_secret = os.getenv("KITE_API_SECRET")
    if not api_key or not api_secret:
        raise HTTPException(status_code=500, detail="KITE_API_KEY/SECRET missing")
    kite = KiteConnect(api_key=api_key)
    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
    except Exception as exc:
        logger.error("kite auth failed: %s", exc)
        return HTMLResponse(content=f"Auth failed: {exc}", status_code=400)
    access_token = data.get("access_token")
    if not access_token:
        return HTMLResponse(content="Auth failed: no access_token returned", status_code=400)
    _save_kite_credentials_file(api_key, access_token)
    frontend_url = os.getenv("FRONTEND_URL", DEFAULT_FRONTEND_URL)
    return RedirectResponse(frontend_url)


@app.get("/auth/status")
def auth_status():
    cached = _load_kite_credentials_file()
    return {
        "has_token": bool(cached.get("access_token")),
        "token_expired": _is_token_expired(cached.get("saved_at")),
        "saved_at": cached.get("saved_at"),
    }


@app.get("/derivatives/nifty-ohlcv")
def nifty_ohlcv(limit: int = 500):
    df = _load_cache("nifty_ohlcv")
    if df.empty:
        raise HTTPException(status_code=404, detail="NIFTY OHLCV cache missing")
    if limit is None or int(limit) <= 0:
        rows = df.to_dict(orient="records")
    else:
        rows = df.tail(int(limit)).to_dict(orient="records")
    summary = {
        "latest_close": float(df["close"].iloc[-1]) if "close" in df.columns else None,
        "high_2y": float(df["high"].max()) if "high" in df.columns else None,
        "low_2y": float(df["low"].min()) if "low" in df.columns else None,
        "avg_volume": float(df["volume"].mean()) if "volume" in df.columns else None,
    }
    return {"rows": rows, "summary": summary}


@app.get("/derivatives/futures")
def futures_data(limit: int = 1000):
    df = _load_cache("nifty_futures")
    if df.empty:
        raise HTTPException(status_code=404, detail="Futures cache missing")
    sort_cols = [c for c in ["expiry_date", "expiry", "date", "timestamp_order"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    try:
        return {"rows": _df_to_records(df, limit)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to serialize futures cache: {exc}")


@app.get("/derivatives/options")
def options_data(limit: int = 1000):
    df_ce = _load_cache("nifty_options_ce")
    df_pe = _load_cache("nifty_options_pe")
    if df_ce.empty and df_pe.empty:
        raise HTTPException(status_code=404, detail="Options cache missing")
    return {
        "ce": df_ce.tail(limit).to_dict(orient="records") if not df_ce.empty else [],
        "pe": df_pe.tail(limit).to_dict(orient="records") if not df_pe.empty else [],
    }


@app.get("/derivatives/options-chain")
def options_chain(expiry: Optional[str] = None):
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    options_df = _filter_latest_snapshot(options_df, "date")
    if options_df.empty:
        raise HTTPException(status_code=404, detail="Options cache missing")

    expiry_col = "expiry" if "expiry" in options_df.columns else "expiry_date"
    if expiry_col not in options_df.columns:
        raise HTTPException(status_code=404, detail="Expiry column missing")
    options_df[expiry_col] = pd.to_datetime(options_df[expiry_col], errors="coerce")
    options_df = options_df.dropna(subset=[expiry_col])
    options_df["expiry_str"] = options_df[expiry_col].dt.date.astype(str)
    expiry_list = sorted([e for e in options_df["expiry_str"].unique().tolist() if e])
    selected_expiry = expiry if expiry in expiry_list else (expiry_list[0] if expiry_list else None)
    if selected_expiry:
        options_df = options_df[options_df["expiry_str"] == selected_expiry]

    oi_col = "open_int" if "open_int" in options_df.columns else "open_interest"
    ltp_col = "ltp" if "ltp" in options_df.columns else ("last_price" if "last_price" in options_df.columns else "close")
    expiry_dt = None
    if selected_expiry:
        try:
            expiry_dt = pd.to_datetime(selected_expiry).to_pydatetime()
        except Exception:
            expiry_dt = None

    def _month_code(dt: datetime) -> str:
        if dt.month <= 9:
            return str(dt.month)
        return {10: "O", 11: "N", 12: "D"}.get(dt.month, str(dt.month))

    def _is_monthly_expiry(expiry_date: datetime) -> bool:
        last_day = datetime(expiry_date.year, expiry_date.month, calendar.monthrange(expiry_date.year, expiry_date.month)[1])
        while last_day.weekday() != 1:
            last_day -= timedelta(days=1)
        if last_day.month == 3 and last_day.day == 31:
            last_day -= timedelta(days=1)
        return expiry_date.date() == last_day.date()

    def _build_symbol(expiry_date: datetime, strike: float, opt_type: str) -> str:
        yy = str(expiry_date.year)[-2:]
        strike_str = f"{int(round(strike))}"
        if _is_monthly_expiry(expiry_date):
            mon = expiry_date.strftime("%b").upper()
            return f"NIFTY{yy}{mon}{strike_str}{opt_type}"
        mm = _month_code(expiry_date)
        dd = f"{expiry_date.day:02d}"
        return f"NIFTY{yy}{mm}{dd}{strike_str}{opt_type}"

    rows = []
    if not options_df.empty:
        strikes = sorted(options_df["strike_price"].dropna().unique().tolist())
        for strike in strikes:
            strike_df = options_df[options_df["strike_price"] == strike]
            ce_row = strike_df[strike_df["option_type"] == "CE"].head(1)
            pe_row = strike_df[strike_df["option_type"] == "PE"].head(1)
            ce = ce_row.iloc[0] if not ce_row.empty else None
            pe = pe_row.iloc[0] if not pe_row.empty else None
            row_expiry = expiry_dt
            try:
                if ce is not None and expiry_col in ce:
                    row_expiry = pd.to_datetime(ce[expiry_col]).to_pydatetime()
                elif pe is not None and expiry_col in pe:
                    row_expiry = pd.to_datetime(pe[expiry_col]).to_pydatetime()
            except Exception:
                pass
            if row_expiry is None:
                continue
            rows.append(
                {
                    "strike": float(strike),
                    "call_ltp": float(ce[ltp_col]) if ce is not None and ltp_col in ce else None,
                    "call_oi": float(ce[oi_col]) if ce is not None and oi_col in ce else None,
                    "call_symbol": _build_symbol(row_expiry, float(strike), "CE"),
                    "call_token": None,
                    "put_ltp": float(pe[ltp_col]) if pe is not None and ltp_col in pe else None,
                    "put_oi": float(pe[oi_col]) if pe is not None and oi_col in pe else None,
                    "put_symbol": _build_symbol(row_expiry, float(strike), "PE"),
                    "put_token": None,
                }
            )

    spot = None
    try:
        spot = float(options_df["underlying_value"].iloc[0])
    except Exception:
        spot = None
    date_str = None
    try:
        date_str = pd.to_datetime(options_df["date"].iloc[0]).date().isoformat()
    except Exception:
        date_str = None

    return {
        "expiry": selected_expiry,
        "expiry_list": expiry_list,
        "spot": spot,
        "date": date_str,
        "rows": rows,
    }


@app.get("/market-regime")
def market_regime(expiry: Optional[str] = None):
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    nifty_df = _load_cache("nifty_ohlcv")
    if options_df.empty or nifty_df.empty:
        raise HTTPException(status_code=404, detail="Required cache missing")
    regime = calculate_market_regime(options_df, nifty_df)

    iv_rv_series: List[Dict[str, object]] = []
    pcr_series: List[Dict[str, object]] = []
    expiry_list: List[str] = []
    expiry_filter = (expiry or "all").strip().lower()
    try:
        df = options_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date_str"] = df["date"].dt.date.astype(str)
        expiry_col = "expiry" if "expiry" in df.columns else "expiry_date"
        df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce")
        df["expiry_str"] = df[expiry_col].dt.date.astype(str)
        expiry_list = sorted([e for e in df["expiry_str"].dropna().unique().tolist() if e])
        if expiry_filter != "all":
            df = df[df["expiry_str"] == expiry_filter]
        unique_dates = sorted(df["date_str"].unique())[-90:]

        nifty_df = nifty_df.copy()
        nifty_df["date"] = pd.to_datetime(nifty_df["date"], errors="coerce")
        nifty_df = nifty_df.dropna(subset=["date"])
        nifty_df["date_str"] = nifty_df["date"].dt.date.astype(str)
        nifty_df = nifty_df.sort_values("date")
        returns = nifty_df["close"].pct_change()
        rv = returns.rolling(30).std() * (252 ** 0.5) * 100.0
        rv_map = dict(zip(nifty_df["date_str"], rv.fillna(0.0)))

        for date_str in unique_dates:
            day_options = df[df["date_str"] == date_str]
            if day_options.empty:
                continue
            try:
                day_spot = float(day_options["underlying_value"].iloc[0])
            except Exception:
                continue
            if not day_spot or pd.isna(day_spot):
                continue
            atm_strike = round(day_spot / 100) * 100
            day_atm = day_options[
                (day_options["strike_price"] >= atm_strike - 100)
                & (day_options["strike_price"] <= atm_strike + 100)
            ]
            ivs: List[float] = []
            for _, row in day_atm.iterrows():
                try:
                    opt_iv = calculate_implied_volatility(
                        float(row.get("ltp", 0.0)),
                        day_spot,
                        float(row.get("strike_price", 0.0)),
                        0.027,
                        str(row.get("option_type", "")).upper(),
                    )
                    if opt_iv and opt_iv > 0:
                        ivs.append(float(opt_iv) * 100.0)
                except Exception:
                    continue
            iv_val = float(sum(ivs) / len(ivs)) if ivs else 0.0
            rv_val = float(rv_map.get(date_str, 0.0))
            iv_rv_series.append({"date": date_str, "iv": iv_val, "rv": rv_val})

            pcr_val = 0.0
            try:
                puts = day_options[day_options["option_type"] == "PE"]
                calls = day_options[day_options["option_type"] == "CE"]
                strike_step = 100
                strike_window = {atm_strike + strike_step * i for i in range(-10, 11)}
                puts = puts[puts["strike_price"].isin(strike_window)]
                calls = calls[calls["strike_price"].isin(strike_window)]
                oi_col = "open_int" if "open_int" in day_options.columns else "open_interest"
                if oi_col in day_options.columns:
                    put_oi = float(puts[oi_col].sum())
                    call_oi = float(calls[oi_col].sum())
                    pcr_val = put_oi / call_oi if call_oi > 0 else 0.0
            except Exception:
                pcr_val = 0.0
            pcr_series.append({"date": date_str, "pcr_oi": pcr_val})
    except Exception:
        iv_rv_series = []
        pcr_series = []

    return {
        "regime": regime,
        "iv_rv_series": iv_rv_series,
        "pcr_series": pcr_series,
        "expiry_list": expiry_list,
        "expiry_filter": expiry_filter,
    }


@app.get("/portfolio/positions")
def portfolio_positions(request: Request, spot: Optional[float] = None):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    if options_df.empty:
        raise HTTPException(status_code=404, detail="Options cache missing")

    positions = _fetch_positions_with_cache(kite)
    nifty_df = _load_cache("nifty_ohlcv")
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    if spot:
        try:
            current_spot = float(spot)
        except Exception:
            pass
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]
    return {"positions": enriched, "current_spot": current_spot}


@app.get("/portfolio/summary")
def portfolio_summary(request: Request, spot: Optional[float] = None):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    if options_df.empty:
        raise HTTPException(status_code=404, detail="Options cache missing")

    positions = _fetch_positions_with_cache(kite)
    nifty_df = _load_cache("nifty_ohlcv")
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    if spot:
        try:
            current_spot = float(spot)
        except Exception:
            pass
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]

    portfolio_greeks = calculate_portfolio_greeks(enriched)
    var_95 = calculate_var(enriched, current_spot, nifty_df)
    stress_up_2 = calculate_stress_pnl(enriched, current_spot, 1.02, 0)
    stress_down_2 = calculate_stress_pnl(enriched, current_spot, 0.98, 0)
    stress_iv_up = calculate_stress_pnl(enriched, current_spot, 1.0, 0.05)

    margins = kite.margins()
    equity = margins.get("equity", {})
    margin_available = float(equity.get("available", {}).get("live_balance", 0.0))
    margin_used = float(equity.get("utilised", {}).get("debits", 0.0))
    account_size = margin_available + margin_used

    total_pnl = sum(p.get("pnl", 0) for p in enriched)
    avg_dte = sum(p.get("dte", 0) for p in enriched) / len(enriched) if enriched else 30
    days_in_trade = max(30 - avg_dte, 1)
    roi_pct = (total_pnl / account_size * 100) if account_size > 0 else 0
    roi_annualized = (total_pnl / account_size) / (days_in_trade / 365) * 100 if account_size > 0 else 0
    total_theta = portfolio_greeks.get("net_theta", 0.0)
    days_to_recover = abs(total_pnl / total_theta) if total_theta not in (0, None) else None
    theta_efficiency = (total_pnl / total_theta * 100) if total_theta not in (0, None) else None
    theta_pct_capital = (abs(total_theta) / account_size * 100) if account_size > 0 else 0
    notional_exposure = sum(abs(p.get("quantity", 0)) * p.get("strike", 0) for p in enriched)
    leverage_ratio = (notional_exposure / account_size) if account_size > 0 else 0

    delta_notional = portfolio_greeks.get("net_delta", 0.0) * current_spot
    delta_notional_pct = (abs(delta_notional) / account_size * 100) if account_size > 0 else 0
    vega_pct = (abs(portfolio_greeks.get("net_vega", 0.0)) / account_size * 100) if account_size > 0 else 0

    total_value = sum(abs(p.get("pnl", 0)) for p in enriched) or 100000
    var_pct = (var_95 / total_value * 100) if total_value else 0
    margin_pct = (margin_used / account_size * 100) if account_size > 0 else 0
    health_emoji, health_status = get_portfolio_health_status(
        {"theta_efficiency": theta_efficiency or 0},
        var_pct,
        margin_pct,
    )

    near_expiry_count = sum(1 for p in enriched if p.get("dte", 999) < 7)
    alignments = []
    if regime:
        alignments = get_alignment_status(
            regime.get("iv_rank", 50),
            portfolio_greeks.get("net_vega", 0.0),
            regime.get("pcr_oi", 1.0),
            portfolio_greeks.get("net_delta", 0.0),
            regime.get("term_structure", 0),
            near_expiry_count,
        )

    recommendations: List[Dict[str, str]] = []
    if regime:
        iv_rank = regime.get("iv_rank", 50)
        vrp = regime.get("vrp", 0)
        term_structure = regime.get("term_structure", 0)
        skew = regime.get("skew", 0)

        if iv_rank > 70 and vrp > 0.05:
            recommendations.append({"type": "Market Strategy", "recommendation": "Sell Volatility (prefer far expiry credit spreads)"})
        elif iv_rank < 30 and vrp < -0.05:
            recommendations.append({"type": "Market Strategy", "recommendation": "Buy Volatility (prefer debit spreads, long options)"})
        else:
            recommendations.append({"type": "Market Strategy", "recommendation": "Neutral — Consider Iron Condors or balanced spreads"})

        net_delta = portfolio_greeks.get("net_delta", 0.0)
        if abs(net_delta) > 20:
            delta_adjust = abs(net_delta * current_spot)
            if net_delta > 20:
                recommendations.append({"type": "Portfolio Adjustment", "recommendation": f"Reduce long delta by ~₹{delta_adjust:,.0f}; add short calls or bearish spreads"})
            else:
                recommendations.append({"type": "Portfolio Adjustment", "recommendation": f"Reduce short delta by ~₹{delta_adjust:,.0f}; add long calls or bullish spreads"})
        else:
            recommendations.append({"type": "Portfolio Adjustment", "recommendation": "Delta exposure within acceptable range"})

        if var_pct > 5:
            recommendations.append({"type": "Risk Flag", "recommendation": f"High VaR ({var_pct:.1f}%) — Consider reducing position sizes"})
        elif near_expiry_count > 5 and term_structure < -0.02:
            recommendations.append({"type": "Risk Flag", "recommendation": f"{near_expiry_count} near-expiry positions in backwardation — Roll to next month"})
        else:
            recommendations.append({"type": "Risk Flag", "recommendation": "Risk metrics within acceptable limits"})

        if abs(skew) > 0.015:
            if skew > 0:
                recommendations.append({"type": "Opportunities", "recommendation": f"Puts overpriced by {skew*100:.1f}% — Short put spreads attractive"})
            else:
                recommendations.append({"type": "Opportunities", "recommendation": f"Calls overpriced by {abs(skew)*100:.1f}% — Short call spreads attractive"})

    signal_emoji, signal_text, signal_reco = get_market_signal(regime or {})

    return {
        "account_size": account_size,
        "margin_available": margin_available,
        "margin_used": margin_used,
        "margin_pct": margin_pct,
        "total_pnl": total_pnl,
        "roi_pct": roi_pct,
        "roi_annualized": roi_annualized,
        "theta_day": total_theta,
        "theta_pct_capital": theta_pct_capital,
        "days_to_recover": days_to_recover,
        "theta_efficiency": theta_efficiency,
        "notional_exposure": notional_exposure,
        "leverage_ratio": leverage_ratio,
        "greeks": portfolio_greeks,
        "delta_notional": delta_notional,
        "delta_notional_pct": delta_notional_pct,
        "vega_pct_capital": vega_pct,
        "var_95": var_95,
        "var_pct": var_pct,
        "stress": {
            "up_2": stress_up_2,
            "down_2": stress_down_2,
            "iv_up_5": stress_iv_up,
        },
        "current_spot": current_spot,
        "market_signal": {
            "emoji": signal_emoji,
            "text": signal_text,
            "recommendation": signal_reco,
        },
        "alignment": alignments,
        "health": {
            "emoji": health_emoji,
            "status": health_status,
        },
        "recommendations": recommendations,
    }


@app.post("/portfolio/virtual-greeks")
def portfolio_virtual_greeks(payload: VirtualPositionsPayload, request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    if options_df.empty:
        raise HTTPException(status_code=404, detail="Options cache missing")

    nifty_df = _load_cache("nifty_ohlcv")
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000

    positions = payload.virtual_positions or []
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]
    greeks = calculate_portfolio_greeks(enriched)
    return {"greeks": greeks, "current_spot": current_spot}


@app.get("/equities/holdings")
def equities_holdings(request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    holdings = kite.holdings() or []
    return {"holdings": holdings}


@app.get("/equities/enriched")
def equities_enriched(
    request: Request,
    use_cache: bool = True,
):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    holdings = _fetch_holdings_with_cache(kite, use_cache=use_cache)
    equities = [p for p in holdings if _is_equity_cash(p)] if holdings else []
    cache = _load_indianapi_cache(max_age_days=30) if use_cache else {}

    enriched_rows: List[Dict[str, Any]] = []
    updated = False
    for pos in equities:
        symbol = pos.get("tradingsymbol") or pos.get("symbol") or pos.get("instrument") or "—"
        key = str(symbol).upper()
        vendor = cache.get(key)
        if not vendor:
            vendor = _indianapi_get("/stock", {"name": symbol})
            cache[key] = vendor
            updated = True
        enriched_rows.append({**pos, "vendor": vendor})

    if updated and use_cache:
        _save_indianapi_cache(cache)

    return {"rows": enriched_rows}


@app.get("/equities/summary")
def equities_summary(
    request: Request,
    shock_levels: str = "-5,-10,-20,-30",
    current_shock: float = -10.0,
    lookback_days: int = 365,
    capital_base: str = "auto",
    use_cache: bool = True,
):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    holdings = _fetch_holdings_with_cache(kite, use_cache=use_cache)

    equities = [p for p in holdings if _is_equity_cash(p)] if holdings else []
    if not equities:
        return {
            "equity_sleeve_value": 0.0,
            "allocation_pct": 0.0,
            "sleeve_drawdown_pct": None,
            "stress_loss_inr": 0.0,
            "stress_loss_pct": 0.0,
            "equity_es99_contrib_pct": None,
            "pct_underwater": 0.0,
            "weighted_time_under_water_days": None,
            "rows": [],
            "warnings": {"history_unavailable": False, "drawdown_history_unavailable": False},
            "shock_losses": [],
            "alloc_series": [],
            "top_stress": [],
            "risk_concentration": {},
        }

    rows: List[Dict[str, Any]] = []
    history_unavailable = False
    drawdown_history_unavailable = False
    es99_map = _equity_es99_per_symbol(equities, int(lookback_days))
    es99_available = any(v is not None for v in es99_map.values())

    for pos in equities:
        symbol = pos.get("tradingsymbol") or pos.get("symbol") or pos.get("instrument") or "—"
        qty = float(pos.get("quantity") or pos.get("qty") or 0.0)
        avg_cost = pos.get("average_price") or pos.get("avg_price") or pos.get("cost_price")
        ltp = pos.get("last_price") or pos.get("ltp") or pos.get("mark_price") or avg_cost or 0.0
        market_value = float(ltp) * qty if ltp is not None else 0.0
        cost_basis = float(avg_cost) * qty if avg_cost is not None else None
        unreal_pnl = (float(ltp) - float(avg_cost)) * qty if avg_cost is not None else None
        unreal_pnl_pct = ((float(ltp) - float(avg_cost)) / float(avg_cost) * 100.0) if avg_cost else None
        return_vs_cost_pct = ((float(ltp) - float(avg_cost)) / float(avg_cost) * 100.0) if avg_cost else None
        stress_loss = float(ltp) * qty * (float(current_shock) / 100.0)

        time_under_water, censored, tuw_note = _compute_time_under_water_days(kite, pos, lookback_days)
        drawdown_from_peak, dd_note = _compute_drawdown_from_peak(kite, pos, lookback_days)
        if dd_note and "history" in dd_note:
            drawdown_history_unavailable = True
        time_under_water_display = None
        if tuw_note:
            time_under_water_display = tuw_note
            if "history" in tuw_note:
                history_unavailable = True
        elif censored:
            time_under_water_display = time_under_water
        else:
            time_under_water_display = f"{time_under_water}d" if time_under_water is not None else "N/A"

        rows.append(
            {
                "symbol": symbol,
                "qty": qty,
                "avg_cost": avg_cost,
                "ltp": ltp,
                "market_value": market_value,
                "unreal_pnl": unreal_pnl,
                "unreal_pnl_pct": unreal_pnl_pct,
                "return_vs_cost_pct": return_vs_cost_pct,
                "drawdown_pct": drawdown_from_peak,
                "time_under_water": time_under_water_display,
                "es99_inr": es99_map.get(symbol),
                "es99_pct": None,
                "stress_loss": stress_loss,
                "cost_basis": cost_basis,
            }
        )

    df = pd.DataFrame(rows)
    equity_value = df["market_value"].sum()
    cost_basis_sum = df["cost_basis"].dropna().sum() if df["cost_basis"].notna().any() else None

    if capital_base == "cost_basis" and cost_basis_sum is not None:
        equity_capital_base = cost_basis_sum
        base_label = "Cost basis"
    elif capital_base == "market_value":
        equity_capital_base = equity_value
        base_label = "Market value"
    else:
        equity_capital_base = cost_basis_sum if cost_basis_sum is not None else equity_value
        base_label = "Cost basis" if cost_basis_sum is not None else "Market value"

    margins = kite.margins()
    equity = margins.get("equity", {})
    margin_available = float(equity.get("available", {}).get("live_balance", 0.0))
    margin_used = float(equity.get("utilised", {}).get("debits", 0.0))
    total_portfolio_value = margin_available + margin_used if (margin_available or margin_used) else equity_value
    denom_label = "Account size" if (margin_available or margin_used) else "Equity value"

    allocation_pct = (equity_value / total_portfolio_value * 100.0) if total_portfolio_value else 0.0
    sleeve_drawdown_pct = None
    if cost_basis_sum:
        sleeve_drawdown_pct = df["unreal_pnl"].dropna().sum() / cost_basis_sum * 100.0

    stress_loss_inr_sleeve = df["stress_loss"].sum()
    stress_loss_pct = (stress_loss_inr_sleeve / equity_capital_base * 100.0) if equity_capital_base else 0.0

    shock_levels_list = [float(x) for x in shock_levels.split(",") if x.strip()]
    shock_losses = [
        {"level": lvl, "loss": float((df["market_value"] * (lvl / 100.0)).sum())}
        for lvl in (shock_levels_list or [-10.0])
    ]

    underwater_count = 0
    weighted_tuw = 0.0
    weight_total = 0.0
    for _, row in df.iterrows():
        if row["return_vs_cost_pct"] is not None and row["return_vs_cost_pct"] < 0:
            underwater_count += 1
        tuw_val = row["time_under_water"]
        if isinstance(tuw_val, str) and tuw_val.startswith(">"):
            try:
                tuw_numeric = float(tuw_val.replace(">", "").replace("d", ""))
            except Exception:
                tuw_numeric = None
        elif isinstance(tuw_val, str) and tuw_val.endswith("d"):
            try:
                tuw_numeric = float(tuw_val.replace("d", ""))
            except Exception:
                tuw_numeric = None
        elif isinstance(tuw_val, (int, float)):
            tuw_numeric = float(tuw_val)
        else:
            tuw_numeric = None
        if tuw_numeric is not None and row["market_value"] > 0:
            weighted_tuw += tuw_numeric * row["market_value"]
            weight_total += row["market_value"]

    pct_underwater = (underwater_count / len(df) * 100.0) if len(df) else 0.0
    weighted_tuw_days = (weighted_tuw / weight_total) if weight_total else None

    portfolio_es99_inr = None
    equity_es99_inr = df["es99_inr"].dropna().sum() if es99_available else None
    equity_es99_pct = None
    if equity_es99_inr is not None and portfolio_es99_inr:
        equity_es99_pct = equity_es99_inr / float(portfolio_es99_inr) * 100.0

    if equity_es99_inr is not None and portfolio_es99_inr:
        df.loc[df["es99_inr"].notna(), "es99_pct"] = (
            df.loc[df["es99_inr"].notna(), "es99_inr"] / float(portfolio_es99_inr) * 100.0
        )

    alloc_series = (
        df.set_index("symbol")["market_value"].sort_values(ascending=False).head(10)
    )
    top_stress = df.set_index("symbol")["stress_loss"].sort_values().head(10)

    top1_pct = (alloc_series.iloc[0] / equity_value * 100.0) if not alloc_series.empty and equity_value else 0.0
    top5_pct = (alloc_series.head(5).sum() / equity_value * 100.0) if equity_value else 0.0
    stress_abs = df.set_index("symbol")["stress_loss"].abs().sort_values(ascending=False)
    total_stress_abs = stress_abs.sum()
    top1_stress_pct = (stress_abs.iloc[0] / total_stress_abs * 100.0) if not stress_abs.empty and total_stress_abs else 0.0
    top5_stress_pct = (stress_abs.head(5).sum() / total_stress_abs * 100.0) if total_stress_abs else 0.0

    return {
        "equity_sleeve_value": equity_value,
        "allocation_pct": allocation_pct,
        "allocation_denom": denom_label,
        "sleeve_drawdown_pct": sleeve_drawdown_pct,
        "stress_loss_inr": stress_loss_inr_sleeve,
        "stress_loss_pct": stress_loss_pct,
        "equity_es99_contrib_pct": equity_es99_pct,
        "pct_underwater": pct_underwater,
        "weighted_time_under_water_days": weighted_tuw_days,
        "capital_base": equity_capital_base,
        "capital_base_label": base_label,
        "warnings": {"history_unavailable": history_unavailable, "drawdown_history_unavailable": drawdown_history_unavailable, "es99_available": es99_available},
        "shock_losses": shock_losses,
        "alloc_series": [{"symbol": k, "value": float(v)} for k, v in alloc_series.items()],
        "top_stress": [{"symbol": k, "value": float(v)} for k, v in top_stress.items()],
        "risk_concentration": {
            "top1_pct": top1_pct,
            "top5_pct": top5_pct,
            "top1_stress_pct": top1_stress_pct,
            "top5_stress_pct": top5_stress_pct,
        },
        "rows": df.to_dict(orient="records"),
    }


@app.get("/equities/simulate")
def equities_simulate(
    request: Request,
    lookback_days: int = 504,
    horizon_days: int = 30,
    n_paths: int = 2000,
    use_cache: bool = True,
):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    holdings = _fetch_holdings_with_cache(kite, use_cache=use_cache)
    equities = [p for p in holdings if _is_equity_cash(p)] if holdings else []
    if not equities:
        return {"status": "no_equities", "kpis": {}, "paths_sample": []}

    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=max(lookback_days, 30))
    histories: List[Dict[str, Any]] = []
    total_value = 0.0

    for pos in equities:
        symbol = pos.get("tradingsymbol") or pos.get("symbol") or pos.get("instrument") or "—"
        token = pos.get("instrument_token")
        qty = float(pos.get("quantity") or pos.get("qty") or 0.0)
        ltp = float(pos.get("last_price") or pos.get("ltp") or pos.get("mark_price") or 0.0)
        if not token or qty == 0 or ltp <= 0:
            continue
        df = _fetch_equity_history_cached(kite, str(symbol), int(token), from_date, to_date)
        if df.empty or "close" not in df.columns:
            continue
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if close.size < 60:
            continue
        rets = np.log(close / close.shift(1)).dropna().to_numpy(dtype=float)
        if rets.size < 40:
            continue
        current_value = ltp * qty
        total_value += current_value
        histories.append(
            {
                "symbol": symbol,
                "returns": rets,
                "current_value": current_value,
            }
        )

    if not histories or total_value <= 0:
        return {"status": "insufficient_history", "kpis": {}, "paths_sample": []}

    horizon = max(1, min(int(horizon_days), 252))
    paths = max(200, min(int(n_paths), 20000))
    port_terms = np.zeros(paths, dtype=float)

    rng = np.random.default_rng(42)
    for h in histories:
        rets = h["returns"]
        curr = float(h["current_value"])
        idx = rng.integers(0, rets.size, size=(paths, horizon))
        sampled = rets[idx]
        term_ret = np.exp(np.sum(sampled, axis=1)) - 1.0
        port_terms += curr * term_ret

    term = port_terms
    sorted_term = np.sort(term)
    var99 = float(np.percentile(term, 1))
    var95 = float(np.percentile(term, 5))
    es99 = float(np.mean(sorted_term[: max(1, int(0.01 * sorted_term.size))]))
    mean = float(np.mean(term))
    median = float(np.median(term))
    prob_loss = float(np.mean(term < 0))

    if term.size > 3000:
        pick = np.linspace(0, term.size - 1, 3000, dtype=int)
        sample = term[pick]
    else:
        sample = term

    return {
        "status": "ok",
        "config": {
            "lookback_days": int(lookback_days),
            "horizon_days": int(horizon),
            "n_paths": int(paths),
        },
        "kpis": {
            "mean": mean,
            "median": median,
            "var95": var95,
            "var99": var99,
            "es99": es99,
            "prob_loss": prob_loss,
            "portfolio_value": float(total_value),
        },
        "paths_sample": sample.astype(float).tolist(),
    }


@app.get("/long-term/instruments")
def long_term_instruments(
    request: Request,
    limit: int = 250,
    instrument_type: Optional[str] = None,
    exchange: Optional[str] = None,
):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    try:
        instruments = kite.instruments() or []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch instruments: {exc}")
    if instrument_type:
        itype = instrument_type.upper()
        instruments = [i for i in instruments if str(i.get("instrument_type", "")).upper() == itype]
    if exchange:
        exch = exchange.upper()
        instruments = [i for i in instruments if str(i.get("exchange", "")).upper() == exch]
    limit = max(1, min(int(limit), 1000))
    rows = instruments[:limit]
    return {"rows": rows, "count": len(instruments), "shown": len(rows)}


@app.get("/long-term/universe-match")
def long_term_universe_match(request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    universe = _load_long_term_universe()
    try:
        instruments = kite.instruments() or []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch instruments: {exc}")

    inst_by_symbol = {}
    for inst in instruments:
        sym = str(inst.get("tradingsymbol") or "").upper()
        if sym and sym not in inst_by_symbol:
            inst_by_symbol[sym] = inst

    def _match(group: List[str]) -> Dict[str, Any]:
        matched = []
        missing = []
        for sym in group:
            key = str(sym).upper()
            inst = inst_by_symbol.get(key)
            if inst:
                matched.append(inst)
            else:
                missing.append(sym)
        return {"matched": matched, "missing": missing}

    nifty = _match(universe.get("nifty100", []))
    mid = _match(universe.get("midcap150", []))

    return {
        "nifty100": {
            "matched": nifty["matched"],
            "missing": nifty["missing"],
            "count": len(nifty["matched"]),
        },
        "midcap150": {
            "matched": mid["matched"],
            "missing": mid["missing"],
            "count": len(mid["matched"]),
        },
    }


@app.get("/long-term/ohlcv")
def long_term_ohlcv(
    request: Request,
    days: int = 30,
    limit: int = 250,
    symbol: Optional[str] = None,
    use_cache: bool = True,
):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    universe = _load_long_term_universe()
    try:
        instruments = kite.instruments() or []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch instruments: {exc}")

    inst_by_symbol = {}
    for inst in instruments:
        sym = str(inst.get("tradingsymbol") or "").upper()
        if sym and sym not in inst_by_symbol:
            inst_by_symbol[sym] = inst

    if symbol:
        symbols = [str(symbol).upper()]
    else:
        symbols = [str(s).upper() for s in (universe.get("nifty100", []) + universe.get("midcap150", []))]
    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=max(7, int(days)))

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        inst = inst_by_symbol.get(sym)
        if not inst:
            continue
        token = inst.get("instrument_token")
        if not token:
            continue
        df = _fetch_equity_history_cached(kite, sym, int(token), from_date, to_date) if use_cache else _fetch_equity_history(kite, int(token), from_date, to_date)
        if df.empty:
            continue
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        for _, r in df.tail(max(1, int(days))).iterrows():
            rows.append(
                {
                    "symbol": sym,
                    "date": _json_safe(r.get("date")),
                    "open": _json_safe(r.get("open")),
                    "high": _json_safe(r.get("high")),
                    "low": _json_safe(r.get("low")),
                    "close": _json_safe(r.get("close")),
                    "volume": _json_safe(r.get("volume")),
                }
            )

    rows = sorted(rows, key=lambda x: (str(x.get("symbol")), str(x.get("date"))))
    if limit:
        rows = rows[: max(1, min(int(limit), 2000))]
    return {"rows": rows, "count": len(rows)}


@app.get("/long-term/ohlcv-latest")
def long_term_ohlcv_latest(
    request: Request,
    days: int = 30,
    limit: int = 250,
    symbol: Optional[str] = None,
    use_cache: bool = True,
):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    universe = _load_long_term_universe()
    try:
        instruments = kite.instruments() or []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch instruments: {exc}")

    inst_by_symbol = {}
    for inst in instruments:
        sym = str(inst.get("tradingsymbol") or "").upper()
        if sym and sym not in inst_by_symbol:
            inst_by_symbol[sym] = inst

    if symbol:
        symbols = [str(symbol).upper()]
    else:
        symbols = [str(s).upper() for s in (universe.get("nifty100", []) + universe.get("midcap150", []))]
    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=max(7, int(days)))

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        inst = inst_by_symbol.get(sym)
        if not inst:
            continue
        token = inst.get("instrument_token")
        if not token:
            continue
        df = _fetch_equity_history_cached(kite, sym, int(token), from_date, to_date) if use_cache else _fetch_equity_history(kite, int(token), from_date, to_date)
        if df.empty:
            continue
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        last = df.tail(1)
        if last.empty:
            continue
        r = last.iloc[0]
        rows.append(
            {
                "symbol": sym,
                "date": _json_safe(r.get("date")),
                "open": _json_safe(r.get("open")),
                "high": _json_safe(r.get("high")),
                "low": _json_safe(r.get("low")),
                "close": _json_safe(r.get("close")),
                "volume": _json_safe(r.get("volume")),
            }
        )

    rows = sorted(rows, key=lambda x: (-(float(x.get("volume") or 0.0)), str(x.get("symbol"))))
    if limit:
        rows = rows[: max(1, min(int(limit), 2000))]
    return {"rows": rows, "count": len(rows)}


@app.get("/long-term/momentum-scores")
def long_term_momentum_scores(
    request: Request,
    lookback_days: int = 260,
    limit: int = 250,
    use_cache: bool = True,
):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    universe = _load_long_term_universe()
    try:
        instruments = kite.instruments() or []
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to fetch instruments: {exc}")

    inst_by_symbol = {}
    for inst in instruments:
        sym = str(inst.get("tradingsymbol") or "").upper()
        if sym and sym not in inst_by_symbol:
            inst_by_symbol[sym] = inst

    symbols = [str(s).upper() for s in (universe.get("nifty100", []) + universe.get("midcap150", []))]
    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=max(120, int(lookback_days)))

    metric_rows: List[Dict[str, Any]] = []
    fallback_windows = [max(120, int(lookback_days)), 180, 120, 90, 60, 45]
    # Keep order but drop duplicates.
    fallback_windows = list(dict.fromkeys([w for w in fallback_windows if w > 0]))
    for sym in symbols:
        inst = inst_by_symbol.get(sym)
        if not inst:
            continue
        token = inst.get("instrument_token")
        if not token:
            continue

        df = pd.DataFrame()
        for window_days in fallback_windows:
            window_from = to_date - timedelta(days=int(window_days))
            if use_cache:
                df = _fetch_equity_history_cached(
                    kite,
                    sym,
                    int(token),
                    window_from,
                    to_date,
                    max_age_hours=24,
                )
            else:
                df = _fetch_equity_history(kite, int(token), window_from, to_date)
            if not df.empty:
                break
        if df.empty:
            continue

        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df.get("close"), errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        if df.empty or len(df) < 20:
            continue

        closes = df["close"].astype(float).reset_index(drop=True)
        latest_close = float(closes.iloc[-1])

        def _ret(n: int) -> Optional[float]:
            if len(closes) <= n:
                return None
            prev = float(closes.iloc[-(n + 1)])
            if prev <= 0:
                return None
            return (latest_close / prev) - 1.0

        r21 = _ret(21)
        r63 = _ret(63)
        r126 = _ret(126)
        sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else None
        sma200 = float(closes.tail(200).mean()) if len(closes) >= 200 else None
        trend_strength = ((sma50 / sma200) - 1.0) if sma50 and sma200 and sma200 > 0 else None
        prev_55_high = float(closes.iloc[:-1].tail(55).max()) if len(closes) > 55 else None
        breakout_55 = ((latest_close / prev_55_high) - 1.0) if prev_55_high and prev_55_high > 0 else None
        ret_series = closes.pct_change().dropna()
        vol_63 = float(ret_series.tail(63).std(ddof=0) * np.sqrt(252.0)) if len(ret_series) >= 20 else None
        risk_adj_63 = (r63 / vol_63) if r63 is not None and vol_63 and vol_63 > 0 else None

        metric_rows.append(
            {
                "symbol": sym,
                "date": _json_safe(df["date"].iloc[-1]),
                "close": _json_safe(latest_close),
                "momentum_1m_pct": _json_safe(r21 * 100.0) if r21 is not None else None,
                "momentum_3m_pct": _json_safe(r63 * 100.0) if r63 is not None else None,
                "momentum_6m_pct": _json_safe(r126 * 100.0) if r126 is not None else None,
                "sma50": _json_safe(sma50),
                "sma200": _json_safe(sma200),
                "trend_strength_pct": _json_safe(trend_strength * 100.0) if trend_strength is not None else None,
                "breakout_55d_pct": _json_safe(breakout_55 * 100.0) if breakout_55 is not None else None,
                "vol_3m_pct": _json_safe(vol_63 * 100.0) if vol_63 is not None else None,
                "risk_adj_3m": _json_safe(risk_adj_63) if risk_adj_63 is not None else None,
            }
        )

    if not metric_rows:
        return {"rows": [], "count": 0}

    score_df = pd.DataFrame(metric_rows)
    score_specs = {
        "score_1m": "momentum_1m_pct",
        "score_3m": "momentum_3m_pct",
        "score_6m": "momentum_6m_pct",
        "score_trend": "trend_strength_pct",
        "score_breakout": "breakout_55d_pct",
        "score_risk_adj": "risk_adj_3m",
    }
    score_cols: List[str] = []
    for score_col, metric_col in score_specs.items():
        if metric_col in score_df.columns:
            vals = pd.to_numeric(score_df[metric_col], errors="coerce")
            score_df[score_col] = vals.rank(method="average", pct=True) * 100.0
            score_cols.append(score_col)

    score_df["composite_score"] = score_df[score_cols].mean(axis=1, skipna=True) if score_cols else np.nan
    score_df = score_df.sort_values(["composite_score", "symbol"], ascending=[False, True])
    if limit:
        score_df = score_df.head(max(1, min(int(limit), 2000)))

    rows = []
    for row in score_df.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            clean[k] = _json_safe(v)
        rows.append(clean)
    return {"rows": rows, "count": len(rows)}


@app.get("/long-term/fundamentals")
def long_term_fundamentals(request: Request, symbol: Optional[str] = None):
    _get_credentials(request)
    universe = _load_long_term_universe()
    raw_cache = _load_indianapi_cache(max_age_days=3650)
    cache: Dict[str, Dict[str, Any]] = {}
    # Store with per-symbol timestamps.
    for k, v in raw_cache.items():
        cache[k] = {"saved_at": datetime.now(timezone.utc).isoformat(), "payload": v}

    if symbol:
        symbols = [str(symbol).upper()]
    else:
        symbols = [str(s).upper() for s in (universe.get("nifty100", []) + universe.get("midcap150", []))]
    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        vendor = (cache.get(sym) or {}).get("payload") or {}
        logger.info("[INDIANAPI] cache-only symbol=%s hit=%s", sym, bool(vendor))
        industry = (
            _get_nested(vendor, "industry")
            or _get_nested(vendor, "companyProfile.mgIndustry")
            or _get_nested(vendor, "sector")
            or _get_nested(vendor, "companyProfile.industry")
        )
        company = (
            vendor.get("companyName")
            or _get_nested(vendor, "companyProfile.companyName")
            or _get_nested(vendor, "companyProfile.company_name")
        )
        market_cap = _pick_first_num(
            vendor,
            [
                "stockDetailsReusableData.marketCap",
                "stockDetailsReusableData.market_cap",
                "keyMetrics.marketCap",
                "marketCap",
                "companyProfile.marketCap",
                "companyProfile.market_cap",
            ],
        )
        if market_cap is None:
            market_cap = _pick_from_keymetrics_contains(vendor, ["market", "cap"])
        eps = _pick_first_num(vendor, ["keyMetrics.eps", "eps", "financials.eps", "earnings.eps"])
        if eps is None:
            eps = _pick_from_keymetrics_contains(vendor, ["earnings", "per", "share"], categories=["persharedata"])
        pe = _pick_first_num(
            vendor,
            [
                "stockDetailsReusableData.pPerEBasicExcludingExtraordinaryItemsTTM",
                "stockDetailsReusableData.priceToEarningsValueRatio",
                "keyMetrics.pe",
                "pe",
                "peRatio",
                "valuation.pe",
            ],
        )
        if pe is None:
            pe = _pick_from_keymetrics_contains(vendor, ["price", "earnings"], categories=["valuation"])
        roe = _pick_first_num(vendor, ["keyMetrics.roe", "roe"])
        if roe is None:
            roe = _pick_from_keymetrics_contains(vendor, ["return", "on", "equity"], categories=["mgmtEffectiveness"])
        roce = _pick_first_num(vendor, ["keyMetrics.roce", "roce"])
        if roce is None:
            roce = _pick_from_keymetrics_contains(vendor, ["return", "on", "capital"], categories=["mgmtEffectiveness"])
        revenue = _pick_first_num(
            vendor,
            [
                "stockDetailsReusableData.totalRevenue",
                "financials.revenue",
                "incomeStatement.revenue",
            ],
        )
        if revenue is None:
            revenue = _pick_from_keymetrics_contains(vendor, ["revenue"], categories=["incomeStatement"])
        net_income = _pick_first_num(
            vendor,
            [
                "stockDetailsReusableData.NetIncome",
                "financials.netIncome",
                "incomeStatement.netIncome",
            ],
        )
        if net_income is None:
            net_income = _pick_from_keymetrics_contains(vendor, ["net", "income"], categories=["incomeStatement"])
        rows.append(
            {
                "symbol": sym,
                "company": company,
                "industry": industry,
                "market_cap": _json_safe(market_cap),
                "eps": _json_safe(eps),
                "pe": _json_safe(pe),
                "roe": _json_safe(roe),
                "roce": _json_safe(roce),
                "revenue": _json_safe(revenue),
                "net_income": _json_safe(net_income),
                "revenue_fmt": _format_inr_scale(revenue),
                "net_income_fmt": _format_inr_scale(net_income),
                "market_cap_fmt": _format_inr_scale(market_cap),
                "DilutedNormalizedEPS": _pick_keymetrics_key(vendor, "DilutedNormalizedEPS"),
                "DilutedEPSExcludingExtraOrdItems": _pick_keymetrics_key(vendor, "DilutedEPSExcludingExtraOrdItems"),
                "NetIncome": _pick_keymetrics_key(vendor, "NetIncome"),
                "TotalRevenue": _pick_keymetrics_key(vendor, "TotalRevenue"),
                "OperatingIncome": _pick_keymetrics_key(vendor, "OperatingIncome"),
                "ePSChangePercentTTMOverTTM": _pick_keymetrics_key(vendor, "ePSChangePercentTTMOverTTM"),
                "revenueChangePercentTTMPOverTTM": _pick_keymetrics_key(vendor, "revenueChangePercentTTMPOverTTM"),
                "ePSGrowthRate5Year": _pick_keymetrics_key(vendor, "ePSGrowthRate5Year"),
                "returnOnAverageEquityMostRecentFiscalYear": _pick_keymetrics_key(vendor, "returnOnAverageEquityMostRecentFiscalYear"),
                "returnOnInvestmentMostRecentFiscalYear": _pick_keymetrics_key(vendor, "returnOnInvestmentMostRecentFiscalYear"),
                "operatingMarginTrailing12Month": _pick_keymetrics_key(vendor, "operatingMarginTrailing12Month"),
                "netProfitMarginPercentTrailing12Month": _pick_keymetrics_key(vendor, "netProfitMarginPercentTrailing12Month"),
                "TotalDebt": _pick_keymetrics_key(vendor, "TotalDebt"),
                "TotalEquity": _pick_keymetrics_key(vendor, "TotalEquity"),
                "ltDebtPerEquityMostRecentFiscalYear": _pick_keymetrics_key(vendor, "ltDebtPerEquityMostRecentFiscalYear"),
                "netInterestCoverageMostRecentFiscalYear": _pick_keymetrics_key(vendor, "netInterestCoverageMostRecentFiscalYear"),
                "CashfromOperatingActivities": _pick_keymetrics_key(vendor, "CashfromOperatingActivities"),
                "freeCashFlowMostRecentFiscalYear": _pick_keymetrics_key(vendor, "freeCashFlowMostRecentFiscalYear"),
                "pPerEBasicExcludingExtraordinaryItemsTTM": _pick_keymetrics_key(vendor, "pPerEBasicExcludingExtraordinaryItemsTTM"),
                "pegRatio": _pick_keymetrics_key(vendor, "pegRatio"),
                "priceToSalesTrailing12Month": _pick_keymetrics_key(vendor, "priceToSalesTrailing12Month"),
                "priceToBookMostRecentFiscalYear": _pick_keymetrics_key(vendor, "priceToBookMostRecentFiscalYear"),
            }
        )

    return {"rows": rows, "count": len(rows)}


@app.get("/long-term/earnings-scores")
def long_term_earnings_scores(request: Request, symbol: Optional[str] = None, limit: int = 250):
    _get_credentials(request)
    universe = _load_long_term_universe()
    raw_cache = _load_indianapi_cache(max_age_days=3650)
    cache: Dict[str, Dict[str, Any]] = {}
    for k, v in raw_cache.items():
        cache[k] = {"saved_at": datetime.now(timezone.utc).isoformat(), "payload": v}

    if symbol:
        symbols = [str(symbol).upper()]
    else:
        symbols = [str(s).upper() for s in (universe.get("nifty100", []) + universe.get("midcap150", []))]

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        vendor = (cache.get(sym) or {}).get("payload") or {}

        eps_qoq_yoy = (
            _pick_keymetrics_key(vendor, "ePSChangePercentMostRecentQuarter1YearAgo")
            or _pick_keymetrics_key(vendor, "ePSChangePercentQTPOverQTP")
        )
        eps_ttm = _pick_keymetrics_key(vendor, "ePSChangePercentTTMOverTTM")
        revenue_qoq_yoy = (
            _pick_keymetrics_key(vendor, "revenueChangePercentMostRecentQuarter1YearAgo")
            or _pick_keymetrics_key(vendor, "revenueChangePercentQTPOverQTP")
        )

        op_margin_ttm = _pick_keymetrics_key(vendor, "operatingMarginTrailing12Month")
        op_margin_5y = _pick_keymetrics_key(vendor, "operatingMargin5YearAverage")
        op_margin_trend = None
        if op_margin_ttm is not None and op_margin_5y is not None:
            op_margin_trend = op_margin_ttm - op_margin_5y

        rows.append(
            {
                "symbol": sym,
                "eps_qoq_yoy_pct": _json_safe(eps_qoq_yoy),
                "eps_ttm_growth_pct": _json_safe(eps_ttm),
                "revenue_qoq_yoy_pct": _json_safe(revenue_qoq_yoy),
                "operating_margin_trend_pct": _json_safe(op_margin_trend),
                "operating_margin_ttm_pct": _json_safe(op_margin_ttm),
                "operating_margin_5y_avg_pct": _json_safe(op_margin_5y),
            }
        )

    if not rows:
        return {"rows": [], "count": 0}

    score_df = pd.DataFrame(rows)
    score_specs = {
        "score_eps_qoq_yoy": ("eps_qoq_yoy_pct", 0.40),
        "score_eps_ttm": ("eps_ttm_growth_pct", 0.30),
        "score_revenue_qoq_yoy": ("revenue_qoq_yoy_pct", 0.20),
        "score_op_margin_trend": ("operating_margin_trend_pct", 0.10),
    }

    available_weight_cols: List[str] = []
    for score_col, (metric_col, weight) in score_specs.items():
        vals = pd.to_numeric(score_df.get(metric_col), errors="coerce")
        score_df[score_col] = vals.rank(method="average", pct=True) * 100.0
        w_col = f"{score_col}_weighted"
        score_df[w_col] = score_df[score_col] * weight
        available_weight_cols.append(w_col)

    row_weight_sum = pd.Series(0.0, index=score_df.index)
    for score_col, (_, weight) in score_specs.items():
        row_weight_sum = row_weight_sum + score_df[score_col].notna().astype(float) * weight

    weighted_sum = score_df[available_weight_cols].sum(axis=1, skipna=True)
    score_df["earnings_acceleration_score"] = np.where(
        row_weight_sum > 0,
        weighted_sum / row_weight_sum,
        np.nan,
    )
    score_df["coverage_weight"] = row_weight_sum * 100.0

    score_df = score_df.sort_values(
        ["earnings_acceleration_score", "coverage_weight", "symbol"],
        ascending=[False, False, True],
    )
    if limit:
        score_df = score_df.head(max(1, min(int(limit), 2000)))

    out_rows: List[Dict[str, Any]] = []
    for row in score_df.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            if k.endswith("_weighted"):
                continue
            clean[k] = _json_safe(v)
        out_rows.append(clean)
    return {"rows": out_rows, "count": len(out_rows)}


@app.get("/long-term/quality-scores")
def long_term_quality_scores(request: Request, symbol: Optional[str] = None, limit: int = 250):
    _get_credentials(request)
    universe = _load_long_term_universe()
    raw_cache = _load_indianapi_cache(max_age_days=3650)
    cache: Dict[str, Dict[str, Any]] = {}
    for k, v in raw_cache.items():
        cache[k] = {"saved_at": datetime.now(timezone.utc).isoformat(), "payload": v}

    if symbol:
        symbols = [str(symbol).upper()]
    else:
        symbols = [str(s).upper() for s in (universe.get("nifty100", []) + universe.get("midcap150", []))]

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        vendor = (cache.get(sym) or {}).get("payload") or {}

        roe = _pick_keymetrics_key(vendor, "returnOnAverageEquityMostRecentFiscalYear")
        roce = _pick_keymetrics_key(vendor, "returnOnInvestmentMostRecentFiscalYear")
        debt_to_equity = _pick_keymetrics_key(vendor, "ltDebtPerEquityMostRecentFiscalYear")
        interest_coverage = _pick_keymetrics_key(vendor, "netInterestCoverageMostRecentFiscalYear")
        operating_margin = _pick_keymetrics_key(vendor, "operatingMarginTrailing12Month")
        net_profit_margin = _pick_keymetrics_key(vendor, "netProfitMarginPercentTrailing12Month")
        beta = (
            _pick_keymetrics_key(vendor, "beta")
            or _pick_from_keymetrics_contains(vendor, ["beta"], categories=["stockPriceSummary"])
            or _pick_from_keymetrics_contains(vendor, ["beta"])
        )

        rows.append(
            {
                "symbol": sym,
                "roe_pct": _json_safe(roe),
                "roce_pct": _json_safe(roce),
                "debt_to_equity": _json_safe(debt_to_equity),
                "interest_coverage": _json_safe(interest_coverage),
                "operating_margin_pct": _json_safe(operating_margin),
                "net_profit_margin_pct": _json_safe(net_profit_margin),
                "beta": _json_safe(beta),
            }
        )

    if not rows:
        return {"rows": [], "count": 0}

    score_df = pd.DataFrame(rows)
    score_specs = {
        "score_roe": ("roe_pct", 0.30, False),
        "score_roce": ("roce_pct", 0.20, False),
        "score_debt_to_equity": ("debt_to_equity", 0.20, True),
        "score_interest_coverage": ("interest_coverage", 0.20, False),
        "score_operating_margin": ("operating_margin_pct", 0.10, False),
    }

    weighted_cols: List[str] = []
    for score_col, (metric_col, weight, inverse) in score_specs.items():
        vals = pd.to_numeric(score_df.get(metric_col), errors="coerce")
        if inverse:
            vals = -vals
        score_df[score_col] = vals.rank(method="average", pct=True) * 100.0
        w_col = f"{score_col}_weighted"
        score_df[w_col] = score_df[score_col] * weight
        weighted_cols.append(w_col)

    row_weight_sum = pd.Series(0.0, index=score_df.index)
    for score_col, (_, weight, _) in score_specs.items():
        row_weight_sum = row_weight_sum + score_df[score_col].notna().astype(float) * weight

    weighted_sum = score_df[weighted_cols].sum(axis=1, skipna=True)
    score_df["quality_score"] = np.where(
        row_weight_sum > 0,
        weighted_sum / row_weight_sum,
        np.nan,
    )
    score_df["coverage_weight"] = row_weight_sum * 100.0

    score_df = score_df.sort_values(
        ["quality_score", "coverage_weight", "symbol"],
        ascending=[False, False, True],
    )
    if limit:
        score_df = score_df.head(max(1, min(int(limit), 2000)))

    out_rows: List[Dict[str, Any]] = []
    for row in score_df.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            if k.endswith("_weighted"):
                continue
            clean[k] = _json_safe(v)
        out_rows.append(clean)
    return {"rows": out_rows, "count": len(out_rows)}


@app.get("/long-term/final-composite-scores")
def long_term_final_composite_scores(
    request: Request,
    symbol: Optional[str] = None,
    limit: int = 250,
    use_cache: bool = True,
):
    _get_credentials(request)

    if use_cache:
        cached_rows = _load_long_term_final_composite_cache(max_age_hours=12)
        if isinstance(cached_rows, list):
            out_rows = cached_rows
            if symbol:
                target = str(symbol).upper()
                out_rows = [r for r in out_rows if str(r.get("symbol", "")).upper() == target]
            if limit:
                out_rows = out_rows[: max(1, min(int(limit), 2000))]
            return {"rows": out_rows, "count": len(out_rows)}

    momentum_payload = long_term_momentum_scores(
        request=request,
        lookback_days=320,
        limit=2000,
        use_cache=use_cache,
    )
    earnings_payload = long_term_earnings_scores(
        request=request,
        symbol=symbol,
        limit=2000,
    )
    quality_payload = long_term_quality_scores(
        request=request,
        symbol=symbol,
        limit=2000,
    )

    momentum_rows = momentum_payload.get("rows", []) if isinstance(momentum_payload, dict) else []
    earnings_rows = earnings_payload.get("rows", []) if isinstance(earnings_payload, dict) else []
    quality_rows = quality_payload.get("rows", []) if isinstance(quality_payload, dict) else []

    momentum_by_symbol: Dict[str, Dict[str, Any]] = {
        str(r.get("symbol", "")).upper(): r for r in momentum_rows if isinstance(r, dict)
    }
    earnings_by_symbol: Dict[str, Dict[str, Any]] = {
        str(r.get("symbol", "")).upper(): r for r in earnings_rows if isinstance(r, dict)
    }
    quality_by_symbol: Dict[str, Dict[str, Any]] = {
        str(r.get("symbol", "")).upper(): r for r in quality_rows if isinstance(r, dict)
    }

    all_symbols = sorted(set(momentum_by_symbol.keys()) | set(earnings_by_symbol.keys()) | set(quality_by_symbol.keys()))

    def _to_finite(v: Any) -> Optional[float]:
        try:
            x = float(v)
            return x if math.isfinite(x) else None
        except Exception:
            return None

    out_rows: List[Dict[str, Any]] = []
    for sym in all_symbols:
        m = _to_finite((momentum_by_symbol.get(sym) or {}).get("composite_score"))
        e = _to_finite((earnings_by_symbol.get(sym) or {}).get("earnings_acceleration_score"))
        q = _to_finite((quality_by_symbol.get(sym) or {}).get("quality_score"))

        weight_sum = (0.55 if m is not None else 0.0) + (0.30 if e is not None else 0.0) + (0.15 if q is not None else 0.0)
        weighted_sum = (m or 0.0) * 0.55 + (e or 0.0) * 0.30 + (q or 0.0) * 0.15
        final_score = (weighted_sum / weight_sum) if weight_sum > 0 else None

        out_rows.append(
            {
                "symbol": sym,
                "momentum_composite_score": _json_safe(m),
                "earnings_acceleration_score": _json_safe(e),
                "quality_score": _json_safe(q),
                "coverage_weight": _json_safe(weight_sum * 100.0),
                "final_composite_score": _json_safe(final_score),
            }
        )

    out_rows = sorted(
        out_rows,
        key=lambda r: (
            -(float(r.get("final_composite_score")) if isinstance(r.get("final_composite_score"), (int, float)) else float("-inf")),
            -(float(r.get("coverage_weight")) if isinstance(r.get("coverage_weight"), (int, float)) else 0.0),
            str(r.get("symbol", "")),
        ),
    )

    if use_cache:
        _save_long_term_final_composite_cache(out_rows)

    if symbol:
        target = str(symbol).upper()
        out_rows = [r for r in out_rows if str(r.get("symbol", "")).upper() == target]
    if limit:
        out_rows = out_rows[: max(1, min(int(limit), 2000))]

    return {"rows": out_rows, "count": len(out_rows)}


@app.get("/long-term/constituents")
def long_term_constituents(
    request: Request,
    top_n: int = 50,
    lookback_days: int = 30,
    use_cache: bool = True,
):
    _get_credentials(request)
    top_n = max(1, min(int(top_n), 200))
    ranked = _load_nifty50_weights()[:top_n]
    if not ranked:
        return {"rows": [], "count": 0}

    score_payload = long_term_final_composite_scores(
        request=request,
        symbol=None,
        limit=2000,
        use_cache=use_cache,
    )
    score_rows = score_payload.get("rows", []) if isinstance(score_payload, dict) else []
    score_by_symbol: Dict[str, float] = {}
    for row in score_rows:
        if not isinstance(row, dict):
            continue
        sym = str(row.get("symbol") or "").upper()
        try:
            s = float(row.get("final_composite_score"))
            if sym and math.isfinite(s):
                score_by_symbol[sym] = s
        except Exception:
            continue

    out_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(ranked, start=1):
        sym = str(item["symbol"]).upper()
        weight = float(item["weight"])
        weight_pct = float(item["weight_pct"])
        score = score_by_symbol.get(sym)

        sdf = _load_equity_history_cache(sym, max_age_hours=24 * 365 * 20)
        latest: Dict[str, Any] = {}
        if not sdf.empty:
            sdf = sdf.copy()
            if "date" in sdf.columns:
                sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")
            if "close" in sdf.columns:
                sdf["close"] = pd.to_numeric(sdf["close"], errors="coerce")
            if "volume" in sdf.columns:
                sdf["volume"] = pd.to_numeric(sdf["volume"], errors="coerce")
            if "date" in sdf.columns:
                sdf = sdf.dropna(subset=["date"]).sort_values("date")
            if not sdf.empty:
                lr = sdf.tail(1).iloc[0]
                latest = {
                    "date": lr.get("date"),
                    "close": lr.get("close"),
                    "volume": lr.get("volume"),
                }

        out_rows.append(
            {
                "rank": idx,
                "symbol": sym,
                "company": item.get("company"),
                "final_composite_score": _json_safe(score),
                "weight": _json_safe(weight),
                "weight_pct": _json_safe(weight_pct),
                "date": _json_safe(latest.get("date")),
                "close": _json_safe(latest.get("close")),
                "volume": _json_safe(latest.get("volume")),
            }
        )

    return {"rows": out_rows, "count": len(out_rows)}


@app.get("/long-term/constituents-contribution-timeseries")
def long_term_constituents_contribution_timeseries(
    request: Request,
    top_n: int = 20,
    days: int = 180,
    use_cache: bool = True,
):
    _get_credentials(request)
    top_n = max(1, min(int(top_n), 100))
    days = max(10, min(int(days), 730))

    cons_payload = long_term_constituents(
        request=request,
        top_n=top_n,
        lookback_days=max(days, 30),
        use_cache=use_cache,
    )
    cons_rows = cons_payload.get("rows", []) if isinstance(cons_payload, dict) else []
    if not cons_rows:
        return {"rows": [], "count": 0, "constituents": []}

    weight_by_symbol: Dict[str, float] = {}
    for r in cons_rows:
        if not isinstance(r, dict):
            continue
        sym = str(r.get("symbol") or "").upper()
        if not sym:
            continue
        try:
            w = float(r.get("weight") or 0.0)
            if math.isfinite(w) and w > 0:
                weight_by_symbol[sym] = w
        except Exception:
            continue
    if not weight_by_symbol:
        return {"rows": [], "count": 0, "constituents": []}

    nifty_df = _load_cache("nifty_ohlcv")
    if nifty_df.empty or "date" not in nifty_df.columns or "close" not in nifty_df.columns:
        raise HTTPException(status_code=404, detail="nifty_ohlcv cache missing")
    nifty_df = nifty_df.copy()
    nifty_df["date"] = pd.to_datetime(nifty_df["date"], errors="coerce")
    nifty_df["close"] = pd.to_numeric(nifty_df["close"], errors="coerce")
    nifty_df = nifty_df.dropna(subset=["date", "close"]).sort_values("date")
    nifty_df["prev_close"] = nifty_df["close"].shift(1)
    nifty_df["nifty_change_points"] = nifty_df["close"] - nifty_df["prev_close"]
    nifty_df["nifty_change_pct"] = np.where(
        nifty_df["prev_close"] > 0,
        (nifty_df["close"] / nifty_df["prev_close"] - 1.0) * 100.0,
        np.nan,
    )
    nifty_rows = nifty_df.tail(days).copy()
    date_keys = [d.date().isoformat() for d in nifty_rows["date"]]

    stock_by_symbol: Dict[str, Dict[str, Dict[str, float]]] = {}
    # In cache-only mode, we intentionally bypass freshness checks and use what exists on disk.
    max_age_hours = 24 * 365 * 20
    for sym in weight_by_symbol.keys():
        sdf = _load_equity_history_cache(sym, max_age_hours=max_age_hours)
        if sdf.empty or "date" not in sdf.columns or "close" not in sdf.columns:
            continue
        sdf = sdf.copy()
        sdf["date"] = pd.to_datetime(sdf["date"], errors="coerce")
        sdf["close"] = pd.to_numeric(sdf["close"], errors="coerce")
        sdf["volume"] = pd.to_numeric(sdf.get("volume"), errors="coerce")
        sdf = sdf.dropna(subset=["date", "close"]).sort_values("date")
        sdf["prev_close"] = sdf["close"].shift(1)
        sdf["ret"] = np.where(
            sdf["prev_close"] > 0,
            (sdf["close"] / sdf["prev_close"]) - 1.0,
            np.nan,
        )
        per_date: Dict[str, Dict[str, float]] = {}
        for _, row in sdf.iterrows():
            d = row["date"]
            if pd.isna(d):
                continue
            key = pd.to_datetime(d).date().isoformat()
            per_date[key] = {
                "close": float(row["close"]) if pd.notna(row["close"]) else float("nan"),
                "ret": float(row["ret"]) if pd.notna(row["ret"]) else float("nan"),
                "volume": float(row["volume"]) if pd.notna(row["volume"]) else 0.0,
            }
        stock_by_symbol[sym] = per_date

    out_rows: List[Dict[str, Any]] = []
    for _, nrow in nifty_rows.iterrows():
        date_val = nrow["date"]
        dkey = pd.to_datetime(date_val).date().isoformat()
        nifty_close = float(nrow["close"]) if pd.notna(nrow["close"]) else None
        nifty_prev = float(nrow["prev_close"]) if pd.notna(nrow["prev_close"]) else None
        nifty_change_points = float(nrow["nifty_change_points"]) if pd.notna(nrow["nifty_change_points"]) else None
        nifty_change_pct = float(nrow["nifty_change_pct"]) if pd.notna(nrow["nifty_change_pct"]) else None

        entries: List[Dict[str, Any]] = []
        for sym, weight in weight_by_symbol.items():
            srow = (stock_by_symbol.get(sym) or {}).get(dkey)
            if not isinstance(srow, dict):
                continue
            stock_ret = srow.get("ret")
            ltp = srow.get("close")
            volume = srow.get("volume")
            if not isinstance(stock_ret, float) or not math.isfinite(stock_ret):
                continue
            contribution_points = (
                float(weight) * float(stock_ret) * float(nifty_prev)
                if nifty_prev is not None and math.isfinite(float(nifty_prev))
                else None
            )
            entries.append(
                {
                    "symbol": sym,
                    "weight_pct": _json_safe(weight * 100.0),
                    "ltp": _json_safe(ltp),
                    "change_pct": _json_safe(stock_ret * 100.0),
                    "volume": _json_safe(volume),
                    "contribution_points": _json_safe(contribution_points),
                }
            )

        entries = sorted(
            entries,
            key=lambda x: (
                -(float(x.get("contribution_points")) if isinstance(x.get("contribution_points"), (int, float)) else 0.0),
                str(x.get("symbol") or ""),
            ),
        )
        pos = [e for e in entries if isinstance(e.get("contribution_points"), (int, float)) and float(e["contribution_points"]) > 0]
        neg = [e for e in entries if isinstance(e.get("contribution_points"), (int, float)) and float(e["contribution_points"]) < 0]
        pos_total = float(sum(float(e["contribution_points"]) for e in pos)) if pos else 0.0
        neg_total = float(sum(float(e["contribution_points"]) for e in neg)) if neg else 0.0
        total = pos_total + neg_total

        out_rows.append(
            {
                "date": dkey,
                "nifty_close": _json_safe(nifty_close),
                "nifty_change_points": _json_safe(nifty_change_points),
                "nifty_change_pct": _json_safe(nifty_change_pct),
                "positive_count": len(pos),
                "negative_count": len(neg),
                "positive_total": _json_safe(pos_total),
                "negative_total": _json_safe(neg_total),
                "net_total": _json_safe(total),
                "constituents": entries,
            }
        )

    out_rows = [r for r in out_rows if str(r.get("date") or "") in set(date_keys)]
    out_rows = sorted(out_rows, key=lambda r: str(r.get("date") or ""))
    return {"rows": out_rows, "count": len(out_rows), "constituents": list(weight_by_symbol.keys())}




@app.get("/long-term/keymetrics")
def long_term_keymetrics(request: Request, symbol: Optional[str] = None):
    _get_credentials(request)
    universe = _load_long_term_universe()
    raw_cache = _load_indianapi_cache(max_age_days=3650)
    cache: Dict[str, Dict[str, Any]] = {}
    for k, v in raw_cache.items():
        cache[k] = {"saved_at": datetime.now(timezone.utc).isoformat(), "payload": v}

    if symbol:
        symbols = [str(symbol).upper()]
    else:
        symbols = [str(s).upper() for s in (universe.get("nifty100", []) + universe.get("midcap150", []))]

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        vendor = (cache.get(sym) or {}).get("payload") or {}
        km = vendor.get("keyMetrics") if isinstance(vendor, dict) else None
        if not isinstance(km, dict):
            continue
        for category, items in km.items():
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                rows.append(
                    {
                        "symbol": sym,
                        "category": category,
                        "displayName": it.get("displayName"),
                        "key": it.get("key"),
                        "value": it.get("value"),
                    }
                )
    return {"rows": rows, "count": len(rows)}


def _flatten_dict(obj: Any, prefix: str = "") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(_flatten_dict(v, path))
    elif isinstance(obj, list):
        for idx, v in enumerate(obj):
            path = f"{prefix}[{idx}]"
            rows.extend(_flatten_dict(v, path))
    else:
        rows.append({"path": prefix, "value": obj})
    return rows


@app.get("/long-term/raw-flat")
def long_term_raw_flat(request: Request, symbol: Optional[str] = None):
    _get_credentials(request)
    universe = _load_long_term_universe()
    raw_cache = _load_indianapi_cache(max_age_days=3650)
    cache: Dict[str, Dict[str, Any]] = {}
    for k, v in raw_cache.items():
        cache[k] = {"saved_at": datetime.now(timezone.utc).isoformat(), "payload": v}

    if symbol:
        symbols = [str(symbol).upper()]
    else:
        symbols = [str(s).upper() for s in (universe.get("nifty100", []) + universe.get("midcap150", []))]

    rows: List[Dict[str, Any]] = []
    for sym in symbols:
        vendor = (cache.get(sym) or {}).get("payload") or {}
        flat = _flatten_dict(vendor)
        for item in flat:
            rows.append({"symbol": sym, "path": item.get("path"), "value": item.get("value")})
    return {"rows": rows, "count": len(rows)}


@app.get("/long-term/raw-flat-live")
def long_term_raw_flat_live(request: Request, symbol: str):
    _get_credentials(request)
    vendor = _indianapi_get("/stock", {"name": symbol})
    flat = _flatten_dict(vendor)
    rows = [{"symbol": symbol, "path": item.get("path"), "value": item.get("value")} for item in flat]
    return {"rows": rows, "count": len(rows)}


@app.get("/long-term/fundamentals-cache")
def long_term_fundamentals_cache(request: Request, symbol: str):
    _get_credentials(request)
    raw_cache = _load_indianapi_cache(max_age_days=3650)
    entry = raw_cache.get(str(symbol).upper())
    if not entry:
        return {"symbol": symbol, "data": None}
    if isinstance(entry, dict) and "payload" in entry:
        return {"symbol": symbol, "data": entry.get("payload")}
    return {"symbol": symbol, "data": entry}


@app.post("/long-term/fundamentals-refresh")
def long_term_fundamentals_refresh(request: Request, symbol: str):
    _get_credentials(request)
    name = symbol
    if symbol.upper() == "RELIANCE":
        name = "Reliance"
    vendor = _indianapi_get("/stock", {"name": name})
    if vendor and isinstance(vendor, dict) and (
        "keyMetrics" in vendor or "financials" in vendor or "companyProfile" in vendor
    ):
        cache = _load_indianapi_cache(max_age_days=3650)
        cache[str(symbol).upper()] = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "payload": vendor,
        }
        _save_indianapi_cache(cache)
        return {"symbol": symbol, "status": "updated", "data": vendor}
    return {"symbol": symbol, "status": "no_data", "data": vendor}


@app.get("/long-term/indianapi-debug")
def long_term_indianapi_debug(name: str = "Reliance"):
    url = f"{_indianapi_base_url()}/stock"
    headers = _indianapi_headers(debug=True)
    params = {"name": name}
    try:
        resp = requests.get(url, params=params, headers=_indianapi_headers(), timeout=20)
        text = resp.text
        if len(text) > 1000:
            text = text[:1000] + "..."
        return {
            "url": url,
            "params": params,
            "debug_header": headers.get("X-Debug-Api-Key"),
            "status_code": resp.status_code,
            "response": text,
        }
    except Exception as exc:
        return {"url": url, "params": params, "debug_header": headers.get("X-Debug-Api-Key"), "error": str(exc)}


@app.get("/long-term/fundamentals-reliance")
def long_term_fundamentals_reliance():
    url = f"{_indianapi_base_url()}/stock"
    params = {"name": "Reliance"}
    try:
        resp = requests.get(url, params=params, headers=_indianapi_headers(debug=True), timeout=20)
        text = resp.text
        data = None
        try:
            data = resp.json()
        except Exception:
            data = None
        return {
            "url": url,
            "params": params,
            "status_code": resp.status_code,
            "debug_header": _indianapi_headers(debug=True).get("X-Debug-Api-Key"),
            "data": data,
            "response_raw": text[:2000] + "..." if len(text) > 2000 else text,
        }
    except Exception as exc:
        return {
            "url": url,
            "params": params,
            "debug_header": _indianapi_headers(debug=True).get("X-Debug-Api-Key"),
            "error": str(exc),
        }


@app.get("/equities/optimize")
def equities_optimize(
    request: Request,
    target_return: float = 0.12,
    lookback_days: int = 756,
    use_cache: bool = True,
):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    holdings = _fetch_holdings_with_cache(kite, use_cache=use_cache)
    equities = [p for p in holdings if _is_equity_cash(p)] if holdings else []
    if not equities:
        return {"status": "no_equities", "weights": [], "kpis": {}}

    to_date = datetime.now().date()
    from_date = to_date - timedelta(days=max(lookback_days, 60))

    frames = []
    symbols = []
    current_values = []
    for pos in equities:
        symbol = pos.get("tradingsymbol") or pos.get("symbol") or pos.get("instrument") or "—"
        token = pos.get("instrument_token")
        qty = float(pos.get("quantity") or pos.get("qty") or 0.0)
        ltp = float(pos.get("last_price") or pos.get("ltp") or pos.get("mark_price") or 0.0)
        if not token or qty == 0 or ltp <= 0:
            continue
        df = _fetch_equity_history_cached(kite, str(symbol), int(token), from_date, to_date)
        if df.empty or "close" not in df.columns:
            continue
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if close.size < 60:
            continue
        rets = np.log(close / close.shift(1)).dropna()
        if rets.size < 40:
            continue
        s = pd.DataFrame({"date": df.loc[rets.index, "date"].values, str(symbol): rets.values})
        frames.append(s)
        symbols.append(str(symbol))
        current_values.append(ltp * qty)

    if not frames or len(symbols) < 2:
        return {"status": "insufficient_history", "weights": [], "kpis": {}}

    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge(merged, f, on="date", how="inner")
    merged = merged.dropna()
    if merged.shape[0] < 40:
        return {"status": "insufficient_overlap", "weights": [], "kpis": {}}

    ret_mat = merged[symbols].to_numpy(dtype=float)
    mu_daily = np.mean(ret_mat, axis=0)
    cov_daily = np.cov(ret_mat, rowvar=False)
    mu = mu_daily * TRADING_DAYS_PER_YEAR
    cov = cov_daily * TRADING_DAYS_PER_YEAR

    inv = np.linalg.pinv(cov)
    ones = np.ones(len(symbols))
    A = float(ones @ inv @ ones)
    B = float(ones @ inv @ mu)
    C = float(mu @ inv @ mu)
    D = A * C - B * B
    if abs(D) < 1e-12:
        return {"status": "singular_cov", "weights": [], "kpis": {}}

    R = float(target_return)
    lam = (C - B * R) / D
    gam = (A * R - B) / D
    w = lam * (inv @ ones) + gam * (inv @ mu)

    exp_return = float(mu @ w)
    exp_vol = float(math.sqrt(max(w @ cov @ w, 0.0)))
    total_value = float(np.sum(current_values))
    curr_weights = [
        (float(val) / total_value * 100.0) if total_value > 0 else 0.0 for val in current_values
    ]

    weights = [
        {
            "symbol": symbols[i],
            "weight": float(w[i]),
            "weight_pct": float(w[i] * 100.0),
            "current_weight_pct": curr_weights[i],
        }
        for i in range(len(symbols))
    ]

    return {
        "status": "ok",
        "config": {
            "target_return": float(target_return),
            "lookback_days": int(lookback_days),
        },
        "kpis": {
            "expected_return": exp_return,
            "expected_vol": exp_vol,
        },
        "weights": weights,
    }


@app.get("/risk-buckets/portfolio")
def risk_buckets_portfolio(request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    options_df = _filter_latest_snapshot(options_df, "date")
    nifty_df = _load_cache("nifty_ohlcv")
    if options_df.empty or nifty_df.empty:
        raise HTTPException(status_code=404, detail="Options/NIFTY cache missing")

    _prepare_rb_state(options_df, nifty_df)

    positions = _fetch_positions_with_cache(kite)
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]

    margins = kite.margins()
    equity = margins.get("equity", {})
    margin_available = float(equity.get("available", {}).get("live_balance", 0.0))
    margin_used = float(equity.get("utilised", {}).get("debits", 0.0))
    total_capital = float(margin_available + margin_used)
    if total_capital <= 0:
        total_capital = 1_750_000.0
    margin_used_pct = (margin_used / total_capital * 100.0) if total_capital > 0 else None

    trades = build_trades_using_existing_grouping(enriched)
    saved_groups = _load_saved_groups()
    if saved_groups:
        trades = build_trades_using_existing_grouping(enriched)
        trades += build_trades_from_saved_groups(enriched, saved_groups)

    trades_df, _ = compute_trade_risk(
        trades,
        total_capital,
        lookback_days=504,
        spot_override=current_spot,
    )
    settings = _load_risk_bucket_settings()
    allocations = {"low": settings["alloc_low"], "med": settings["alloc_med"], "high": settings["alloc_high"]}
    thresholds = {"low": settings["trade_low_max"], "med": settings["trade_med_max"]}
    manual_bucket_map = _load_manual_bucket_overrides()
    trades_df = assign_buckets(trades_df, total_capital, allocations, thresholds, zone_map={}, manual_bucket_map=manual_bucket_map)

    agg = aggregate_portfolio(trades_df)

    advanced_sim = run_advanced_forward_risk(
        positions=positions,
        options_df=options_df,
        spot_history_df=nifty_df,
        spot=current_spot,
        config={
            "horizon_days": int(settings.get("sim_days", 10)),
            "n_paths": int(settings.get("sim_paths", 2000)),
            "dt": 1.0 / 252.0,
            "seed": 42,
            "risk_free_rate": 0.06,
            "dividend_yield": 0.01,
            "daily_loss_limit": None,
            "total_loss_limit": (float(settings.get("portfolio_es_limit", 4.0)) / 100.0) * float(total_capital) if total_capital else None,
            "engines": ("fhs", "garch", "egarch", "gjr", "heston", "bates"),
            "pnl_modes": ("greeks", "repricing"),
            "iv_rules": ("flat", "surface"),
            "repricing_models": ("bs76",),
            "use_evt_overlay": True,
            "simulate_surface": True,
            "repricing_anchor": "market_t0",
        },
    )
    _log_advanced_simulation_snapshot(
        advanced_sim=advanced_sim,
        current_spot=current_spot,
        positions_count=len(positions),
    )

    iv_percentile = 35
    iv_regime, _ = ra_tab.get_iv_regime(iv_percentile)
    greeks = calculate_portfolio_greeks(enriched)
    capital_lakhs = total_capital / 100000.0 if total_capital else 0.0
    theta_norm = abs(greeks.get("net_theta", 0.0)) / capital_lakhs if capital_lakhs else 0.0
    gamma_norm = greeks.get("net_gamma", 0.0) / capital_lakhs if capital_lakhs else 0.0
    vega_norm = greeks.get("net_vega", 0.0) / capital_lakhs if capital_lakhs else 0.0
    zone_num, zone_name, zone_color, zone_message = ra_tab.classify_zone(
        theta_norm, gamma_norm, vega_norm, iv_regime
    )

    top_underlying = (
        agg["by_underlying"].head(10).reset_index().rename(columns={"index": "underlying", "trade_es99_inr": "es99"})
        if not agg["by_underlying"].empty
        else pd.DataFrame(columns=["underlying", "es99"])
    )
    by_bucket = agg["by_bucket"].reset_index().rename(columns={"bucket": "bucket", "trade_es99_inr": "es99"})
    by_week = (
        trades_df.groupby(["underlying", "week_id", "option_side"])["trade_es99_inr"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    # Backward-compat summary payloads are now sourced from advanced simulation defaults.
    engines_block = (advanced_sim or {}).get("engines", {}) if isinstance(advanced_sim, dict) else {}
    engine_candidates = list(engines_block.keys()) if isinstance(engines_block, dict) else []
    engine_name = next((k for k in engine_candidates if str(k).startswith("gbm|flat")), None)
    if not engine_name:
        engine_name = next((k for k in engine_candidates if str(k).startswith("gbm|surface")), None)
    if not engine_name:
        engine_name = next((k for k in engine_candidates if str(k).startswith("gbm|")), None)
    if not engine_name:
        engine_name = next(iter(engines_block), None)
    mode_candidates = (engines_block.get(engine_name, {}).get("modes", {}) or {}) if engine_name else {}
    mode_name = "repricing_bs"
    if mode_name not in mode_candidates:
        mode_name = next((m for m in mode_candidates.keys() if str(m).startswith("repricing")), None)
    if not mode_name:
        mode_name = next(iter(mode_candidates), "greeks")
    selected_mode = (
        (engines_block.get(engine_name, {}).get("modes", {}) or {}).get(mode_name, {})
        if engine_name
        else {}
    )
    selected_kpis = (selected_mode or {}).get("kpis", {}) if isinstance(selected_mode, dict) else {}
    selected_fan = (selected_kpis.get("fan", {}) or {}) if isinstance(selected_kpis, dict) else {}
    terminal_sample = (selected_mode.get("terminal_pnl_sample", []) or []) if isinstance(selected_mode, dict) else []

    fan_rows = []
    p50 = selected_fan.get("p50", []) if isinstance(selected_fan, dict) else []
    p10 = selected_fan.get("p10", []) if isinstance(selected_fan, dict) else []
    p1 = selected_fan.get("p1", []) if isinstance(selected_fan, dict) else []
    for i in range(min(len(p50), len(p10), len(p1))):
        fan_rows.append({"Day": i + 1, "P50": float(p50[i]), "P10": float(p10[i]), "P1": float(p1[i])})
    distribution = None
    if terminal_sample:
        final_pnl = np.asarray(terminal_sample, dtype=float)
        distribution = {
            "final_pnl": final_pnl.tolist(),
            "mean": float(np.mean(final_pnl)),
            "median": float(np.median(final_pnl)),
            "p5": float(np.percentile(final_pnl, 5)),
            "p1": float(np.percentile(final_pnl, 1)),
        }
    sim_summary = None
    if selected_kpis:
        quantiles = selected_kpis.get("quantiles", {}) if isinstance(selected_kpis, dict) else {}
        sim_summary = {
            "mean": float(selected_kpis.get("mean", 0.0)),
            "median": float(selected_kpis.get("median", 0.0)),
            "p95": float(quantiles.get("p95", np.percentile(np.asarray(terminal_sample, dtype=float), 95) if terminal_sample else 0.0)),
            "p5": float(quantiles.get("p5", 0.0)),
            "p1": float(quantiles.get("p1", 0.0)),
            "prob_loss": float(selected_kpis.get("prob_loss", 0.0)),
            "prob_breach": float(selected_kpis.get("prob_breach_total", 0.0)),
        }
    return {
        "sim_summary": sim_summary,
        "sim_fan": fan_rows,
        "sim_distribution": distribution,
        "advanced_simulation": advanced_sim,
        "account_size": total_capital,
        "margin_used": margin_used,
        "portfolio_es99_inr": float(agg["portfolio_es99_inr"]),
        "portfolio_es99_pct": (float(agg["portfolio_es99_inr"]) / total_capital * 100) if total_capital else 0.0,
        "margin_used_pct": margin_used_pct,
        "top_underlying": top_underlying.to_dict(orient="records"),
        "by_bucket": by_bucket.to_dict(orient="records"),
        "by_week": by_week.to_dict(orient="records"),
        "zone": {
            "iv_regime": iv_regime,
            "theta_norm": theta_norm,
            "gamma_norm": gamma_norm,
            "vega_norm": vega_norm,
            "zone_num": zone_num,
            "zone_name": zone_name,
            "zone_message": zone_message,
            "zone_color": zone_color,
        },
    }


@app.post("/risk-buckets/portfolio/simulate")
def risk_buckets_portfolio_simulate(payload: PortfolioSimPayload, request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    options_df = _filter_latest_snapshot(options_df, "date")
    nifty_df = _load_cache("nifty_ohlcv")
    if options_df.empty or nifty_df.empty:
        raise HTTPException(status_code=404, detail="Options/NIFTY cache missing")

    _prepare_rb_state(options_df, nifty_df)

    positions = _fetch_positions_with_cache(kite)
    if payload.virtual_positions:
        positions = payload.virtual_positions
    else:
        ids = {str(i) for i in (payload.position_ids or [])}
        if ids:
            positions = [p for p in positions if _position_id(p) in ids]

    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]

    margins = kite.margins()
    equity = margins.get("equity", {})
    margin_available = float(equity.get("available", {}).get("live_balance", 0.0))
    margin_used = float(equity.get("utilised", {}).get("debits", 0.0))
    total_capital = float(margin_available + margin_used)
    if total_capital <= 0:
        total_capital = 1_750_000.0
    margin_used_pct = (margin_used / total_capital * 100.0) if total_capital > 0 else None

    trades = build_trades_using_existing_grouping(enriched)
    saved_groups = _load_saved_groups()
    if saved_groups:
        trades = build_trades_using_existing_grouping(enriched)
        trades += build_trades_from_saved_groups(enriched, saved_groups)

    trades_df, _ = compute_trade_risk(
        trades,
        total_capital,
        lookback_days=504,
        spot_override=current_spot,
    )
    settings = _load_risk_bucket_settings()
    allocations = {"low": settings["alloc_low"], "med": settings["alloc_med"], "high": settings["alloc_high"]}
    thresholds = {"low": settings["trade_low_max"], "med": settings["trade_med_max"]}
    manual_bucket_map = _load_manual_bucket_overrides()
    trades_df = assign_buckets(trades_df, total_capital, allocations, thresholds, zone_map={}, manual_bucket_map=manual_bucket_map)

    agg = aggregate_portfolio(trades_df)

    advanced_sim = run_advanced_forward_risk(
        positions=positions,
        options_df=options_df,
        spot_history_df=nifty_df,
        spot=current_spot,
        config={
            "horizon_days": int(settings.get("sim_days", 10)),
            "n_paths": int(settings.get("sim_paths", 2000)),
            "dt": 1.0 / 252.0,
            "seed": 42,
            "risk_free_rate": 0.06,
            "dividend_yield": 0.01,
            "daily_loss_limit": None,
            "total_loss_limit": (float(settings.get("portfolio_es_limit", 4.0)) / 100.0) * float(total_capital) if total_capital else None,
            "engines": ("fhs", "garch", "egarch", "gjr", "heston", "bates"),
            "pnl_modes": ("greeks", "repricing"),
            "iv_rules": ("flat", "surface"),
            "repricing_models": ("bs76",),
            "use_evt_overlay": True,
            "simulate_surface": True,
            "repricing_anchor": "market_t0",
        },
    )
    _log_advanced_simulation_snapshot(
        advanced_sim=advanced_sim,
        current_spot=current_spot,
        positions_count=len(positions),
    )

    iv_percentile = 35
    iv_regime, _ = ra_tab.get_iv_regime(iv_percentile)
    greeks = calculate_portfolio_greeks(enriched)
    capital_lakhs = total_capital / 100000.0 if total_capital else 0.0
    theta_norm = abs(greeks.get("net_theta", 0.0)) / capital_lakhs if capital_lakhs else 0.0
    gamma_norm = greeks.get("net_gamma", 0.0) / capital_lakhs if capital_lakhs else 0.0
    vega_norm = greeks.get("net_vega", 0.0) / capital_lakhs if capital_lakhs else 0.0
    zone_num, zone_name, zone_color, zone_message = ra_tab.classify_zone(
        theta_norm, gamma_norm, vega_norm, iv_regime
    )

    top_underlying = (
        agg["by_underlying"].head(10).reset_index().rename(columns={"index": "underlying", "trade_es99_inr": "es99"})
        if not agg["by_underlying"].empty
        else pd.DataFrame(columns=["underlying", "es99"])
    )
    by_bucket = agg["by_bucket"].reset_index().rename(columns={"bucket": "bucket", "trade_es99_inr": "es99"})
    by_week = (
        trades_df.groupby(["underlying", "week_id", "option_side"])["trade_es99_inr"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    engines_block = (advanced_sim or {}).get("engines", {}) if isinstance(advanced_sim, dict) else {}
    engine_candidates = list(engines_block.keys()) if isinstance(engines_block, dict) else []
    engine_name = next((k for k in engine_candidates if str(k).startswith("gbm|flat")), None)
    if not engine_name:
        engine_name = next((k for k in engine_candidates if str(k).startswith("gbm|surface")), None)
    if not engine_name:
        engine_name = next((k for k in engine_candidates if str(k).startswith("gbm|")), None)
    if not engine_name:
        engine_name = next(iter(engines_block), None)
    mode_candidates = (engines_block.get(engine_name, {}).get("modes", {}) or {}) if engine_name else {}
    mode_name = "repricing_bs"
    if mode_name not in mode_candidates:
        mode_name = next((m for m in mode_candidates.keys() if str(m).startswith("repricing")), None)
    if not mode_name:
        mode_name = next(iter(mode_candidates), "greeks")
    selected_mode = (
        (engines_block.get(engine_name, {}).get("modes", {}) or {}).get(mode_name, {})
        if engine_name
        else {}
    )
    selected_kpis = (selected_mode or {}).get("kpis", {}) if isinstance(selected_mode, dict) else {}
    selected_fan = (selected_kpis.get("fan", {}) or {}) if isinstance(selected_kpis, dict) else {}
    terminal_sample = (selected_mode.get("terminal_pnl_sample", []) or []) if isinstance(selected_mode, dict) else []

    fan_rows = []
    p50 = selected_fan.get("p50", []) if isinstance(selected_fan, dict) else []
    p10 = selected_fan.get("p10", []) if isinstance(selected_fan, dict) else []
    p1 = selected_fan.get("p1", []) if isinstance(selected_fan, dict) else []
    for i in range(min(len(p50), len(p10), len(p1))):
        fan_rows.append({"Day": i + 1, "P50": float(p50[i]), "P10": float(p10[i]), "P1": float(p1[i])})
    distribution = None
    if terminal_sample:
        final_pnl = np.asarray(terminal_sample, dtype=float)
        distribution = {
            "final_pnl": final_pnl.tolist(),
            "mean": float(np.mean(final_pnl)),
            "median": float(np.median(final_pnl)),
            "p5": float(np.percentile(final_pnl, 5)),
            "p1": float(np.percentile(final_pnl, 1)),
        }
    sim_summary = None
    if selected_kpis:
        quantiles = selected_kpis.get("quantiles", {}) if isinstance(selected_kpis, dict) else {}
        sim_summary = {
            "mean": float(selected_kpis.get("mean", 0.0)),
            "median": float(selected_kpis.get("median", 0.0)),
            "p95": float(quantiles.get("p95", np.percentile(np.asarray(terminal_sample, dtype=float), 95) if terminal_sample else 0.0)),
            "p5": float(quantiles.get("p5", 0.0)),
            "p1": float(quantiles.get("p1", 0.0)),
            "prob_loss": float(selected_kpis.get("prob_loss", 0.0)),
            "prob_breach": float(selected_kpis.get("prob_breach_total", 0.0)),
        }
    return {
        "sim_summary": sim_summary,
        "sim_fan": fan_rows,
        "sim_distribution": distribution,
        "advanced_simulation": advanced_sim,
        "account_size": total_capital,
        "margin_used": margin_used,
        "portfolio_es99_inr": float(agg["portfolio_es99_inr"]),
        "portfolio_es99_pct": (float(agg["portfolio_es99_inr"]) / total_capital * 100) if total_capital else 0.0,
        "margin_used_pct": margin_used_pct,
        "top_underlying": top_underlying.to_dict(orient="records"),
        "by_bucket": by_bucket.to_dict(orient="records"),
        "by_week": by_week.to_dict(orient="records"),
        "zone": {
            "iv_regime": iv_regime,
            "theta_norm": theta_norm,
            "gamma_norm": gamma_norm,
            "vega_norm": vega_norm,
            "zone_num": zone_num,
            "zone_name": zone_name,
            "zone_message": zone_message,
            "zone_color": zone_color,
        },
    }


@app.get("/risk-buckets/artifact/{name}")
def risk_buckets_artifact(name: str):
    allowed = {"overlay_cdf.png", "overlay_tail.png"}
    if name not in allowed:
        raise HTTPException(status_code=404, detail="Artifact not found")
    path = ROOT / "artifacts" / name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not available")
    return FileResponse(path)


@app.get("/risk-buckets/buckets")
def risk_buckets_bucket(request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    nifty_df = _load_cache("nifty_ohlcv")
    if options_df.empty or nifty_df.empty:
        raise HTTPException(status_code=404, detail="Options/NIFTY cache missing")

    _prepare_rb_state(options_df, nifty_df)
    positions = _fetch_positions_with_cache(kite)
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]

    total_capital = float(
        (kite.margins().get("equity", {}).get("available", {}).get("live_balance", 0.0))
        + (kite.margins().get("equity", {}).get("utilised", {}).get("debits", 0.0))
    )
    if total_capital <= 0:
        total_capital = 1_750_000.0

    trades = build_trades_using_existing_grouping(enriched)
    trades_df, _ = compute_trade_risk(
        trades,
        total_capital,
        lookback_days=504,
        spot_override=current_spot,
    )
    settings = _load_risk_bucket_settings()
    allocations = {"low": settings["alloc_low"], "med": settings["alloc_med"], "high": settings["alloc_high"]}
    thresholds = {"low": settings["trade_low_max"], "med": settings["trade_med_max"]}
    manual_bucket_map = _load_manual_bucket_overrides()
    trades_df = assign_buckets(trades_df, total_capital, allocations, thresholds, zone_map={}, manual_bucket_map=manual_bucket_map)
    bucket_limits = {
        "low": settings["bucket_es_limit_low"],
        "med": settings["bucket_es_limit_med"],
        "high": settings["bucket_es_limit_high"],
    }
    bucket_df = aggregate_buckets(trades_df, total_capital, allocations, bucket_limits)
    if "expected_pnl_inr" in trades_df.columns:
        expected_by_bucket = trades_df.groupby("bucket")["expected_pnl_inr"].sum()
        bucket_df["bucket_expected_pnl_inr"] = bucket_df["bucket"].map(expected_by_bucket).fillna(0.0)

    sim_cfg = SimulationConfig(
        horizon_days=int(settings.get("sim_days", 10)),
        paths=int(settings.get("sim_paths", 2000)),
        iv_mode=settings.get("iv_mode", "IV Flat"),
        iv_shock=float(settings.get("iv_shock", 2.0)),
        include_spot_shocks=True,
    )

    bucket_sims: Dict[str, Any] = {}
    for bucket in ["Low", "Med", "High"]:
        bucket_positions = []
        for _, row in trades_df[trades_df["bucket"] == bucket].iterrows():
            bucket_positions.extend(row["legs_detail"])
        bucket_sims[bucket] = simulate_forward_pnl(
            bucket_positions,
            sim_cfg,
            limit_pct=bucket_limits[bucket.lower()],
            limit_base=total_capital * allocations[bucket.lower()] / 100.0,
        )

    return {
        "bucket_rows": bucket_df.to_dict(orient="records"),
        "bucket_sims": {
            k: (v["summary"] if v else None) for k, v in bucket_sims.items()
        },
    }


@app.get("/risk-buckets/trades")
def risk_buckets_trades(request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    nifty_df = _load_cache("nifty_ohlcv")
    if options_df.empty or nifty_df.empty:
        raise HTTPException(status_code=404, detail="Options/NIFTY cache missing")

    _prepare_rb_state(options_df, nifty_df)
    positions = _fetch_positions_with_cache(kite)
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]
    total_capital = 1_750_000.0
    trades = build_trades_using_existing_grouping(enriched)
    trades_df, _ = compute_trade_risk(
        trades,
        total_capital,
        lookback_days=504,
        spot_override=current_spot,
    )
    settings = _load_risk_bucket_settings()
    allocations = {"low": settings["alloc_low"], "med": settings["alloc_med"], "high": settings["alloc_high"]}
    thresholds = {"low": settings["trade_low_max"], "med": settings["trade_med_max"]}
    manual_bucket_map = _load_manual_bucket_overrides()
    trades_df = assign_buckets(trades_df, total_capital, allocations, thresholds, zone_map={}, manual_bucket_map=manual_bucket_map)
    return {"rows": trades_df.to_dict(orient="records")}


@app.get("/risk-buckets/trade-detail")
def risk_buckets_trade_detail(request: Request, trade_id: str):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    nifty_df = _load_cache("nifty_ohlcv")
    _prepare_rb_state(options_df, nifty_df)
    positions = _fetch_positions_with_cache(kite)
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]
    total_capital = 1_750_000.0
    trades = build_trades_using_existing_grouping(enriched)
    trades_df, legs_by_trade = compute_trade_risk(
        trades,
        total_capital,
        lookback_days=504,
        spot_override=current_spot,
    )
    legs = legs_by_trade.get(trade_id, [])
    if not legs:
        raise HTTPException(status_code=404, detail="trade_id not found")
    scenarios = ra_tab.get_weighted_scenarios(ra_tab.get_iv_regime(35)[0])
    scenario_df = _build_trade_scenario_table(
        legs, scenarios, current_spot, None, None, total_capital
    )
    payoff_df = _compute_trade_payoff_curve(
        legs, current_spot, None, None, include_unrealized_pnl=True
    )
    expiry_df = _compute_expiry_payoff_curve(
        legs, current_spot, include_unrealized_pnl=True
    )
    iv_debug = []
    for leg in legs:
        iv_val = leg.get("implied_vol") or leg.get("iv") or leg.get("implied_volatility")
        iv_debug.append(
            {
                "symbol": leg.get("tradingsymbol"),
                "qty": leg.get("quantity"),
                "strike": leg.get("strike") or leg.get("strike_price"),
                "type": leg.get("option_type") or leg.get("instrument_type"),
                "price": leg.get("last_price") or leg.get("ltp") or leg.get("close"),
                "expiry": leg.get("expiry"),
                "dte": leg.get("dte"),
                "tte": leg.get("time_to_expiry"),
                "iv": iv_val,
            }
        )
    dist_rows = []
    if not scenario_df.empty:
        dist_df = scenario_df.copy()
        dist_df["pnl_inr"] = (
            dist_df["Repriced P&L (₹)"].astype(str).str.replace("₹", "").str.replace(",", "")
        )
        dist_df["pnl_inr"] = pd.to_numeric(dist_df["pnl_inr"], errors="coerce").fillna(0.0)
        dist_df["prob_pct"] = (
            dist_df["Probability"].astype(str).str.replace("%", "")
        )
        dist_df["prob_pct"] = pd.to_numeric(dist_df["prob_pct"], errors="coerce").fillna(0.0)
        dist_df = dist_df.sort_values("pnl_inr")
        dist_df["cum_prob"] = dist_df["prob_pct"].cumsum()
        dist_rows = [
            {
                "scenario": row.get("Scenario"),
                "pnl_inr": float(row.get("pnl_inr", 0.0)),
                "prob_pct": float(row.get("prob_pct", 0.0)),
                "cum_prob": float(row.get("cum_prob", 0.0)),
            }
            for _, row in dist_df.iterrows()
        ]
    return {
        "trade_id": trade_id,
        "scenario_rows": scenario_df.to_dict(orient="records"),
        "payoff": payoff_df.to_dict(orient="records"),
        "expiry_payoff": expiry_df.to_dict(orient="records"),
        "legs": legs,
        "iv_debug": iv_debug,
        "scenario_dist": dist_rows,
    }


@app.get("/risk-buckets/trade-details")
def risk_buckets_trade_details(request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    nifty_df = _load_cache("nifty_ohlcv")
    _prepare_rb_state(options_df, nifty_df)
    positions = _fetch_positions_with_cache(kite)
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]
    total_capital = 1_750_000.0
    trades = build_trades_using_existing_grouping(enriched)
    trades_df, legs_by_trade = compute_trade_risk(
        trades, total_capital, lookback_days=504, spot_override=current_spot
    )

    scenarios = ra_tab.get_weighted_scenarios(ra_tab.get_iv_regime(35)[0])
    details: Dict[str, Any] = {}
    for trade_id, legs in legs_by_trade.items():
        scenario_df = _build_trade_scenario_table(
            legs, scenarios, current_spot, None, None, total_capital
        )
        payoff_df = _compute_trade_payoff_curve(
            legs, current_spot, None, None, include_unrealized_pnl=True
        )
        expiry_df = _compute_expiry_payoff_curve(
            legs, current_spot, include_unrealized_pnl=True
        )
        iv_debug = []
        for leg in legs:
            iv_val = leg.get("implied_vol") or leg.get("iv") or leg.get("implied_volatility")
            iv_debug.append(
                {
                    "symbol": leg.get("tradingsymbol"),
                    "qty": leg.get("quantity"),
                    "strike": leg.get("strike") or leg.get("strike_price"),
                    "type": leg.get("option_type") or leg.get("instrument_type"),
                    "price": leg.get("last_price") or leg.get("ltp") or leg.get("close"),
                    "expiry": leg.get("expiry"),
                    "dte": leg.get("dte"),
                    "tte": leg.get("time_to_expiry"),
                    "iv": iv_val,
                }
            )
        dist_rows = []
        if not scenario_df.empty:
            dist_df = scenario_df.copy()
            dist_df["pnl_inr"] = (
                dist_df["Repriced P&L (₹)"].astype(str).str.replace("₹", "").str.replace(",", "")
            )
            dist_df["pnl_inr"] = pd.to_numeric(dist_df["pnl_inr"], errors="coerce").fillna(0.0)
            dist_df["prob_pct"] = (
                dist_df["Probability"].astype(str).str.replace("%", "")
            )
            dist_df["prob_pct"] = pd.to_numeric(dist_df["prob_pct"], errors="coerce").fillna(0.0)
            dist_df = dist_df.sort_values("pnl_inr")
            dist_df["cum_prob"] = dist_df["prob_pct"].cumsum()
            dist_rows = [
                {
                    "scenario": row.get("Scenario"),
                    "pnl_inr": float(row.get("pnl_inr", 0.0)),
                    "prob_pct": float(row.get("prob_pct", 0.0)),
                    "cum_prob": float(row.get("cum_prob", 0.0)),
                }
                for _, row in dist_df.iterrows()
            ]
        details[trade_id] = {
            "trade_id": trade_id,
            "scenario_rows": scenario_df.to_dict(orient="records"),
            "payoff": payoff_df.to_dict(orient="records"),
            "expiry_payoff": expiry_df.to_dict(orient="records"),
            "legs": legs,
            "iv_debug": iv_debug,
            "scenario_dist": dist_rows,
        }

    return {"details": details, "trade_ids": list(details.keys())}


@app.get("/risk-buckets/meta")
def risk_buckets_meta():
    iv_regime, _ = ra_tab.get_iv_regime(35)
    zone_rules = _zone_rules_table(iv_regime).to_dict(orient="records")
    bucket_meta = [
        {"Bucket": "A", "Label": "Calm", "Definition": "|Return| ≤ 0.5% and |IV proxy| ≤ 1", "Notes": "Low-vol day"},
        {"Bucket": "B", "Label": "Normal", "Definition": "|Return| ≤ 1.0% and |IV proxy| ≤ 2", "Notes": "Routine"},
        {"Bucket": "C", "Label": "Elevated", "Definition": "|Return| ≤ 1.5% and |IV proxy| ≤ 3", "Notes": "Elevated range"},
        {"Bucket": "D", "Label": "Stress", "Definition": "|Return| ≤ 2.5% and |IV proxy| ≤ 5", "Notes": "Large range"},
        {"Bucket": "E", "Label": "Gap / Tail", "Definition": "|Return| > 2.5% or |IV proxy| > 5", "Notes": "Tail event"},
    ]
    return {
        "zone_rules": zone_rules,
        "bucket_meta": bucket_meta,
        "iv_regime": iv_regime,
        "allocations": {"low": 50.0, "med": 30.0, "high": 20.0},
        "limits": {"portfolio_es99": 4.0, "bucket_low": 2.0, "bucket_med": 3.0, "bucket_high": 5.0},
        "gate_thresholds": {
            "gate_tail_ratio_watch": 30.0,
            "gate_tail_ratio_fail": 60.0,
            "gate_prob_loss_watch": 40.0,
            "gate_prob_loss_fail": 50.0,
            "gate_p1_breach_fail_days": 2,
            "gate_portfolio_p10_fail_days": 4,
            "gate_bucket_p10_fail_days_med": 6,
            "gate_bucket_p10_fail_days_high": 3,
        },
    }


@app.get("/risk-buckets/settings")
def risk_buckets_settings(request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    nifty_df = _load_cache("nifty_ohlcv")
    if options_df.empty or nifty_df.empty:
        raise HTTPException(status_code=404, detail="Options/NIFTY cache missing")

    positions = _fetch_positions_with_cache(kite)
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]

    saved_groups = _load_saved_groups()
    grouped_leg_ids = {leg_id for group in saved_groups for leg_id in group.get("legs", [])}

    trades = build_trades_using_existing_grouping(enriched)
    if saved_groups:
        trades += build_trades_from_saved_groups(enriched, saved_groups)

    total_capital = float(
        (kite.margins().get("equity", {}).get("available", {}).get("live_balance", 0.0))
        + (kite.margins().get("equity", {}).get("utilised", {}).get("debits", 0.0))
    )
    if total_capital <= 0:
        total_capital = 1_750_000.0

    trades_df, _ = compute_trade_risk(
        trades,
        total_capital,
        lookback_days=504,
        spot_override=current_spot,
    )
    allocations = {"low": 50.0, "med": 30.0, "high": 20.0}
    thresholds = {"low": 1.0, "med": 2.0}
    manual_bucket_map = _load_manual_bucket_overrides()
    trades_df = assign_buckets(trades_df, total_capital, allocations, thresholds, zone_map={}, manual_bucket_map=manual_bucket_map)

    grouped_trades = trades_df[trades_df["is_group_trade"]].copy()
    grouped_rows = []
    for _, row in grouped_trades.iterrows():
        trade_id = str(row.get("trade_id"))
        grouped_rows.append(
            {
                "trade_id": trade_id,
                "trade_es99_inr": float(row.get("trade_es99_inr", 0.0)),
                "expected_pnl_inr": float(row.get("expected_pnl_inr", 0.0)),
                "bucket": row.get("bucket"),
                "manual_bucket": manual_bucket_map.get(trade_id, "Auto"),
            }
        )

    positions_rows = []
    for idx, pos in enumerate(enriched):
        positions_rows.append(
            {
                "id": str(idx),
                "label": _position_label(pos, idx),
                "is_grouped": str(idx) in grouped_leg_ids,
            }
        )

    groups_rows = []
    for idx, group in enumerate(saved_groups):
        groups_rows.append(
            {
                "index": idx,
                "name": group.get("name") or f"Group {idx + 1}",
                "legs": group.get("legs", []),
            }
        )

    return {
        "grouped_trades": grouped_rows,
        "positions": positions_rows,
        "groups": groups_rows,
        "overrides": manual_bucket_map,
    }


@app.get("/risk-buckets/settings/config")
def risk_buckets_settings_config():
    return {"settings": _load_risk_bucket_settings()}


@app.post("/risk-buckets/settings/config")
def risk_buckets_settings_update(payload: RiskBucketSettingsPayload):
    current = _load_risk_bucket_settings()
    merged = {**current, **payload.settings}
    _save_risk_bucket_settings(merged)
    return {"status": "ok", "settings": merged}


@app.post("/risk-buckets/settings/overrides")
def risk_buckets_update_override(payload: BucketOverridePayload):
    overrides = _load_manual_bucket_overrides()
    bucket = payload.bucket.strip()
    if bucket.lower() == "auto" or bucket == "":
        overrides.pop(str(payload.trade_id), None)
    else:
        overrides[str(payload.trade_id)] = bucket
    _save_manual_bucket_overrides(overrides)
    return {"status": "ok", "overrides": overrides}


@app.post("/risk-buckets/settings/groups")
def risk_buckets_add_group(payload: TradeGroupPayload):
    groups = _load_saved_groups()
    legs = [str(l) for l in payload.legs if str(l).strip() != ""]
    if not legs:
        raise HTTPException(status_code=400, detail="No legs provided")
    groups.append({"name": payload.name or "Manual Trade", "legs": legs})
    try:
        from views.tabs.risk_buckets_tab import _save_saved_groups as _save_groups
        _save_groups(groups)
    except Exception:
        pass
    return {"status": "ok", "groups": groups}


@app.delete("/risk-buckets/settings/groups/{index}")
def risk_buckets_remove_group(index: int):
    groups = _load_saved_groups()
    if index < 0 or index >= len(groups):
        raise HTTPException(status_code=404, detail="Group not found")
    groups = [g for i, g in enumerate(groups) if i != index]
    try:
        from views.tabs.risk_buckets_tab import _save_saved_groups as _save_groups
        _save_groups(groups)
    except Exception:
        pass
    return {"status": "ok", "groups": groups}


@app.delete("/risk-buckets/settings/groups")
def risk_buckets_clear_groups():
    try:
        from views.tabs.risk_buckets_tab import _save_saved_groups as _save_groups
        _save_groups([])
    except Exception:
        pass
    return {"status": "ok", "groups": []}


@app.get("/risk-buckets/history")
def risk_buckets_history():
    history_df = _build_bucket_history(lookback=504)
    if history_df is None or history_df.empty:
        return {"drift_counts": [], "ohlc": []}
    drift_pct = history_df["returnPct"] * 100
    bins = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = [
        "-10 to -9", "-9 to -8", "-8 to -7", "-7 to -6", "-6 to -5",
        "-5 to -4", "-4 to -3", "-3 to -2", "-2 to -1", "-1 to -0.5",
        "-0.5 to 0.5",
        "0.5 to 1", "1 to 2", "2 to 3", "3 to 4", "4 to 5",
        "5 to 6", "6 to 7", "7 to 8", "8 to 9", "9 to 10",
    ]
    drift_bins = pd.cut(drift_pct, bins=bins, labels=labels, include_lowest=True)
    drift_counts = drift_bins.value_counts().reindex(labels, fill_value=0)
    drift_df = drift_counts.rename_axis("drift").reset_index(name="count")
    return {
        "drift_counts": drift_df.to_dict(orient="records"),
        "ohlc": history_df[["date", "open", "high", "low", "close", "bucket"]].to_dict(orient="records"),
    }


@app.get("/pre-trade/analysis")
def pre_trade_analysis(
    request: Request,
    spot: Optional[float] = None,
    capital: Optional[float] = None,
    es_limit: float = 4.0,
    delta: Optional[float] = None,
    theta: Optional[float] = None,
    gamma: Optional[float] = None,
    vega: Optional[float] = None,
    repricing_model: str = "black76",
    forward_override: Optional[float] = None,
    audit_scenario: Optional[str] = None,
    audit_symbol: Optional[str] = None,
):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    nifty_df = _load_cache("nifty_ohlcv")
    _prepare_rb_state(options_df, nifty_df)

    positions = _fetch_positions_with_cache(kite)
    regime = calculate_market_regime(options_df, nifty_df)
    current_spot = float(regime.get("current_spot", 25000)) if regime else 25000
    current_spot = float(spot) if spot else current_spot
    enriched = [enrich_position_with_greeks(p, options_df, current_spot) for p in positions]
    greeks = calculate_portfolio_greeks(enriched)

    margin = kite.margins().get("equity", {})
    account_size = float(margin.get("available", {}).get("live_balance", 0.0)) + float(margin.get("utilised", {}).get("debits", 0.0))
    if capital:
        account_size = float(capital)
    if account_size <= 0:
        account_size = 1_000_000.0

    use_delta = float(delta) if delta is not None else float(greeks.get("net_delta", 0.0) or 0.0)
    use_theta = float(theta) if theta is not None else float(greeks.get("net_theta", 0.0) or 0.0)
    use_gamma = float(gamma) if gamma is not None else float(greeks.get("net_gamma", 0.0) or 0.0)
    use_vega = float(vega) if vega is not None else float(greeks.get("net_vega", 0.0) or 0.0)

    normalized = st_tab._compute_normalized(
        use_delta,
        use_theta,
        use_gamma,
        use_vega,
        account_size,
    )
    iv_regime, _ = ra_tab.get_iv_regime(35)
    zone_num, zone_name, zone_color, zone_message = ra_tab.classify_zone(
        normalized["theta"], normalized["gamma"], normalized["vega"], iv_regime
    )
    actionable_insights = st_tab.INSIGHTS.get(zone_num, st_tab.INSIGHTS.get(0, []))

    risk_warnings: List[str] = []
    try:
        if zone_num == 3 and iv_regime == "Low IV":
            risk_warnings.append("Zone 3 posture is blocked in Low IV. Scale down theta or wait for IV expansion.")
        if normalized["theta"] > st_tab.ZONE_RANGES["Theta"]["Zone 3"][1]:
            risk_warnings.append("Theta exceeds Zone 3 ceiling (₹300/₹1L). Lower size or add calendars.")
        if normalized["gamma"] < st_tab.ZONE_RANGES["Gamma"]["Zone 3"][1]:
            risk_warnings.append("Short gamma is beyond risk guardrail (-0.055/₹1L). Reduce near-expiry shorts.")
        vega_ceiling = st_tab.VEGA_RANGES_BY_IV.get(iv_regime, st_tab.VEGA_RANGES_BY_IV["Mid IV"])["Zone 3"][1]
        if vega_ceiling is not None and normalized["vega"] < vega_ceiling:
            risk_warnings.append("Vega is beyond extreme short limit. Add long vega or trim short straddles.")
    except Exception:
        risk_warnings = []
    comparison = st_tab._build_comparison_table(
        normalized["theta"], normalized["gamma"], normalized["vega"], iv_regime
    ).to_dict(orient="records")
    bucket_probs, history_count, used_fallback = ra_tab.compute_historical_bucket_probabilities(
        lookback=252, smoothing_enabled=False, smoothing_span=63
    )
    bucket_rows = st_tab._compute_bucket_rows(
        use_delta,
        use_gamma,
        use_vega,
        account_size,
        current_spot,
        bucket_probs,
    )
    scenarios = ra_tab.get_weighted_scenarios(iv_regime)
    threshold_context = st_tab.build_threshold_report(
        portfolio={
            "delta": use_delta,
            "gamma": use_gamma,
            "vega": use_vega,
            "spot": current_spot,
            "nav": account_size,
            "margin": account_size,
        },
        scenarios=[
            {"name": s.name, "dS_pct": s.ds_pct, "dIV_pts": s.div_pts, "type": s.category.upper()}
            for s in scenarios
        ],
        master_pct=float(es_limit),
        hard_stop_pct=float(es_limit) * 1.2,
        normal_share=st_tab.DEFAULT_THRESHOLD_NORMAL_SHARE,
        stress_share=st_tab.DEFAULT_THRESHOLD_STRESS_SHARE,
    )
    repricing_available = st_tab.pricing_model_available(repricing_model)
    repriced_rows: List[Dict[str, object]] = []
    repriced_skipped = {"missing_fields": 0, "no_price": 0}
    if repricing_available:
        repriced_rows, repriced_skipped = st_tab._repriced_scenario_rows(
            enriched,
            scenarios,
            current_spot,
            account_size,
            threshold_context.get("thresholds", {}),
            pricing_model=repricing_model,
            forward_price=forward_override if forward_override and forward_override > 0 else None,
        )

    repriced_map = {row.get("Scenario"): row for row in repriced_rows}
    scenario_rows = threshold_context.get("rows", [])
    bucket_counts: Dict[str, int] = {}
    for row in scenario_rows:
        bucket = row.get("bucket")
        if bucket:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    scenario_table: List[Dict[str, object]] = []
    for row in scenario_rows:
        scenario = row.get("scenario", {})
        name = scenario.get("name")
        status = row.get("status", "INFO")
        if status == "PASS":
            status_display = "🟢 PASS"
        elif status == "FAIL":
            status_display = "🔴 FAIL"
        else:
            status_display = "ℹ️ INFO"
        repriced = repriced_map.get(name) if name else None
        bucket = row.get("bucket")
        bucket_prob = bucket_probs.get(bucket, 0.0) if bucket else 0.0
        count = bucket_counts.get(bucket, 0) if bucket else 0
        scenario_prob = (bucket_prob / count) if bucket_prob and count else 0.0
        scenario_table.append(
            {
                "Scenario": name,
                "Bucket": bucket,
                "dS% / dIV": f"{scenario.get('dS_pct', 0.0):+.2f}% / {scenario.get('dIV_pts', 0.0):+.1f}",
                "Δ P&L (₹)": st_tab.format_inr(row.get("pnl_delta", 0.0)),
                "Γ P&L (₹)": st_tab.format_inr(row.get("pnl_gamma", 0.0)),
                "Vega P&L (₹)": st_tab.format_inr(row.get("pnl_vega", 0.0)),
                "Total P&L (₹)": st_tab.format_inr(row.get("pnl_total", 0.0)),
                "Repriced P&L (₹)": repriced.get("Repriced P&L (₹)") if repriced else "—",
                "Repriced Loss % NAV": repriced.get("Loss % Capital") if repriced else "—",
                "Loss % NAV": f"{row.get('loss_pct_nav', 0.0):.2f}%",
                "Threshold % NAV": f"{row.get('threshold_pct', 0.0):.2f}%",
                "Probability": f"{scenario_prob * 100:.2f}%",
                "Status": status_display,
            }
        )

    audit_rows: List[Dict[str, object]] = []
    audit_total_pnl: Optional[float] = None
    scenario_names = [scenario.name for scenario in scenarios]
    selected_name = audit_scenario or (scenario_names[0] if scenario_names else None)
    selected_scenario = next((s for s in scenarios if s.name == selected_name), None)
    if selected_scenario and repricing_available:
        audit_df = st_tab._repriced_position_audit(
            enriched,
            selected_scenario,
            current_spot,
            pricing_model=repricing_model,
            forward_price=forward_override if forward_override and forward_override > 0 else None,
        )
        if audit_df is not None and not audit_df.empty:
            audit_rows = audit_df.to_dict(orient="records")
            audit_total_pnl = float(audit_df["P&L (₹)"].sum())

    symbols = sorted({pos.get("tradingsymbol") for pos in enriched if pos.get("tradingsymbol")})
    selected_symbol = audit_symbol or (symbols[0] if symbols else None)
    iv_debug_payload = None
    if selected_symbol:
        match = next((p for p in enriched if p.get("tradingsymbol") == selected_symbol), None)
        if match:
            debug = match.get("iv_debug", {}) if isinstance(match.get("iv_debug"), dict) else {}
            spot_from_cache = None
            if isinstance(nifty_df, pd.DataFrame) and not nifty_df.empty and "close" in nifty_df.columns:
                try:
                    spot_from_cache = float(nifty_df["close"].dropna().iloc[-1])
                except Exception:
                    spot_from_cache = None
            iv_debug_payload = {
                "spot_used": debug.get("spot_used"),
                "spot_from_nifty_df": spot_from_cache,
                "option_price_used": debug.get("option_price_used"),
                "time_to_expiry_years": debug.get("time_to_expiry"),
                "time_to_expiry_days": (float(debug.get("time_to_expiry", 0.0)) * 365.0) if debug.get("time_to_expiry") is not None else None,
                "expiry_date": debug.get("expiry_date"),
                "match_count": debug.get("match_count"),
                "implied_vol_pct": (float(match.get("implied_vol")) * 100.0) if match.get("implied_vol") is not None else None,
                "last_price": match.get("last_price"),
            }

    return {
        "spot": current_spot,
        "capital": account_size,
        "greeks": {
            "net_delta": use_delta,
            "net_theta": use_theta,
            "net_gamma": use_gamma,
            "net_vega": use_vega,
        },
        "normalized": normalized,
        "iv_regime": iv_regime,
        "zone": {"num": zone_num, "name": zone_name, "color": zone_color, "message": zone_message},
        "comparison_table": comparison,
        "actionable_insights": actionable_insights,
        "risk_warnings": risk_warnings,
        "bucket_probs": bucket_probs,
        "bucket_rows": bucket_rows,
        "history_count": history_count,
        "used_fallback": used_fallback,
        "threshold_rows": threshold_context.get("rows", []),
        "scenario_table": scenario_table,
        "repriced_rows": repriced_rows,
        "repriced_skipped": repriced_skipped,
        "repricing_available": repricing_available,
        "repricing_model": repricing_model,
        "audit_scenario": selected_name,
        "audit_rows": audit_rows,
        "audit_total_pnl": audit_total_pnl,
        "iv_debug_symbols": symbols,
        "iv_debug_symbol": selected_symbol,
        "iv_debug": iv_debug_payload,
    }


@app.get("/trade-selector/run")
def trade_selector_run(
    underlying: Optional[str] = None,
    spot: Optional[float] = None,
    expiry: Optional[str] = None,
    strategy: str = "Bull Spread",
    short_min: int = -700,
    short_max: int = 700,
    short_step: int = 100,
    wing_min: int = -1000,
    wing_max: int = 1000,
    wing_step: int = 100,
    lot_size: float = 50.0,
    short_lots: float = 1.0,
    wing_lots: float = 1.0,
    slippage_per_leg: float = 0.0,
    brokerage_per_leg: float = 0.0,
    n_paths: int = 5000,
    rf: float = 0.0,
    jump_enabled: bool = False,
    cvar_level: float = 0.99,
    max_margin: float = 0.0,
    max_spread_pct: float = 0.0,
    min_oi: float = 0.0,
    min_volume: float = 0.0,
    min_pop: float = 0.0,
    positive_ev_only: bool = True,
):
    options_df = _load_cache("nifty_options_ce")
    pe_df = _load_cache("nifty_options_pe")
    if not pe_df.empty:
        options_df = pd.concat([options_df, pe_df], ignore_index=True)
    chain = ts_tab._normalize_chain(options_df)
    if chain.empty:
        raise HTTPException(status_code=404, detail="Options cache missing")
    underlyings = sorted(chain["underlying"].dropna().unique().tolist())
    underlying = underlying or (underlyings[0] if underlyings else "NIFTY")
    chain = chain[chain["underlying"] == underlying]
    spot_default = float(chain["spot"].dropna().median()) if not chain["spot"].dropna().empty else 0.0
    spot_val = float(spot) if spot is not None else spot_default
    expiries = sorted(chain["expiry"].dropna().unique())
    expiry_choice = expiry or (expiries[0].strftime("%Y-%m-%d") if expiries else "")

    distances = tuple(range(int(short_min), int(short_max) + 1, int(short_step))) if short_step else (int(short_min),)
    widths = tuple(range(int(wing_min), int(wing_max) + 1, int(wing_step))) if wing_step else (int(wing_min),)

    strategies_to_run = []
    if strategy == "Bull Spread":
        strategies_to_run = ["Bull Put Spread", "Bull Call Spread"]
    elif strategy == "Bear Spread":
        strategies_to_run = ["Bear Put Spread", "Bear Call Spread"]
    else:
        strategies_to_run = [strategy]

    results_list = []
    for strat in strategies_to_run:
        results = ts_tab._build_trade_table(
            chain,
            strat,
            expiry_choice,
            spot_val,
            distances,
            widths,
            "MID",
            float(slippage_per_leg),
            float(brokerage_per_leg),
            int(n_paths),
            float(rf),
            bool(jump_enabled),
            float(cvar_level),
            float(max_margin),
            float(max_spread_pct),
            float(min_oi),
            float(min_volume),
            float(lot_size),
            float(short_lots),
            float(wing_lots),
        )
        if not results.empty:
            results_list.append(results)

    if results_list:
        results_df = pd.concat(results_list, ignore_index=True)
    else:
        results_df = pd.DataFrame()

    if positive_ev_only and not results_df.empty:
        results_df = results_df[results_df["expected_pnl"] > 0]
    if min_pop > 0 and not results_df.empty:
        results_df = results_df[results_df["pop"] >= (float(min_pop) / 100.0)]

    results_df = results_df.sort_values(["ev_over_cvar", "expected_pnl"], ascending=[False, False]) if not results_df.empty else results_df
    if not results_df.empty:
        results_df = results_df.reset_index(drop=True)
        results_df["row_id"] = results_df.index.astype(int)

    expiry_preview = []
    expiry_stats = {}
    preview_rows = []
    if expiry_choice:
        expiry_dt = pd.to_datetime(expiry_choice, errors="coerce")
        expiry_chain = chain[chain["expiry"] == expiry_dt].copy()
        if not expiry_chain.empty:
            expiry_stats = {
                "rows": int(len(expiry_chain)),
                "price_gt_zero": int((expiry_chain["price"] > 0).sum()),
                "close_gt_zero": int((expiry_chain["close"] > 0).sum()) if "close" in expiry_chain.columns else 0,
            }
            expiry_preview = expiry_chain[
                ["option_type", "strike", "price", "close", "ltp", "oi", "volume"]
            ].head(10).to_dict(orient="records")
        preview_df = ts_tab._preview_selected_legs(
            strategies_to_run[0] if strategies_to_run else "Bull Put Spread",
            expiry_dt,
            spot_val,
            chain,
            list(distances),
            list(widths),
        )
        if preview_df is not None and not preview_df.empty:
            preview_rows = preview_df.to_dict(orient="records")

    return {
        "rows": results_df.to_dict(orient="records"),
        "underlying": underlying,
        "expiry": expiry_choice,
        "spot": spot_val,
        "debug": {
            "expiry_stats": expiry_stats,
            "expiry_preview": expiry_preview,
            "preview_legs": preview_rows,
        },
    }


@app.get("/historical/summary")
def historical_summary():
    tradebook_path = ROOT / "database" / "tradebook.csv"
    if not tradebook_path.exists():
        raise HTTPException(status_code=404, detail="tradebook.csv not found")
    df = pd.read_csv(tradebook_path)
    df = hp_tab._normalize_df(df)
    trades, warnings = hp_tab._prepare_trades(df)
    if trades.empty:
        return {"warnings": warnings, "trades": [], "monthly": [], "summary": {}}
    expiry_trades = hp_tab._aggregate_by_expiry(trades)
    monthly = hp_tab._monthly_summary(expiry_trades)
    summary = {
        "total_realized": float(expiry_trades["realized_pnl"].sum()),
        "total_trades": int(len(expiry_trades)),
    }
    by_type = expiry_trades.groupby("option_type")["realized_pnl"].sum().reset_index()
    expiry_summary = (
        expiry_trades.groupby("expiry_type")["realized_pnl"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
    strikes = expiry_trades.dropna(subset=["strike"]).copy()
    strike_summary = pd.DataFrame()
    if not strikes.empty:
        min_strike = strikes["strike"].min()
        max_strike = strikes["strike"].max()
        step = 500 if max_strike - min_strike <= 20000 else 1000
        bins = list(range(int(min_strike // step * step), int(max_strike + step), step))
        strikes["strike_bucket"] = pd.cut(strikes["strike"], bins=bins, include_lowest=True).astype(str)
        strike_summary = strikes.groupby("strike_bucket")["realized_pnl"].sum().reset_index()
    pnl_pct = expiry_trades["realized_pnl_pct"].dropna().tolist()
    wins = int((expiry_trades["realized_pnl"] > 0).sum())
    losses = int((expiry_trades["realized_pnl"] <= 0).sum())
    top_contrib = expiry_trades.sort_values("realized_pnl", ascending=False).head(10)
    bottom_contrib = expiry_trades.sort_values("realized_pnl", ascending=True).head(10)
    return {
        "warnings": warnings,
        "trades": expiry_trades.to_dict(orient="records"),
        "monthly": monthly.to_dict(orient="records"),
        "summary": summary,
        "by_type": by_type.to_dict(orient="records"),
        "by_expiry_type": expiry_summary.to_dict(orient="records"),
        "by_strike": strike_summary.to_dict(orient="records"),
        "pnl_pct": pnl_pct,
        "win_loss": {"wins": wins, "losses": losses},
        "top_contributors": top_contrib.to_dict(orient="records"),
        "bottom_contributors": bottom_contrib.to_dict(orient="records"),
    }


@app.post("/historical/tradebook")
async def upload_tradebook(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    tradebook_path = ROOT / "database" / "tradebook.csv"
    tradebook_path.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    tradebook_path.write_bytes(content)
    return {"status": "ok", "filename": file.filename}


@app.get("/product/overview")
def product_overview():
    return {
        "framework": "Portfolio → Bucket → Trade → Meta workflow",
        "daily_workflow": [
            "Morning check",
            "Intraday check",
            "EOD review",
        ],
    }


@app.get("/logs/ping")
def log_ping():
    logger.info("log_ping")
    return {"ok": True}


@app.get("/diagnostics/cache")
def diagnostics_cache():
    def _status(name: str) -> Dict[str, Any]:
        path = CACHE_DIR / f"{name}.csv"
        exists = path.exists()
        return {
            "name": name,
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else 0,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None,
        }

    return {
        "cache": [
            _status("nifty_ohlcv"),
            _status("nifty_futures"),
            _status("nifty_options_ce"),
            _status("nifty_options_pe"),
        ]
    }


@app.get("/diagnostics/kite")
def diagnostics_kite(request: Request):
    api_key = request.headers.get("x-kite-api-key") or os.getenv("KITE_API_KEY")
    access_token = request.headers.get("x-kite-access-token") or os.getenv("KITE_ACCESS_TOKEN")
    cached = _load_kite_credentials_file()
    cached_valid = bool(
        cached.get("api_key")
        and cached.get("access_token")
        and not _is_token_expired(cached.get("saved_at"))
    )
    return {
        "env_api_key": bool(api_key),
        "env_access_token": bool(access_token),
        "cache_file_exists": KITE_CREDS_FILE.exists(),
        "cached_valid": cached_valid,
        "cached_saved_at": cached.get("saved_at"),
        "token_expired": _is_token_expired(cached.get("saved_at")),
    }


@app.get("/data-source/cache-status")
def data_source_cache_status():
    def _status(name: str) -> Dict[str, Any]:
        path = CACHE_DIR / f"{name}.csv"
        exists = path.exists()
        updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None
        return {
            "name": name,
            "exists": exists,
            "updated_at": updated_at,
            "size_bytes": path.stat().st_size if exists else 0,
            "latest_data_date": updated_at[:10] if isinstance(updated_at, str) else None,
        }

    return {
        "nifty_ohlcv": _status("nifty_ohlcv"),
        "nifty_futures": _status("nifty_futures"),
        "nifty_options_ce": _status("nifty_options_ce"),
        "nifty_options_pe": _status("nifty_options_pe"),
        "participants": _participant_cache_status(),
    }


@app.get("/data-source/preview/{name}")
def data_source_preview(name: str, limit: int = 20):
    name = name.strip()
    if name not in {"nifty_ohlcv", "nifty_futures", "nifty_options_ce", "nifty_options_pe"}:
        raise HTTPException(status_code=400, detail="Unsupported dataset")
    df = _load_cache(name)
    if df.empty:
        raise HTTPException(status_code=404, detail="Cache missing")
    try:
        return {"rows": _df_to_records(df, limit)}
    except Exception as exc:
        logger.exception("preview serialize failed for %s", name)
        raise HTTPException(status_code=500, detail=f"Failed to serialize preview: {exc}")


def _last_tuesday(year: int, month: int) -> datetime:
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])
    while last_day.weekday() != 1:
        last_day -= timedelta(days=1)
    if last_day.month == 3 and last_day.day == 31:
        last_day -= timedelta(days=1)
    return last_day


def _build_expiry_list(previous_count: int = 5, next_count: int = 5) -> List[datetime]:
    today = datetime.now()
    months_back = max(1, int(previous_count)) + 1
    months_fwd = max(1, int(next_count)) + 1
    expiries: List[datetime] = []
    start_month_anchor = datetime(today.year, today.month, 1) - timedelta(days=31 * months_back)
    end_month_anchor = datetime(today.year, today.month, 1) + timedelta(days=31 * months_fwd)
    year = start_month_anchor.year
    month = start_month_anchor.month
    while (year, month) <= (end_month_anchor.year, end_month_anchor.month):
        expiries.append(_last_tuesday(year, month))
        month += 1
        if month > 12:
            month = 1
            year += 1

    past = sorted([e for e in expiries if e < today], reverse=True)[: max(0, int(previous_count))]
    future = sorted([e for e in expiries if e >= today])[: max(0, int(next_count))]
    unique_expiries = sorted({e.date(): e for e in (past + future)}.values())
    return unique_expiries


def _iter_dates(start_date: date, end_date: date) -> List[date]:
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="start_date must be <= end_date")
    days = (end_date - start_date).days
    return [start_date + timedelta(days=offset) for offset in range(days + 1)]


def _nse_reports_archives_payload() -> List[Dict[str, str]]:
    return [
        {
            "name": "F&O - Participant wise Open Interest(csv)",
            "type": "archives",
            "category": "derivatives",
            "section": "equity",
        },
        {
            "name": "F&O - Participant wise Trading Volumes(csv)",
            "type": "archives",
            "category": "derivatives",
            "section": "equity",
        },
    ]


def _extract_date_from_participant_filename(path: Path) -> Optional[date]:
    token = path.stem.split("_")[-1]
    if len(token) != 8 or not token.isdigit():
        return None
    try:
        return datetime.strptime(token, "%d%m%Y").date()
    except Exception:
        return None


def _to_num(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def _parse_participant_csv(path: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path, skiprows=1)
    df.columns = [str(col).strip() for col in df.columns]
    if "Client Type" not in df.columns:
        raise ValueError(f"Unexpected participant CSV format: {path}")

    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        raw_name = str(row.get("Client Type", "")).strip()
        if not raw_name:
            continue
        normalized = raw_name.upper()
        if normalized == "TOTAL":
            continue
        key = "PRO" if normalized == "PRO" else normalized
        if key not in {"FII", "DII", "PRO", "CLIENT"}:
            continue
        fi_long = _to_num(row.get("Future Index Long"))
        fi_short = _to_num(row.get("Future Index Short"))
        call_long = _to_num(row.get("Option Index Call Long"))
        call_short = _to_num(row.get("Option Index Call Short"))
        put_long = _to_num(row.get("Option Index Put Long"))
        put_short = _to_num(row.get("Option Index Put Short"))
        total_long = _to_num(row.get("Total Long Contracts"))
        total_short = _to_num(row.get("Total Short Contracts"))
        out[key] = {
            "future_index_long": fi_long,
            "future_index_short": fi_short,
            "future_index_net": fi_long - fi_short,
            "option_index_call_long": call_long,
            "option_index_call_short": call_short,
            "option_index_call_net": call_long - call_short,
            "option_index_put_long": put_long,
            "option_index_put_short": put_short,
            "option_index_put_net": put_long - put_short,
            "total_long": total_long,
            "total_short": total_short,
            "total_net": total_long - total_short,
        }
    return out


def _participant_cache_status() -> Dict[str, Any]:
    nse_dir = ROOT / "database" / "nse_reports"
    if not nse_dir.exists():
        return {
            "name": "participants",
            "exists": False,
            "updated_at": None,
            "size_bytes": 0,
            "latest_data_date": None,
        }

    oi_paths = sorted(nse_dir.glob("fao_participant_oi_*.csv"))
    vol_paths = sorted(nse_dir.glob("fao_participant_vol_*.csv"))
    oi_by_date: Dict[date, Path] = {}
    vol_by_date: Dict[date, Path] = {}

    for path in oi_paths:
        parsed_date = _extract_date_from_participant_filename(path)
        if parsed_date:
            oi_by_date[parsed_date] = path
    for path in vol_paths:
        parsed_date = _extract_date_from_participant_filename(path)
        if parsed_date:
            vol_by_date[parsed_date] = path

    common_dates = sorted(set(oi_by_date.keys()) & set(vol_by_date.keys()))
    if not common_dates:
        return {
            "name": "participants",
            "exists": False,
            "updated_at": None,
            "size_bytes": 0,
            "latest_data_date": None,
        }

    latest_date = common_dates[-1]
    latest_files = [oi_by_date[latest_date], vol_by_date[latest_date]]
    latest_mtime = max([f.stat().st_mtime for f in latest_files if f.exists()], default=0.0)
    total_size = sum(path.stat().st_size for path in (oi_paths + vol_paths) if path.exists())
    return {
        "name": "participants",
        "exists": True,
        "updated_at": datetime.fromtimestamp(latest_mtime).isoformat() if latest_mtime > 0 else None,
        "size_bytes": total_size,
        "latest_data_date": latest_date.isoformat(),
    }


def _load_nifty_close_map() -> Dict[date, float]:
    path = CACHE_DIR / "nifty_ohlcv.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "date" not in df.columns or "close" not in df.columns:
        return {}
    dt = pd.to_datetime(df["date"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    out: Dict[date, float] = {}
    for idx in range(len(df)):
        d = dt.iloc[idx]
        c = close.iloc[idx]
        if pd.isna(d) or pd.isna(c):
            continue
        out[d.date()] = float(c)
    return out


def _wilson_score_interval(successes: int, trials: int, z: float = 1.96) -> Dict[str, float]:
    n = int(max(0, trials))
    k = int(max(0, min(successes, n)))
    if n == 0:
        return {"low": 0.0, "high": 0.0}
    phat = k / n
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (phat + (z2 / (2.0 * n))) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    return {"low": max(0.0, center - margin), "high": min(1.0, center + margin)}


def _gap_bucket_metrics(segment: pd.DataFrame, direction: str, threshold_pct: float) -> Dict[str, Any]:
    if segment.empty:
        return {
            "events": 0,
            "hit_count": 0,
            "hit_rate_pct": 0.0,
            "hit_rate_ci_low_pct": 0.0,
            "hit_rate_ci_high_pct": 0.0,
            "extend_beyond_open_gap_count": 0,
            "extend_beyond_open_gap_pct": 0.0,
            "reversal_count": 0,
            "reversal_pct": 0.0,
            "mean_close_pct": None,
            "median_gap_pct": None,
            "median_close_pct": None,
            "mean_close_minus_gap_pct": None,
        }

    if direction == "up":
        hit_mask = segment["close_pct"] > float(threshold_pct)
        extend_mask = segment["close_pct"] >= segment["gap_pct"]
        reversal_mask = segment["close_pct"] < 0.0
    else:
        hit_mask = segment["close_pct"] < (-1.0 * float(threshold_pct))
        extend_mask = segment["close_pct"] <= segment["gap_pct"]
        reversal_mask = segment["close_pct"] > 0.0

    events = int(len(segment))
    hit_count = int(hit_mask.sum())
    extend_count = int(extend_mask.sum())
    reversal_count = int(reversal_mask.sum())
    ci = _wilson_score_interval(hit_count, events)
    return {
        "events": events,
        "hit_count": hit_count,
        "hit_rate_pct": float((hit_count / events) * 100.0),
        "hit_rate_ci_low_pct": float(ci["low"] * 100.0),
        "hit_rate_ci_high_pct": float(ci["high"] * 100.0),
        "extend_beyond_open_gap_count": extend_count,
        "extend_beyond_open_gap_pct": float((extend_count / events) * 100.0),
        "reversal_count": reversal_count,
        "reversal_pct": float((reversal_count / events) * 100.0),
        "mean_close_pct": float(segment["close_pct"].mean()),
        "median_gap_pct": float(segment["gap_pct"].median()),
        "median_close_pct": float(segment["close_pct"].median()),
        "mean_close_minus_gap_pct": float((segment["close_pct"] - segment["gap_pct"]).mean()),
    }


WORLD_INDEXES: List[Dict[str, str]] = [
    {"country": "US", "symbol": "^GSPC", "name": "S&P 500"},
    {"country": "US", "symbol": "^IXIC", "name": "NASDAQ Composite"},
    {"country": "US", "symbol": "^DJI", "name": "Dow Jones"},
    {"country": "Japan", "symbol": "^N225", "name": "Nikkei 225"},
    {"country": "Germany", "symbol": "^GDAXI", "name": "DAX"},
    {"country": "Germany", "symbol": "^MDAXI", "name": "MDAX"},
    {"country": "France", "symbol": "^FCHI", "name": "CAC 40"},
    {"country": "France", "symbol": "^SBF120", "name": "SBF 120"},
    {"country": "UK", "symbol": "^FTSE", "name": "FTSE 100"},
    {"country": "UK", "symbol": "^FTMC", "name": "FTSE 250"},
]


def _safe_ret(series: List[float], lookback: int) -> Optional[float]:
    if len(series) <= lookback:
        return None
    prev = float(series[-(lookback + 1)])
    curr = float(series[-1])
    if prev == 0:
        return None
    return ((curr / prev) - 1.0) * 100.0


def _fetch_yahoo_index_series(symbol: str, range_window: str = "6mo") -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": "1d", "range": range_window}
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    chart = (((payload or {}).get("chart") or {}).get("result") or [])
    if not chart:
        return pd.DataFrame(columns=["date", "close"])
    block = chart[0] if isinstance(chart[0], dict) else {}
    ts = block.get("timestamp") or []
    indicators = block.get("indicators") or {}
    quote = (indicators.get("quote") or [{}])[0]
    open_arr = quote.get("open") or []
    high_arr = quote.get("high") or []
    low_arr = quote.get("low") or []
    close_arr = quote.get("close") or []
    if not ts or not close_arr:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])

    def _aligned(values: Any, n: int) -> List[Any]:
        if not isinstance(values, list):
            return [None] * n
        if len(values) >= n:
            return values[:n]
        return values + ([None] * (n - len(values)))

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "open": _aligned(open_arr, len(ts)),
            "high": _aligned(high_arr, len(ts)),
            "low": _aligned(low_arr, len(ts)),
            "close": _aligned(close_arr, len(ts)),
        }
    )
    out["date"] = pd.to_datetime(out["timestamp"], unit="s", utc=True).dt.date
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return out[["date", "open", "high", "low", "close"]]


def _nifty_world_correlation_analysis(
    world_history_by_symbol: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    nifty_path = CACHE_DIR / "nifty_ohlcv.csv"
    if not nifty_path.exists():
        return {"error": "NIFTY OHLCV cache missing"}
    try:
        nifty_df = pd.read_csv(nifty_path)
    except Exception as exc:
        return {"error": f"Failed to read nifty_ohlcv.csv: {exc}"}

    if nifty_df.empty or "date" not in nifty_df.columns or "close" not in nifty_df.columns:
        return {"error": "nifty_ohlcv.csv missing required columns: date/close"}

    ndt = pd.to_datetime(nifty_df["date"], errors="coerce")
    nclose = pd.to_numeric(nifty_df["close"], errors="coerce")
    n = pd.DataFrame({"date": ndt.dt.date, "close": nclose}).dropna(subset=["date", "close"]).copy()
    n = n.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    if n.empty:
        return {"error": "NIFTY series empty after cleaning"}

    n["nifty_ret"] = n["close"].pct_change()
    wide = n[["date", "nifty_ret"]].copy()

    for symbol, sdf in world_history_by_symbol.items():
        if sdf.empty:
            continue
        d = sdf[["date", "close"]].copy()
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d = d.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if d.empty:
            continue
        d[f"ret_{symbol}"] = d["close"].pct_change()
        # Keep NIFTY trading calendar as base; do not require exact all-market overlap.
        wide = wide.merge(d[["date", f"ret_{symbol}"]], on="date", how="left")

    ret_cols = [c for c in wide.columns if c.startswith("ret_")]
    wide = wide.dropna(subset=["nifty_ret"]).copy()
    if wide.empty or not ret_cols:
        return {"error": "Insufficient data for NIFTY/world return analysis"}

    corr_rows: List[Dict[str, Any]] = []
    for c in ret_cols:
        pair = wide[["nifty_ret", c]].dropna()
        corr = pair["nifty_ret"].corr(pair[c]) if not pair.empty else np.nan
        overlap_samples = int(len(pair))
        symbol = c.replace("ret_", "", 1)
        corr_rows.append(
            {
                "symbol": symbol,
                "corr_with_nifty": float(corr) if pd.notna(corr) else None,
                "overlap_samples": overlap_samples,
            }
        )
    corr_rows = sorted(corr_rows, key=lambda r: abs(float(r.get("corr_with_nifty") or 0.0)), reverse=True)

    # Linear model: nifty_ret = intercept + sum(beta_i * world_ret_i)
    x_cols = [r["symbol"] for r in corr_rows]
    x_frame = wide[[f"ret_{s}" for s in x_cols]].copy()
    coverage_by_symbol: List[Dict[str, Any]] = []
    for s in x_cols:
        col = f"ret_{s}"
        cov = float(x_frame[col].notna().mean()) if col in x_frame.columns and len(x_frame) > 0 else 0.0
        coverage_by_symbol.append({"symbol": s, "coverage_pct": cov * 100.0})
    x_mat = x_frame.fillna(0.0).to_numpy(dtype=float)
    y_vec = wide["nifty_ret"].to_numpy(dtype=float)
    if x_mat.size == 0 or y_vec.size == 0:
        return {
            "samples": int(len(wide)),
            "date_start": str(wide["date"].iloc[0]),
            "date_end": str(wide["date"].iloc[-1]),
            "correlation_by_symbol": corr_rows,
            "model": {"error": "No valid predictors"},
        }

    x_aug = np.column_stack([np.ones(len(x_mat)), x_mat])
    coef, *_ = np.linalg.lstsq(x_aug, y_vec, rcond=None)
    y_pred = x_aug @ coef
    ss_res = float(np.sum((y_vec - y_pred) ** 2))
    ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

    beta_rows: List[Dict[str, Any]] = []
    for i, symbol in enumerate(x_cols, start=1):
        beta_rows.append({"symbol": symbol, "beta": float(coef[i])})
    beta_rows = sorted(beta_rows, key=lambda r: abs(float(r["beta"])), reverse=True)

    return {
        "samples": int(len(wide)),
        "date_start": str(wide["date"].iloc[0]),
        "date_end": str(wide["date"].iloc[-1]),
        "target": "nifty_ret",
        "feature_count": int(len(x_cols)),
        "correlation_by_symbol": corr_rows,
        "model": {
            "method": "OLS",
            "intercept": float(coef[0]),
            "r2": r2,
            "betas": beta_rows,
            "feature_coverage": coverage_by_symbol,
            "imputation": "Missing input returns filled with 0.0 on NIFTY trading dates",
        },
    }


@app.get("/spot-analysis/world-indexes")
def spot_analysis_world_indexes():
    db_dir = ROOT / "database" / "world_indexes"
    db_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = db_dir / "world_indexes_snapshot.json"
    history_path = db_dir / "world_indexes_history.csv"

    rows: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []
    history_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    history_frames_by_symbol: Dict[str, pd.DataFrame] = {}
    generated_at = datetime.now(timezone.utc).isoformat()

    for meta in WORLD_INDEXES:
        symbol = str(meta["symbol"])
        try:
            series_df = _fetch_yahoo_index_series(symbol, range_window="2y")
        except Exception as exc:
            rows.append(
                {
                    "country": meta["country"],
                    "symbol": symbol,
                    "name": meta["name"],
                    "error": str(exc),
                }
            )
            continue

        if series_df.empty:
            rows.append(
                {
                    "country": meta["country"],
                    "symbol": symbol,
                    "name": meta["name"],
                    "error": "No data returned",
                }
            )
            continue

        history_frames_by_symbol[symbol] = series_df[["date", "close"]].copy()

        closes = [float(x) for x in series_df["close"].tolist() if pd.notna(x)]
        latest_close = closes[-1] if closes else None
        prev_close = closes[-2] if len(closes) >= 2 else None
        day_change_pct = (
            ((latest_close / prev_close) - 1.0) * 100.0
            if latest_close is not None and prev_close is not None and prev_close != 0
            else None
        )
        last_date = series_df["date"].iloc[-1]
        rows.append(
            {
                "country": meta["country"],
                "symbol": symbol,
                "name": meta["name"],
                "last_date": last_date.isoformat() if hasattr(last_date, "isoformat") else str(last_date),
                "close": latest_close,
                "prev_close": prev_close,
                "day_change_pct": day_change_pct,
                "ret_5d_pct": _safe_ret(closes, 5),
                "ret_1m_pct": _safe_ret(closes, 21),
                "ret_3m_pct": _safe_ret(closes, 63),
                "points_count": int(len(closes)),
            }
        )

        symbol_hist: List[Dict[str, Any]] = []
        for _, row in series_df.iterrows():
            d = row["date"]
            row_payload = {
                "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            symbol_hist.append(row_payload)
            history_rows.append(
                {
                    **row_payload,
                    "country": meta["country"],
                    "symbol": symbol,
                    "name": meta["name"],
                }
            )
        history_by_symbol[symbol] = symbol_hist

    history_df = pd.DataFrame(history_rows)
    if not history_df.empty:
        history_df = history_df.sort_values(["country", "symbol", "date"]).drop_duplicates(
            subset=["symbol", "date"], keep="last"
        )
        history_df.to_csv(history_path, index=False)

    snapshot_payload = {
        "generated_at": generated_at,
        "source": "Yahoo Finance chart API",
        "rows": rows,
        "history_by_symbol": history_by_symbol,
        "nifty_relationship": _nifty_world_correlation_analysis(history_frames_by_symbol),
        "db": {
            "snapshot_path": str(snapshot_path),
            "history_path": str(history_path),
            "symbols_count": len(WORLD_INDEXES),
        },
    }
    snapshot_path.write_text(json.dumps(snapshot_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    return snapshot_payload


@app.get("/spot-analysis/gap-edge")
def spot_analysis_gap_edge(threshold_pct: float = 0.5):
    path = CACHE_DIR / "nifty_ohlcv.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="NIFTY OHLCV cache missing")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read nifty_ohlcv.csv: {exc}")

    if df.empty or "date" not in df.columns or "open" not in df.columns or "close" not in df.columns:
        raise HTTPException(status_code=500, detail="nifty_ohlcv.csv missing required columns: date/open/close")

    dt = pd.to_datetime(df["date"], errors="coerce")
    open_px = pd.to_numeric(df["open"], errors="coerce")
    close_px = pd.to_numeric(df["close"], errors="coerce")
    ohlc = pd.DataFrame({"date": dt, "open": open_px, "close": close_px}).dropna(subset=["date", "open", "close"])
    if ohlc.empty:
        raise HTTPException(status_code=500, detail="NIFTY OHLCV cache has no valid rows")

    ohlc = ohlc.sort_values("date").reset_index(drop=True)
    ohlc["prev_close"] = ohlc["close"].shift(1)
    ohlc = ohlc.dropna(subset=["prev_close"]).copy()
    ohlc = ohlc[ohlc["prev_close"] != 0].copy()
    if ohlc.empty:
        raise HTTPException(status_code=500, detail="NIFTY OHLCV cache does not have valid previous-close rows")

    ohlc["gap_pct"] = ((ohlc["open"] / ohlc["prev_close"]) - 1.0) * 100.0
    ohlc["close_pct"] = ((ohlc["close"] / ohlc["prev_close"]) - 1.0) * 100.0
    max_date = ohlc["date"].max()
    years_windows = [1, 3, 5]

    windows: List[Dict[str, Any]] = []
    for years in years_windows:
        start_date = max_date - pd.DateOffset(years=years)
        segment = ohlc[ohlc["date"] >= start_date].copy()
        gap_up = segment[segment["gap_pct"] > float(threshold_pct)].copy()
        gap_down = segment[segment["gap_pct"] < (-1.0 * float(threshold_pct))].copy()
        combined_events = int(len(gap_up) + len(gap_down))
        combined_hit_count = int(
            (gap_up["close_pct"] > float(threshold_pct)).sum() + (gap_down["close_pct"] < (-1.0 * float(threshold_pct))).sum()
        )
        combined_ci = _wilson_score_interval(combined_hit_count, combined_events)
        windows.append(
            {
                "years": int(years),
                "start_date": start_date.date().isoformat(),
                "end_date": max_date.date().isoformat(),
                "sample_days": int(len(segment)),
                "threshold_pct": float(threshold_pct),
                "gap_up": _gap_bucket_metrics(gap_up, direction="up", threshold_pct=threshold_pct),
                "gap_down": _gap_bucket_metrics(gap_down, direction="down", threshold_pct=threshold_pct),
                "combined": {
                    "events": combined_events,
                    "hit_count": combined_hit_count,
                    "hit_rate_pct": float((combined_hit_count / combined_events) * 100.0) if combined_events > 0 else 0.0,
                    "hit_rate_ci_low_pct": float(combined_ci["low"] * 100.0),
                    "hit_rate_ci_high_pct": float(combined_ci["high"] * 100.0),
                },
            }
        )

    return {
        "source": str(path),
        "as_of_date": max_date.date().isoformat(),
        "threshold_pct": float(threshold_pct),
        "windows": windows,
    }


@app.get("/spot-analysis/participants")
def spot_analysis_participants(limit: int = 120):
    nse_dir = ROOT / "database" / "nse_reports"
    if not nse_dir.exists():
        raise HTTPException(status_code=404, detail="NSE reports directory not found")

    oi_paths = sorted(nse_dir.glob("fao_participant_oi_*.csv"))
    vol_paths = sorted(nse_dir.glob("fao_participant_vol_*.csv"))
    if not oi_paths or not vol_paths:
        raise HTTPException(status_code=404, detail="Participant OI/VOL files not found")

    oi_by_date: Dict[date, Path] = {}
    vol_by_date: Dict[date, Path] = {}
    for p in oi_paths:
        d = _extract_date_from_participant_filename(p)
        if d:
            oi_by_date[d] = p
    for p in vol_paths:
        d = _extract_date_from_participant_filename(p)
        if d:
            vol_by_date[d] = p

    all_common_dates = sorted(set(oi_by_date.keys()) & set(vol_by_date.keys()))
    if not all_common_dates:
        return {"rows": []}

    close_map = _load_nifty_close_map()
    sorted_close_dates = sorted(close_map.keys())
    previous_close: Dict[date, float] = {}
    prev_val: Optional[float] = None
    for d in sorted_close_dates:
        if prev_val is not None:
            previous_close[d] = prev_val
        prev_val = close_map[d]

    parsed_by_date: Dict[date, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for d in all_common_dates:
        parsed_by_date[d] = {
            "oi": _parse_participant_csv(oi_by_date[d]),
            "vol": _parse_participant_csv(vol_by_date[d]),
        }

    rows_asc: List[Dict[str, Any]] = []
    prev_date: Optional[date] = None
    for d in all_common_dates:
        oi_data = parsed_by_date[d]["oi"]
        vol_data = parsed_by_date[d]["vol"]
        prev_oi_data = parsed_by_date[prev_date]["oi"] if prev_date in parsed_by_date else {}
        participants: Dict[str, Dict[str, Any]] = {}
        for name in ["FII", "DII", "PRO", "CLIENT"]:
            oi = oi_data.get(name, {})
            vol = vol_data.get(name, {})
            prev_oi = prev_oi_data.get(name, {}) if isinstance(prev_oi_data, dict) else {}
            fut_net_oi = _to_num(oi.get("future_index_net"))
            fut_oi_chg = fut_net_oi - _to_num(prev_oi.get("future_index_net")) if prev_date else None
            call_net_oi = _to_num(oi.get("option_index_call_net"))
            put_net_oi = _to_num(oi.get("option_index_put_net"))
            call_oi_chg = call_net_oi - _to_num(prev_oi.get("option_index_call_net")) if prev_date else None
            put_oi_chg = put_net_oi - _to_num(prev_oi.get("option_index_put_net")) if prev_date else None
            participants[name.lower()] = {
                "futures_index_net_oi": fut_net_oi,
                "futures_index_net_volume": _to_num(vol.get("future_index_net")),
                "futures_index_oi_change": fut_oi_chg,
                "option_index_call_net_oi": _to_num(oi.get("option_index_call_net")),
                "option_index_put_net_oi": _to_num(oi.get("option_index_put_net")),
                "option_index_call_oi_change": call_oi_chg,
                "option_index_put_oi_change": put_oi_chg,
                "option_index_call_net_volume": _to_num(vol.get("option_index_call_net")),
                "option_index_put_net_volume": _to_num(vol.get("option_index_put_net")),
                "total_net_volume": _to_num(vol.get("total_net")),
                "total_net_oi": _to_num(oi.get("total_net")),
                "signal": "BULLISH" if _to_num(fut_oi_chg) >= 0 else "BEARISH",
            }

        close = close_map.get(d)
        prev_close = previous_close.get(d)
        pct = None
        if close is not None and prev_close is not None and prev_close != 0:
            pct = ((close - prev_close) / prev_close) * 100.0

        rows_asc.append(
            {
                "date": d.isoformat(),
                "nifty_close": close,
                "nifty_change_pct": pct,
                "fii_buy_sell_amt_cr": (
                    (_to_num(participants.get("fii", {}).get("futures_index_oi_change")) * float(close)) / 1e7
                    if close is not None
                    else None
                ),
                "fii_cash_cr": (
                    (_to_num(participants.get("fii", {}).get("total_net_volume")) * float(close)) / 1e7
                    if close is not None
                    else None
                ),
                "dii_cash_cr": (
                    (_to_num(participants.get("dii", {}).get("total_net_volume")) * float(close)) / 1e7
                    if close is not None
                    else None
                ),
                "participants": participants,
            }
        )
        prev_date = d

    rows = list(reversed(rows_asc))
    if limit > 0:
        rows = rows[:limit]
    return {"rows": rows}


@app.get("/spot-analysis/flow-model-summary")
def spot_analysis_flow_model_summary(pred_limit: int = 20):
    model_dir = ROOT / "artifacts" / "flow_spot_models"
    metrics_path = model_dir / "metrics.json"
    preds_path = model_dir / "test_predictions.csv"
    diagnostics_path = model_dir / "diagnostics.json"
    live_signal_path = model_dir / "live_signal.json"

    if not metrics_path.exists() or not preds_path.exists():
        raise HTTPException(status_code=404, detail="Model artifacts not found. Run training first.")

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read metrics.json: {exc}")

    try:
        preds_df = pd.read_csv(preds_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read test_predictions.csv: {exc}")

    rows: List[Dict[str, Any]] = []
    if not preds_df.empty:
        preds_df["date"] = pd.to_datetime(preds_df.get("date"), errors="coerce")
        preds_df = preds_df.sort_values(["horizon", "date"], ascending=[True, False])
        if pred_limit > 0:
            preds_df = preds_df.groupby("horizon", as_index=False).head(pred_limit)
        for row in preds_df.to_dict(orient="records"):
            clean: Dict[str, Any] = {}
            for k, v in row.items():
                clean[k] = _json_safe(v)
            rows.append(clean)

    diagnostics: Dict[str, Any] = {}
    if diagnostics_path.exists():
        try:
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        except Exception:
            diagnostics = {}

    live_signal: Dict[str, Any] = {"rows": []}
    if live_signal_path.exists():
        try:
            live_signal = json.loads(live_signal_path.read_text(encoding="utf-8"))
        except Exception:
            live_signal = {"rows": []}

    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "predictions": rows,
        "live_signal": live_signal,
    }


@app.get("/spot-analysis/feature-diagnostics")
def spot_analysis_feature_diagnostics(
    rolling_window: int = 90,
    top_n: int = 30,
    corr_threshold: float = 0.9,
):
    artifacts_dir = ROOT / "artifacts" / "flow_spot_models"
    metrics_path = artifacts_dir / "metrics.json"
    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    target_mode = str(metrics.get("target_mode") or "forward_mean_return")
    test_days = int(metrics.get("test_days") or 10)
    horizons_raw = metrics.get("horizons") or []
    horizons = sorted(
        {int(h.get("horizon")) for h in horizons_raw if isinstance(h, dict) and h.get("horizon") is not None}
    ) or [1, 3, 5, 10]
    selected_by_h: Dict[int, List[str]] = {
        int(h.get("horizon")): [str(x) for x in (h.get("selected_features") or [])]
        for h in horizons_raw
        if isinstance(h, dict) and h.get("horizon") is not None
    }

    try:
        df = flow_train._build_dataset()
        feature_cols = flow_train._feature_columns(df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build feature dataset: {exc}")
    if not feature_cols:
        raise HTTPException(status_code=500, detail="No feature columns available")

    working = df.dropna(subset=feature_cols).reset_index(drop=True)
    rw = max(20, int(rolling_window))
    tn = max(5, int(top_n))
    th = max(0.0, min(0.999, float(corr_threshold)))

    by_horizon: Dict[str, List[Dict[str, Any]]] = {}
    all_selected_union: List[str] = []
    for h in horizons:
        target_col = f"target_ret_{h}"
        d = flow_train._make_targets(working, h, target_mode=target_mode).dropna(subset=[target_col]).copy()
        if len(d) <= test_days:
            by_horizon[str(h)] = []
            continue
        train = d.iloc[:-test_days].copy()
        y = pd.to_numeric(train[target_col], errors="coerce")
        selected = [f for f in selected_by_h.get(h, []) if f in train.columns]
        if not selected:
            selected = feature_cols[: min(25, len(feature_cols))]
        all_selected_union.extend(selected)

        rows: List[Dict[str, Any]] = []
        for c in selected:
            x = pd.to_numeric(train[c], errors="coerce")
            full_corr = x.corr(y)
            roll_corr = x.rolling(rw).corr(y)
            valid = roll_corr.dropna()
            if valid.empty:
                continue
            mean_ic = float(valid.mean())
            median_ic = float(valid.median())
            ic_std = float(valid.std(ddof=0))
            ic_ir = float(mean_ic / ic_std) if ic_std > 1e-12 else 0.0
            pos = float((valid > 0).mean())
            neg = float((valid < 0).mean())
            rows.append(
                {
                    "feature": c,
                    "full_corr": _json_safe(full_corr),
                    "median_ic": _json_safe(median_ic),
                    "mean_ic": _json_safe(mean_ic),
                    "ic_std": _json_safe(ic_std),
                    "ic_ir": _json_safe(ic_ir),
                    "sign_consistency": _json_safe(max(pos, neg)),
                    "coverage": _json_safe(float(valid.size) / float(len(train))),
                }
            )
        rows = sorted(rows, key=lambda r: abs(float(r.get("median_ic") or 0.0)), reverse=True)[:tn]
        by_horizon[str(h)] = rows

    selected_unique = sorted(set(all_selected_union))
    col_pairs: List[Dict[str, Any]] = []
    if len(selected_unique) >= 2:
        cx = working[selected_unique].apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill")
        cmat = cx.corr(numeric_only=True).abs()
        for i, a in enumerate(selected_unique):
            for b in selected_unique[i + 1:]:
                v = cmat.at[a, b] if (a in cmat.index and b in cmat.columns) else np.nan
                if pd.notna(v) and float(v) >= th:
                    col_pairs.append({"feature_a": a, "feature_b": b, "abs_corr": float(v)})
        col_pairs = sorted(col_pairs, key=lambda r: float(r["abs_corr"]), reverse=True)[:100]

    return {
        "target_mode": target_mode,
        "test_days": test_days,
        "rolling_window": rw,
        "top_n": tn,
        "corr_threshold": th,
        "by_horizon": by_horizon,
        "collinearity_pairs": col_pairs,
    }


@app.post("/spot-analysis/flow-model/train")
def spot_analysis_flow_model_train(timeout_seconds: int = 600):
    return _run_flow_model_training(timeout_seconds=timeout_seconds)


def _run_flow_model_training(timeout_seconds: int = 600) -> Dict[str, Any]:
    script_path = ROOT / "scripts" / "data" / "train_flow_probability_models.py"
    if not script_path.exists():
        raise HTTPException(status_code=404, detail=f"Training script not found: {script_path}")

    cmd = [sys.executable, str(script_path)]
    started_at = datetime.now(timezone.utc)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=max(30, int(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "status": "timeout",
                "timeout_seconds": int(timeout_seconds),
                "stdout": (exc.stdout or "")[-4000:],
                "stderr": (exc.stderr or "")[-4000:],
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run training script: {exc}")

    ended_at = datetime.now(timezone.utc)
    artifacts_dir = ROOT / "artifacts" / "flow_spot_models"
    metrics_path = artifacts_dir / "metrics.json"
    diagnostics_path = artifacts_dir / "diagnostics.json"
    preds_path = artifacts_dir / "test_predictions.csv"
    live_signal_path = artifacts_dir / "live_signal.json"

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "return_code": result.returncode,
                "stdout": (result.stdout or "")[-4000:],
                "stderr": (result.stderr or "")[-4000:],
            },
        )

    metrics: Dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}

    return {
        "status": "ok",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "duration_seconds": (ended_at - started_at).total_seconds(),
        "command": " ".join(cmd),
        "stdout": (result.stdout or "")[-4000:],
        "stderr": (result.stderr or "")[-4000:],
        "artifacts": {
            "metrics": str(metrics_path),
            "diagnostics": str(diagnostics_path),
            "predictions": str(preds_path),
            "live_signal": str(live_signal_path),
        },
        "metrics": metrics,
    }


def _latest_participant_common_date() -> Optional[date]:
    nse_dir = ROOT / "database" / "nse_reports"
    if not nse_dir.exists():
        return None
    oi_paths = sorted(nse_dir.glob("fao_participant_oi_*.csv"))
    vol_paths = sorted(nse_dir.glob("fao_participant_vol_*.csv"))
    if not oi_paths or not vol_paths:
        return None
    oi_dates = {d for p in oi_paths if (d := _extract_date_from_participant_filename(p))}
    vol_dates = {d for p in vol_paths if (d := _extract_date_from_participant_filename(p))}
    common = sorted(oi_dates & vol_dates)
    return common[-1] if common else None


@app.post("/spot-analysis/flow-model/retrain-latest")
def spot_analysis_flow_model_retrain_latest(
    timeout_seconds: int = 900,
):
    latest_local = _latest_participant_common_date()
    train_payload = _run_flow_model_training(timeout_seconds=timeout_seconds)
    return {
        "status": "skipped",
        "reason": "cache_only_retrain",
        "note": "No data fetch performed. Training used local caches only.",
        "latest_local_date": latest_local.isoformat() if latest_local else None,
        "training": train_payload,
    }


def _fetch_nse_report_for_date(session: requests.Session, dt: date, timeout_seconds: float) -> Dict[str, Any]:
    api_url = "https://www.nseindia.com/api/reports"
    params = {
        "archives": json.dumps(_nse_reports_archives_payload(), separators=(",", ":")),
        "date": dt.strftime("%d-%b-%Y"),
        "type": "Archives",
    }
    response = session.get(api_url, params=params, timeout=timeout_seconds)
    raw_preview = response.content[:120]
    logger.info(
        (
            "[NSE_ARCHIVE][RESPONSE] date=%s status=%s content_type=%s "
            "content_encoding=%s content_length_header=%s bytes=%d preview_hex=%s"
        ),
        dt.isoformat(),
        response.status_code,
        response.headers.get("content-type"),
        response.headers.get("content-encoding"),
        response.headers.get("content-length"),
        len(response.content),
        raw_preview.hex(),
    )
    response.raise_for_status()
    content_type = (response.headers.get("content-type") or "").lower()
    is_zip = "application/zip" in content_type or response.content[:2] == b"PK"
    if is_zip:
        return {
            "format": "zip",
            "content_type": response.headers.get("content-type"),
            "body": response.content,
        }
    return {
        "format": "json",
        "content_type": response.headers.get("content-type"),
        "body": response.json(),
    }


def _fetch_nse_reports_range_impl(
    start_date: date,
    end_date: date,
    delay_seconds: float = 2.0,
    timeout_seconds: float = 10.0,
) -> Dict[str, Any]:
    dates = _iter_dates(start_date, end_date)
    output_dir = ROOT / "database" / "nse_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "user-agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "referer": "https://www.nseindia.com/reports",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-requested-with": "XMLHttpRequest",
        }
    )

    try:
        session.get("https://www.nseindia.com/reports", timeout=timeout_seconds)
    except Exception:
        logger.warning("NSE warmup request failed; continuing with direct API calls")

    results: List[Dict[str, Any]] = []
    for idx, dt in enumerate(dates):
        iso_day = dt.isoformat()
        start_ts = time.perf_counter()
        logger.info("[NSE_ARCHIVE] start date=%s", iso_day)
        try:
            payload = _fetch_nse_report_for_date(session=session, dt=dt, timeout_seconds=timeout_seconds)
            fmt = str(payload.get("format"))
            if fmt == "zip":
                with zipfile.ZipFile(io.BytesIO(payload["body"])) as zf:
                    members = [name for name in zf.namelist() if name and not name.endswith("/")]
                    zf.extractall(output_dir)
                saved_files = [str(output_dir / name) for name in members]
            else:
                file_path = output_dir / f"nse_{iso_day}.json"
                file_path.write_text(json.dumps(payload["body"], ensure_ascii=True), encoding="utf-8")
                saved_files = [str(file_path)]
            elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
            logger.info(
                "[NSE_ARCHIVE] finish date=%s status=ok format=%s files=%s elapsed_ms=%d",
                iso_day,
                fmt,
                saved_files,
                elapsed_ms,
            )
            results.append(
                {"date": iso_day, "status": "ok", "format": fmt, "files": saved_files, "elapsed_ms": elapsed_ms}
            )
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
            logger.exception("[NSE_ARCHIVE] finish date=%s status=error elapsed_ms=%d", iso_day, elapsed_ms)
            results.append({"date": iso_day, "status": "error", "error": str(exc), "elapsed_ms": elapsed_ms})

        if idx < len(dates) - 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

    ok_count = sum(1 for item in results if item.get("status") == "ok")
    return {
        "status": "completed",
        "total_dates": len(dates),
        "ok": ok_count,
        "failed": len(dates) - ok_count,
        "delay_seconds": delay_seconds,
        "timeout_seconds": timeout_seconds,
        "results": results,
    }


@app.post("/data-source/nse-reports/fetch-range")
def fetch_nse_reports_range(
    start_date: date,
    end_date: date,
    delay_seconds: float = 2.0,
    timeout_seconds: float = 10.0,
):
    return _fetch_nse_reports_range_impl(
        start_date=start_date,
        end_date=end_date,
        delay_seconds=delay_seconds,
        timeout_seconds=timeout_seconds,
    )


@app.post("/data-source/refresh/{name}")
def data_source_refresh(
    name: str,
    request: Request,
    previous_count: Optional[int] = None,
    next_count: Optional[int] = None,
    lookback_days: int = 1825,
):
    name = name.strip()
    if name not in {"nifty_ohlcv", "nifty_futures", "nifty_options_ce", "nifty_options_pe", "participants"}:
        raise HTTPException(status_code=400, detail="Unsupported dataset")

    if name == "participants":
        status = _participant_cache_status()
        today = datetime.now().date()
        latest_cached: Optional[date] = None
        latest_data_date = status.get("latest_data_date")
        if isinstance(latest_data_date, str):
            try:
                latest_cached = datetime.strptime(latest_data_date, "%Y-%m-%d").date()
            except Exception:
                latest_cached = None

        start_date = latest_cached if latest_cached is not None else (today - timedelta(days=7))
        if start_date > today:
            start_date = today

        result = _fetch_nse_reports_range_impl(
            start_date=start_date,
            end_date=today,
            delay_seconds=2.0,
            timeout_seconds=10.0,
        )
        result["dataset"] = "participants"
        result["start_date"] = start_date.isoformat()
        result["end_date"] = today.isoformat()
        result["mode"] = "from_last_update_to_today" if latest_cached is not None else "recent_7d_seed"
        return result

    if name == "nifty_ohlcv":
        creds = _get_credentials(request)
        kite = _get_kite_client(creds.api_key, creds.access_token)
        to_date = datetime.now()
        use_lookback_days = max(30, int(lookback_days))
        from_date = to_date - timedelta(days=use_lookback_days)
        instrument_token = 256265
        data = kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval="day"
        )
        df = pd.DataFrame(data)
        if df.empty:
            raise HTTPException(status_code=500, detail="No OHLCV data returned")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_DIR / "nifty_ohlcv.csv", index=False)
        return {
            "status": "ok",
            "rows": len(df),
            "lookback_days": use_lookback_days,
            "from_date": from_date.date().isoformat(),
            "to_date": to_date.date().isoformat(),
        }

    fetcher = NSEDataFetcher()
    today = datetime.now().date()

    # Default behavior preserved; callers can override via query params.
    default_previous = 5 if name == "nifty_futures" else 4
    default_next = 5 if name == "nifty_futures" else 3
    use_previous = default_previous if previous_count is None else int(previous_count)
    use_next = default_next if next_count is None else int(next_count)
    if use_previous < 0 or use_next < 0:
        raise HTTPException(status_code=400, detail="previous_count and next_count must be >= 0")
    if (use_previous + use_next) <= 0:
        raise HTTPException(status_code=400, detail="At least one of previous_count or next_count must be > 0")
    expiries = _build_expiry_list(previous_count=use_previous, next_count=use_next)

    if name == "nifty_futures":
        all_futures = []
        for expiry in expiries:
            expiry_date = expiry.date()
            window_from = expiry_date - timedelta(days=60)
            window_to = min(expiry_date, today)
            from_date_str = window_from.strftime('%d-%m-%Y')
            to_date_str = window_to.strftime('%d-%m-%Y')
            raw = fetcher.fetch_futures_data(
                from_date=from_date_str,
                to_date=to_date_str,
                symbol="NIFTY",
                expiry_str=expiry,
                year=expiry.year
            )
            if raw and 'data' in raw and len(raw['data']) > 0:
                df_expiry = fetcher.parse_futures_data(raw)
                if not df_expiry.empty:
                    all_futures.append(df_expiry)
            time.sleep(1)
        if not all_futures:
            raise HTTPException(status_code=500, detail="No futures data fetched")
        df = pd.concat(all_futures, ignore_index=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / "nifty_futures.csv"
        merged = _append_csv_dedup(
            cache_path,
            df,
            dedup_cols=["date", "expiry_date", "symbol", "instrument_type"],
        )
        merged.to_csv(cache_path, index=False)
        return {
            "status": "ok",
            "fetched_rows": len(df),
            "total_rows": len(merged),
            "previous_count": use_previous,
            "next_count": use_next,
            "expiries": [e.strftime("%Y-%m-%d") for e in expiries],
        }

    opt_type = "CE" if name == "nifty_options_ce" else "PE"
    all_options = []
    for expiry in expiries:
        expiry_date = expiry.date()
        window_from = expiry_date - timedelta(days=60)
        window_to = min(expiry_date, today)
        from_date_str = window_from.strftime('%d-%m-%Y')
        to_date_str = window_to.strftime('%d-%m-%Y')
        raw = fetcher.fetch_options_data(
            from_date=from_date_str,
            to_date=to_date_str,
            symbol="NIFTY",
            expiry_str=expiry,
            option_type=opt_type,
            year=expiry.year
        )
        if raw and 'data' in raw and len(raw['data']) > 0:
            df_expiry = fetcher.parse_options_data(raw)
            if not df_expiry.empty:
                all_options.append(df_expiry)
        time.sleep(1)
    if not all_options:
        raise HTTPException(status_code=500, detail="No options data fetched")
    df = pd.concat(all_options, ignore_index=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_DIR / f"{name}.csv", index=False)
    return {
        "status": "ok",
        "rows": len(df),
        "previous_count": use_previous,
        "next_count": use_next,
        "expiries": [e.strftime("%Y-%m-%d") for e in expiries],
    }


if FRONTEND_DIST.exists():
    logger.info("[static:init] mounting StaticFiles at '/' from dist=%s", FRONTEND_DIST)
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="spa")
else:
    logger.warning("[static:init] dist folder missing at startup: %s", FRONTEND_DIST)
