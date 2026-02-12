from __future__ import annotations

import logging
import os
import time
import calendar
import math
import numpy as np
from datetime import date
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scripts.utils import (
    calculate_market_regime,
    calculate_portfolio_greeks,
    calculate_stress_pnl,
    calculate_var,
    enrich_position_with_greeks,
)
from scripts.data import NSEDataFetcher
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

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "database" / "derivatives_cache"
KITE_CREDS_FILE = CACHE_DIR / "kite_credentials.json"
POSITIONS_CACHE_FILE = CACHE_DIR / "positions_cache.json"
EQUITIES_CACHE_FILE = CACHE_DIR / "equities_holdings.json"
MANUAL_BUCKET_FILE = ROOT / "database" / "manual_bucket_overrides.json"
RISK_BUCKET_SETTINGS_FILE = ROOT / "database" / "risk_bucket_settings.json"
FRONTEND_DIST = ROOT / "frontend" / "dist"
KITE_TOKEN_TTL = timedelta(hours=12)
DEFAULT_REDIRECT_URL = "http://localhost:8000/auth/callback"
DEFAULT_FRONTEND_URL = "http://localhost:5173/login?auth=success"

logger = logging.getLogger("gammashield.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="GammaShield API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    rows = df.tail(limit).to_dict(orient="records")
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


@app.get("/equities/holdings")
def equities_holdings(request: Request):
    creds = _get_credentials(request)
    kite = _get_kite_client(creds.api_key, creds.access_token)
    holdings = kite.holdings() or []
    return {"holdings": holdings}


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

    total_capital = float(
        (kite.margins().get("equity", {}).get("available", {}).get("live_balance", 0.0))
        + (kite.margins().get("equity", {}).get("utilised", {}).get("debits", 0.0))
    )
    if total_capital <= 0:
        total_capital = 1_750_000.0

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
            "total_loss_limit": (4.0 / 100.0) * float(total_capital) if total_capital else None,
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
        "portfolio_es99_inr": float(agg["portfolio_es99_inr"]),
        "portfolio_es99_pct": (float(agg["portfolio_es99_inr"]) / total_capital * 100) if total_capital else 0.0,
        "margin_used_pct": None,
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
        return {
            "name": name,
            "exists": exists,
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None,
            "size_bytes": path.stat().st_size if exists else 0,
        }

    return {
        "nifty_ohlcv": _status("nifty_ohlcv"),
        "nifty_futures": _status("nifty_futures"),
        "nifty_options_ce": _status("nifty_options_ce"),
        "nifty_options_pe": _status("nifty_options_pe"),
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


def _build_expiry_list(count: int = 3) -> List[datetime]:
    today = datetime.now()
    expiries: List[datetime] = []
    year = today.year
    month = today.month
    while len(expiries) < count:
        expiry = _last_tuesday(year, month)
        if expiry >= today:
            expiries.append(expiry)
        month += 1
        if month > 12:
            month = 1
            year += 1
    unique_expiries = sorted({expiry.date(): expiry for expiry in expiries}.values())
    return unique_expiries


@app.post("/data-source/refresh/{name}")
def data_source_refresh(name: str, request: Request):
    name = name.strip()
    if name not in {"nifty_ohlcv", "nifty_futures", "nifty_options_ce", "nifty_options_pe"}:
        raise HTTPException(status_code=400, detail="Unsupported dataset")

    if name == "nifty_ohlcv":
        creds = _get_credentials(request)
        kite = _get_kite_client(creds.api_key, creds.access_token)
        to_date = datetime.now()
        from_date = to_date - timedelta(days=730)
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
        return {"status": "ok", "rows": len(df)}

    fetcher = NSEDataFetcher()
    expiries = _build_expiry_list()
    to_date = datetime.now()
    from_date = to_date - timedelta(days=90)
    from_date_str = from_date.strftime('%d-%m-%Y')
    to_date_str = to_date.strftime('%d-%m-%Y')

    if name == "nifty_futures":
        all_futures = []
        for expiry in expiries:
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
        df.to_csv(CACHE_DIR / "nifty_futures.csv", index=False)
        return {"status": "ok", "rows": len(df)}

    opt_type = "CE" if name == "nifty_options_ce" else "PE"
    all_options = []
    for expiry in expiries:
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
    return {"status": "ok", "rows": len(df)}


if FRONTEND_DIST.exists():
    @app.get("/")
    def frontend_index():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}")
    def frontend_spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
