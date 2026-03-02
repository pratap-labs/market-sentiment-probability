from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, mean_pinball_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[2]
NSE_REPORTS_DIR = ROOT / "database" / "nse_reports"
NIFTY_CACHE_FILE = ROOT / "database" / "derivatives_cache" / "nifty_ohlcv.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "flow_spot_models"

PARTICIPANTS = ("fii", "dii", "pro", "client")
HORIZONS = (1, 3, 5, 10)
TARGET_MODES = ("point_return", "forward_mean_return")
TEST_DAYS = 10


@dataclass
class HorizonResult:
    horizon: int
    train_rows: int
    test_rows: int
    auc: float | None
    brier: float | None
    pinball_q10: float
    pinball_q50: float
    pinball_q90: float
    wf_folds: int
    wf_auc: float | None
    wf_brier: float | None
    wf_pinball_q10: float | None
    wf_pinball_q50: float | None
    wf_pinball_q90: float | None
    selected_feature_count: int
    selected_features: List[str]


def _parse_date_from_name(path: Path) -> datetime.date | None:
    token = path.stem.split("_")[-1]
    if len(token) != 8 or not token.isdigit():
        return None
    try:
        return datetime.strptime(token, "%d%m%Y").date()
    except Exception:
        return None


def _to_num(x: object) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if not s:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _parse_participant_file(path: Path) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(path, skiprows=1)
    df.columns = [str(c).strip() for c in df.columns]
    if "Client Type" not in df.columns:
        raise ValueError(f"Unexpected format in {path}")

    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        raw = str(row.get("Client Type", "")).strip().upper()
        if raw == "TOTAL" or not raw:
            continue
        key = "pro" if raw == "PRO" else raw.lower()
        if key not in PARTICIPANTS:
            continue
        fut_long = _to_num(row.get("Future Index Long"))
        fut_short = _to_num(row.get("Future Index Short"))
        call_long = _to_num(row.get("Option Index Call Long"))
        call_short = _to_num(row.get("Option Index Call Short"))
        put_long = _to_num(row.get("Option Index Put Long"))
        put_short = _to_num(row.get("Option Index Put Short"))
        out[key] = {
            "fut_oi_net": fut_long - fut_short,
            "call_oi_net": call_long - call_short,
            "put_oi_net": put_long - put_short,
        }
    return out


def _load_flow_frame() -> pd.DataFrame:
    oi_files = sorted(NSE_REPORTS_DIR.glob("fao_participant_oi_*.csv"))
    by_date: Dict[datetime.date, Dict[str, Dict[str, float]]] = {}
    for p in oi_files:
        d = _parse_date_from_name(p)
        if d is None:
            continue
        by_date[d] = _parse_participant_file(p)

    rows: List[Dict[str, float]] = []
    for d in sorted(by_date.keys()):
        record: Dict[str, float] = {"date": pd.Timestamp(d)}
        for part in PARTICIPANTS:
            vals = by_date[d].get(part, {})
            record[f"{part}_fut_oi_net"] = _to_num(vals.get("fut_oi_net"))
            record[f"{part}_call_oi_net"] = _to_num(vals.get("call_oi_net"))
            record[f"{part}_put_oi_net"] = _to_num(vals.get("put_oi_net"))
        rows.append(record)
    if not rows:
        raise RuntimeError(f"No OI files found in {NSE_REPORTS_DIR}")
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _load_nifty_close() -> pd.DataFrame:
    if not NIFTY_CACHE_FILE.exists():
        raise RuntimeError(f"NIFTY cache not found: {NIFTY_CACHE_FILE}")
    df = pd.read_csv(NIFTY_CACHE_FILE)
    if "date" not in df.columns or "close" not in df.columns:
        raise RuntimeError("nifty_ohlcv.csv missing date/close columns")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"])[["date", "close"]].sort_values("date").drop_duplicates("date", keep="last")
    return df.reset_index(drop=True)


def _build_dataset() -> pd.DataFrame:
    flow = _load_flow_frame()
    nifty = _load_nifty_close()
    df = flow.merge(nifty, on="date", how="inner").sort_values("date").reset_index(drop=True)

    # Confirmed fields: OI changes and OI levels, no cash/amount features.
    for part in PARTICIPANTS:
        df[f"{part}_fut_oi_chg"] = df[f"{part}_fut_oi_net"].diff()
        df[f"{part}_call_oi_chg"] = df[f"{part}_call_oi_net"].diff()
        df[f"{part}_put_oi_chg"] = df[f"{part}_put_oi_net"].diff()
        df[f"{part}_call_put_imb"] = df[f"{part}_call_oi_chg"] - df[f"{part}_put_oi_chg"]

    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_3d"] = df["close"].pct_change(3)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_10d"] = df["close"].pct_change(10)
    df["ret_20d"] = df["close"].pct_change(20)
    df["rv_20d"] = df["ret_1d"].rolling(20).std()

    base_cols = [c for c in df.columns if c.endswith("_chg") or c.endswith("_net") or c.endswith("_imb")]
    base_cols += ["ret_1d", "ret_3d", "ret_5d", "ret_10d", "ret_20d", "rv_20d"]
    base_cols = sorted(set(base_cols))
    lag_blocks: List[pd.DataFrame] = []
    for lag in range(1, 4):
        block = {f"{col}_lag{lag}": df[col].shift(lag) for col in base_cols}
        lag_blocks.append(pd.DataFrame(block))
    if lag_blocks:
        df = pd.concat([df] + lag_blocks, axis=1)

    return df


def _feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"date", "close"}
    exclude |= {f"target_ret_{h}" for h in HORIZONS}
    exclude |= {f"target_up_{h}" for h in HORIZONS}
    # Use the full engineered feature set for both modeling and diagnostics.
    return [
        c
        for c in df.columns
        if (
            c not in exclude
            and not c.startswith("target_")
        )
    ]


def _make_targets(df: pd.DataFrame, horizon: int, target_mode: str = "forward_mean_return") -> pd.DataFrame:
    out = df.copy()
    if target_mode == "point_return":
        out[f"target_ret_{horizon}"] = out["close"].shift(-horizon) / out["close"] - 1.0
    else:
        # Smoothed forward target: mean(close[t+1..t+h]) relative to close[t].
        future_closes = [out["close"].shift(-k) for k in range(1, horizon + 1)]
        out[f"target_ret_{horizon}"] = (pd.concat(future_closes, axis=1).mean(axis=1) / out["close"]) - 1.0
    out[f"target_up_{horizon}"] = (out[f"target_ret_{horizon}"] > 0).astype(float)
    return out


def _split_last_n(df: pd.DataFrame, n_test: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= n_test:
        raise RuntimeError(f"Not enough rows ({len(df)}) for test size {n_test}")
    return df.iloc[:-n_test].copy(), df.iloc[-n_test:].copy()


def _build_diagnostics(df: pd.DataFrame, feature_cols: List[str], target_mode: str = "forward_mean_return") -> Dict[str, object]:
    core_cols = [
        c for c in feature_cols
        if ("_lag" not in c) and ("ret_" in c or c.endswith("_chg") or c.endswith("_net") or c.endswith("_imb"))
    ]
    core_cols = core_cols[:24]
    corr_df = df[core_cols].copy() if core_cols else pd.DataFrame()
    corr_matrix = corr_df.corr(numeric_only=True) if not corr_df.empty else pd.DataFrame()

    x = df[feature_cols].to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x = np.where(np.isfinite(x), x, med)
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=min(12, x.shape[1], x.shape[0]))
    pca.fit(x)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    top_by_horizon: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    var_explained_by_horizon: Dict[str, List[Dict[str, float | int | str]]] = {}
    for h in HORIZONS:
        d = _make_targets(df, h, target_mode=target_mode).dropna(subset=[f"target_ret_{h}"]).copy()
        if d.empty:
            continue
        train_d, _ = _split_last_n(d, TEST_DAYS)
        if train_d.empty:
            continue

        ret_corr: List[Tuple[str, float]] = []
        up_corr: List[Tuple[str, float]] = []
        for c in feature_cols:
            xv = pd.to_numeric(train_d[c], errors="coerce")
            yr = pd.to_numeric(train_d[f"target_ret_{h}"], errors="coerce")
            yu = pd.to_numeric(train_d[f"target_up_{h}"], errors="coerce")
            if xv.notna().sum() < 10:
                continue
            cr = xv.corr(yr)
            cu = xv.corr(yu)
            if pd.notna(cr):
                ret_corr.append((c, float(cr)))
            if pd.notna(cu):
                up_corr.append((c, float(cu)))

        ret_corr = sorted(ret_corr, key=lambda t: abs(t[1]), reverse=True)[:12]
        up_corr = sorted(up_corr, key=lambda t: abs(t[1]), reverse=True)[:12]
        top_by_horizon[str(h)] = {
            "target_ret": [{"feature": f, "corr": v} for f, v in ret_corr],
            "target_up": [{"feature": f, "corr": v} for f, v in up_corr],
        }

        # Feature-to-spot variance explained using linear R^2 (train window only).
        # Individual R^2: one feature at a time.
        # Cumulative R^2: forward-add top features (sorted by individual R^2).
        y = pd.to_numeric(train_d[f"target_ret_{h}"], errors="coerce")
        y = y.fillna(y.median() if y.notna().any() else 0.0)
        x_df = train_d[feature_cols].apply(pd.to_numeric, errors="coerce")
        x_df = x_df.fillna(x_df.median(numeric_only=True)).fillna(0.0)

        uni_scores: List[Tuple[str, float]] = []
        for c in feature_cols:
            xv = x_df[c].to_numpy(dtype=float).reshape(-1, 1)
            if np.nanstd(xv) <= 1e-12:
                uni_scores.append((c, 0.0))
                continue
            try:
                lr = LinearRegression()
                lr.fit(xv, y)
                r2 = float(lr.score(xv, y))
                if not np.isfinite(r2):
                    r2 = 0.0
                uni_scores.append((c, max(0.0, r2)))
            except Exception:
                uni_scores.append((c, 0.0))

        ranked = sorted(uni_scores, key=lambda t: t[1], reverse=True)
        running_features: List[str] = []
        rows: List[Dict[str, float | int | str]] = []
        prev_cum = 0.0
        for i, (feat, ind_r2) in enumerate(ranked[:25], start=1):
            running_features.append(feat)
            try:
                lr = LinearRegression()
                lr.fit(x_df[running_features], y)
                cum_r2 = float(lr.score(x_df[running_features], y))
                if not np.isfinite(cum_r2):
                    cum_r2 = prev_cum
            except Exception:
                cum_r2 = prev_cum
            cum_r2 = max(prev_cum, max(0.0, cum_r2))
            rows.append(
                {
                    "rank": i,
                    "feature": feat,
                    "individual_r2": float(ind_r2),
                    "cumulative_r2": float(cum_r2),
                    "marginal_r2": float(max(0.0, cum_r2 - prev_cum)),
                }
            )
            prev_cum = cum_r2
        var_explained_by_horizon[str(h)] = rows

    return {
        "core_features": core_cols,
        "corr_matrix": {
            "features": corr_matrix.columns.tolist(),
            "values": corr_matrix.values.tolist() if not corr_matrix.empty else [],
        },
        "pca": {
            "components": [f"PC{i+1}" for i in range(len(evr))],
            "explained_variance_ratio": evr.tolist(),
            "cumulative_explained_variance": cum.tolist(),
        },
        "top_feature_correlations": top_by_horizon,
        "spot_variance_explained": {
            "target": "future_spot_return",
            "method": "linear_r2_train_window",
            "by_horizon": var_explained_by_horizon,
        },
    }


def _safe_tss_splits(n_rows: int, min_fold_size: int = 12, max_splits: int = 5) -> int:
    if n_rows < (min_fold_size * 2):
        return 2 if n_rows >= (min_fold_size + 6) else 0
    return max(2, min(max_splits, n_rows // min_fold_size))


def _select_stable_features(data: pd.DataFrame, feat_cols: List[str], target_col: str, max_features: int = 20) -> List[str]:
    if not feat_cols:
        return []
    d = data.dropna(subset=[target_col]).copy()
    if d.empty:
        return feat_cols[:max_features]
    if len(d) < 32:
        return feat_cols[:max_features]

    n_splits = _safe_tss_splits(len(d), min_fold_size=12, max_splits=6)
    if n_splits == 0:
        return feat_cols[:max_features]
    tss = TimeSeriesSplit(n_splits=n_splits)

    scores: Dict[str, List[float]] = {c: [] for c in feat_cols}
    for _, val_idx in tss.split(d):
        val = d.iloc[val_idx]
        yv = pd.to_numeric(val[target_col], errors="coerce")
        for c in feat_cols:
            xv = pd.to_numeric(val[c], errors="coerce")
            corr = xv.corr(yv)
            if pd.notna(corr):
                scores[c].append(float(abs(corr)))

    ranked: List[Tuple[str, float, float, float]] = []
    for c in feat_cols:
        arr = np.array(scores[c], dtype=float)
        if arr.size == 0:
            continue
        mean_abs = float(np.nanmean(arr))
        std_abs = float(np.nanstd(arr))
        stability = mean_abs / (std_abs + 1e-6)
        ranked.append((c, stability, mean_abs, std_abs))
    if not ranked:
        return feat_cols[:max_features]

    ranked = sorted(ranked, key=lambda t: (t[1], t[2]), reverse=True)
    return [c for c, _, _, _ in ranked[:max_features]]


def _fit_predict_prob(x_train: pd.DataFrame, y_up_train: pd.Series, x_pred: pd.DataFrame) -> np.ndarray:
    y_vals = pd.Series(y_up_train).astype(int)
    if y_vals.nunique() < 2:
        return np.full(len(x_pred), float(y_vals.iloc[-1]) if len(y_vals) else 0.5, dtype=float)

    n_splits = _safe_tss_splits(len(x_train), min_fold_size=20, max_splits=4)
    base_clf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("logit", LogisticRegression(max_iter=4000, class_weight="balanced", C=0.08)),
        ]
    )
    hgb_clf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("hgb", HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=180, random_state=42)),
        ]
    )

    probs: List[np.ndarray] = []
    if n_splits >= 2 and n_splits < len(x_train):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        try:
            c1 = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=tscv)
            c1.fit(x_train, y_vals)
            probs.append(c1.predict_proba(x_pred)[:, 1])
        except Exception:
            base_clf.fit(x_train, y_vals)
            probs.append(base_clf.predict_proba(x_pred)[:, 1])
        try:
            c2 = CalibratedClassifierCV(estimator=hgb_clf, method="sigmoid", cv=tscv)
            c2.fit(x_train, y_vals)
            probs.append(c2.predict_proba(x_pred)[:, 1])
        except Exception:
            hgb_clf.fit(x_train, y_vals)
            probs.append(hgb_clf.predict_proba(x_pred)[:, 1])
    else:
        base_clf.fit(x_train, y_vals)
        probs.append(base_clf.predict_proba(x_pred)[:, 1])
        hgb_clf.fit(x_train, y_vals)
        probs.append(hgb_clf.predict_proba(x_pred)[:, 1])

    if not probs:
        return np.full(len(x_pred), 0.5, dtype=float)
    return np.mean(np.vstack(probs), axis=0)


def _fit_predict_quantiles(x_train: pd.DataFrame, y_ret_train: pd.Series, x_pred: pd.DataFrame) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, alpha in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)):
        q_model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "gbm",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=alpha,
                        n_estimators=250,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        )
        q_model.fit(x_train, y_ret_train)
        out[name] = q_model.predict(x_pred)
    return out


def _walk_forward_eval(
    df: pd.DataFrame,
    horizon: int,
    feat_cols: List[str],
    target_mode: str = "forward_mean_return",
) -> Dict[str, float | int | None]:
    data = _make_targets(df, horizon, target_mode=target_mode).dropna(subset=[f"target_ret_{horizon}"]).copy()
    if len(data) <= (TEST_DAYS + 30):
        return {"folds": 0, "auc": None, "brier": None, "pinball_q10": None, "pinball_q50": None, "pinball_q90": None}
    pretest = data.iloc[:-TEST_DAYS].copy()
    n_splits = _safe_tss_splits(len(pretest), min_fold_size=12, max_splits=5)
    if n_splits < 2:
        return {"folds": 0, "auc": None, "brier": None, "pinball_q10": None, "pinball_q50": None, "pinball_q90": None}

    aucs: List[float] = []
    briers: List[float] = []
    p10s: List[float] = []
    p50s: List[float] = []
    p90s: List[float] = []
    folds = 0
    tss = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in tss.split(pretest):
        train_df = pretest.iloc[train_idx]
        val_df = pretest.iloc[val_idx]
        if train_df.empty or val_df.empty:
            continue

        x_train = train_df[feat_cols]
        x_val = val_df[feat_cols]
        y_up_train = train_df[f"target_up_{horizon}"].astype(int)
        y_up_val = val_df[f"target_up_{horizon}"].astype(int)
        y_ret_train = train_df[f"target_ret_{horizon}"]
        y_ret_val = val_df[f"target_ret_{horizon}"]

        p_up = _fit_predict_prob(x_train, y_up_train, x_val)
        q_preds = _fit_predict_quantiles(x_train, y_ret_train, x_val)
        if len(np.unique(y_up_val)) > 1:
            aucs.append(float(roc_auc_score(y_up_val, p_up)))
        briers.append(float(brier_score_loss(y_up_val, p_up)))
        p10s.append(float(mean_pinball_loss(y_ret_val, q_preds["q10"], alpha=0.1)))
        p50s.append(float(mean_pinball_loss(y_ret_val, q_preds["q50"], alpha=0.5)))
        p90s.append(float(mean_pinball_loss(y_ret_val, q_preds["q90"], alpha=0.9)))
        folds += 1

    return {
        "folds": folds,
        "auc": float(np.mean(aucs)) if aucs else None,
        "brier": float(np.mean(briers)) if briers else None,
        "pinball_q10": float(np.mean(p10s)) if p10s else None,
        "pinball_q50": float(np.mean(p50s)) if p50s else None,
        "pinball_q90": float(np.mean(p90s)) if p90s else None,
    }


def _train_for_horizon(
    df: pd.DataFrame,
    horizon: int,
    feat_cols: List[str],
    target_mode: str = "forward_mean_return",
) -> Tuple[HorizonResult, pd.DataFrame]:
    data = _make_targets(df, horizon, target_mode=target_mode)
    data = data.dropna(subset=[f"target_ret_{horizon}"]).copy()
    train_df, test_df = _split_last_n(data, TEST_DAYS)

    x_train = train_df[feat_cols]
    x_test = test_df[feat_cols]
    y_up_train = train_df[f"target_up_{horizon}"].astype(int)
    y_up_test = test_df[f"target_up_{horizon}"].astype(int)
    y_ret_train = train_df[f"target_ret_{horizon}"]
    y_ret_test = test_df[f"target_ret_{horizon}"]

    p_up = _fit_predict_prob(x_train, y_up_train, x_test)
    q_preds = _fit_predict_quantiles(x_train, y_ret_train, x_test)

    auc = None
    if len(np.unique(y_up_test)) > 1:
        auc = float(roc_auc_score(y_up_test, p_up))
    brier = float(brier_score_loss(y_up_test, p_up))
    pin10 = float(mean_pinball_loss(y_ret_test, q_preds["q10"], alpha=0.1))
    pin50 = float(mean_pinball_loss(y_ret_test, q_preds["q50"], alpha=0.5))
    pin90 = float(mean_pinball_loss(y_ret_test, q_preds["q90"], alpha=0.9))

    out = test_df[["date", "close"]].copy()
    out["horizon"] = horizon
    out["actual_ret"] = y_ret_test.values
    out["p_up"] = p_up
    out["q10_ret"] = q_preds["q10"]
    out["q50_ret"] = q_preds["q50"]
    out["q90_ret"] = q_preds["q90"]
    out["pred_close_q10"] = out["close"] * (1.0 + out["q10_ret"])
    out["pred_close_q50"] = out["close"] * (1.0 + out["q50_ret"])
    out["pred_close_q90"] = out["close"] * (1.0 + out["q90_ret"])

    return (
        HorizonResult(
            horizon=horizon,
            train_rows=len(train_df),
            test_rows=len(test_df),
            auc=auc,
            brier=brier,
            pinball_q10=pin10,
            pinball_q50=pin50,
            pinball_q90=pin90,
            wf_folds=0,
            wf_auc=None,
            wf_brier=None,
            wf_pinball_q10=None,
            wf_pinball_q50=None,
            wf_pinball_q90=None,
            selected_feature_count=len(feat_cols),
            selected_features=list(feat_cols),
        ),
        out,
    )


def _latest_signal_for_horizon(
    df: pd.DataFrame,
    horizon: int,
    feat_cols: List[str],
    target_mode: str = "forward_mean_return",
) -> Dict[str, float | int | str | None]:
    data = _make_targets(df, horizon, target_mode=target_mode)
    train_df = data.dropna(subset=[f"target_ret_{horizon}"]).copy()
    latest = df.tail(1).copy()
    if train_df.empty or latest.empty:
        return {"horizon": horizon}

    x_train = train_df[feat_cols]
    y_up_train = train_df[f"target_up_{horizon}"].astype(int)
    y_ret_train = train_df[f"target_ret_{horizon}"]
    x_latest = latest[feat_cols]

    p_up = float(_fit_predict_prob(x_train, y_up_train, x_latest)[0])
    q_raw = _fit_predict_quantiles(x_train, y_ret_train, x_latest)
    q_preds: Dict[str, float] = {k: float(v[0]) for k, v in q_raw.items()}

    close = float(latest["close"].iloc[0])
    as_of = pd.to_datetime(latest["date"].iloc[0], errors="coerce")
    return {
        "horizon": horizon,
        "as_of_date": as_of.date().isoformat() if pd.notna(as_of) else None,
        "close": close,
        "p_up": p_up,
        "pred_ret_q10": q_preds["q10"],
        "pred_ret_q50": q_preds["q50"],
        "pred_ret_q90": q_preds["q90"],
        "pred_close_q10": close * (1.0 + q_preds["q10"]),
        "pred_close_q50": close * (1.0 + q_preds["q50"]),
        "pred_close_q90": close * (1.0 + q_preds["q90"]),
        "signal": "BULLISH" if p_up >= 0.55 else ("BEARISH" if p_up <= 0.45 else "NEUTRAL"),
    }


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    df = _build_dataset()
    feature_cols = _feature_columns(df)
    working = df.dropna(subset=feature_cols).reset_index(drop=True)
    active_target_mode = "forward_mean_return"

    all_results: List[HorizonResult] = []
    pred_frames: List[pd.DataFrame] = []
    selected_by_h: Dict[int, List[str]] = {}
    wf_by_h: Dict[int, Dict[str, float | int | None]] = {}
    for h in HORIZONS:
        target_col = f"target_ret_{h}"
        tmp = _make_targets(working, h, target_mode=active_target_mode).dropna(subset=[target_col]).copy()
        tmp_pretest = tmp.iloc[:-TEST_DAYS].copy() if len(tmp) > TEST_DAYS else tmp
        selected = _select_stable_features(tmp_pretest, feature_cols, target_col=target_col, max_features=20)
        if not selected:
            selected = feature_cols[:20]
        selected_by_h[h] = selected
        wf = _walk_forward_eval(working, h, selected, target_mode=active_target_mode)
        wf_by_h[h] = wf

        res, preds = _train_for_horizon(working, h, selected, target_mode=active_target_mode)
        res.wf_folds = int(wf.get("folds", 0) or 0)
        res.wf_auc = wf.get("auc")
        res.wf_brier = wf.get("brier")
        res.wf_pinball_q10 = wf.get("pinball_q10")
        res.wf_pinball_q50 = wf.get("pinball_q50")
        res.wf_pinball_q90 = wf.get("pinball_q90")
        res.selected_feature_count = len(selected)
        res.selected_features = list(selected)
        all_results.append(res)
        pred_frames.append(preds)

    mode_comparison: List[Dict[str, object]] = []
    for mode in TARGET_MODES:
        rows: List[Dict[str, object]] = []
        for h in HORIZONS:
            target_col = f"target_ret_{h}"
            tmp = _make_targets(working, h, target_mode=mode).dropna(subset=[target_col]).copy()
            tmp_pretest = tmp.iloc[:-TEST_DAYS].copy() if len(tmp) > TEST_DAYS else tmp
            selected = _select_stable_features(tmp_pretest, feature_cols, target_col=target_col, max_features=20)
            if not selected:
                selected = feature_cols[:20]
            wf = _walk_forward_eval(working, h, selected, target_mode=mode)
            res, _ = _train_for_horizon(working, h, selected, target_mode=mode)
            rows.append(
                {
                    "horizon": h,
                    "auc": res.auc,
                    "brier": res.brier,
                    "pinball_q10": res.pinball_q10,
                    "pinball_q50": res.pinball_q50,
                    "pinball_q90": res.pinball_q90,
                    "wf_folds": int(wf.get("folds", 0) or 0),
                    "wf_auc": wf.get("auc"),
                    "wf_brier": wf.get("brier"),
                }
            )
        mode_comparison.append({"target_mode": mode, "horizons": rows})

    metrics_payload = {
        "test_days": TEST_DAYS,
        "target_mode": active_target_mode,
        "rows_after_feature_filter": int(len(working)),
        "features_count": int(len(feature_cols)),
        "mode_comparison": mode_comparison,
        "horizons": [
            {
                "horizon": r.horizon,
                "train_rows": r.train_rows,
                "test_rows": r.test_rows,
                "auc": r.auc,
                "brier": r.brier,
                "pinball_q10": r.pinball_q10,
                "pinball_q50": r.pinball_q50,
                "pinball_q90": r.pinball_q90,
                "wf_folds": r.wf_folds,
                "wf_auc": r.wf_auc,
                "wf_brier": r.wf_brier,
                "wf_pinball_q10": r.wf_pinball_q10,
                "wf_pinball_q50": r.wf_pinball_q50,
                "wf_pinball_q90": r.wf_pinball_q90,
                "selected_feature_count": r.selected_feature_count,
                "selected_features": r.selected_features,
            }
            for r in all_results
        ],
        "feature_columns": feature_cols,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    diagnostics_payload = _build_diagnostics(working, feature_cols, target_mode=active_target_mode)
    (ARTIFACT_DIR / "diagnostics.json").write_text(json.dumps(diagnostics_payload, indent=2), encoding="utf-8")

    preds_df = pd.concat(pred_frames, ignore_index=True).sort_values(["horizon", "date"])
    preds_df.to_csv(ARTIFACT_DIR / "test_predictions.csv", index=False)
    live_rows = [
        _latest_signal_for_horizon(
            working,
            h,
            selected_by_h.get(h, feature_cols[:20]),
            target_mode=active_target_mode,
        )
        for h in HORIZONS
    ]
    (ARTIFACT_DIR / "live_signal.json").write_text(
        json.dumps({"rows": live_rows}, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {ARTIFACT_DIR / 'metrics.json'}")
    print(f"Wrote {ARTIFACT_DIR / 'diagnostics.json'}")
    print(f"Wrote {ARTIFACT_DIR / 'test_predictions.csv'}")
    print(f"Wrote {ARTIFACT_DIR / 'live_signal.json'}")
    for r in all_results:
        print(
            f"h={r.horizon} train={r.train_rows} test={r.test_rows} "
            f"auc={r.auc} brier={r.brier:.6f} "
            f"pin(q10/q50/q90)=({r.pinball_q10:.6f}/{r.pinball_q50:.6f}/{r.pinball_q90:.6f})"
        )


if __name__ == "__main__":
    main()
