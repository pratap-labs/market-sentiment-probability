import { useEffect, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import DataTable from "../components/DataTable";
import Plot from "../components/Plot";
import { apiFetch } from "../api/client";
import { useCachedApi } from "../hooks/useCachedApi";
import { formatPct, formatNumber } from "../components/format";
import { useNotifications } from "../state/NotificationContext";
import { useControls } from "../state/ControlsContext";

function formatInrLac(value: unknown): string {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return `₹${(num / 100000).toFixed(2)}L`;
}

function toLac(value: unknown): number {
  const num = Number(value);
  if (!Number.isFinite(num)) return 0;
  return num / 100000;
}

function pct(arr: number[], p: number): number {
  if (!arr.length) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.floor((p / 100) * (s.length - 1))));
  return s[idx];
}

function avg(arr: number[]): number {
  if (!arr.length) return NaN;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

type OverlayCdfRow = { engine?: string; x?: number[]; cdf?: number[]; var99?: number };
type OverlayTailRow = { engine?: string; x?: number[]; cdf?: number[]; ccdf?: number[]; es99?: number };
type SpotPathsSample = {
  days?: number[];
  paths?: number[][];
  path_idx?: number[];
  terminal_pnl?: number[];
  max_drawdown?: number[];
  drawdown_series?: number[][];
};
type ModePayload = { kpis?: Record<string, unknown>; terminal_pnl_sample?: number[] };
type EnginePayload = {
  fit_kpis?: Record<string, unknown>;
  modes?: Record<string, ModePayload>;
  spot_paths_sample?: SpotPathsSample;
  spot_paths_worst?: SpotPathsSample | null;
  spot_paths_worst_all?: SpotPathsSample | null;
  spot_paths_worst_dd?: SpotPathsSample | null;
  spot_paths_worst_mode?: string;
};

export type PortfolioRB = {
  advanced_simulation?: {
    model_version?: string;
    summary_rows?: Record<string, unknown>[];
    engines?: Record<string, EnginePayload>;
    config?: { horizon_days?: number };
    overlay_plots?: {
      cdf?: OverlayCdfRow[];
      tail?: OverlayTailRow[];
      tail_xlim?: number[] | null;
    };
  } | null;
  account_size?: number;
  margin_used?: number;
  margin_used_pct?: number | null;
  zone?: {
    theta_norm?: number;
    gamma_norm?: number;
    vega_norm?: number;
  };
};
type Ohlcv = { rows?: Record<string, unknown>[] };
type SettingsConfig = { portfolio_es_limit: number };

type RiskBucketsPortfolioProps = {
  dataOverride?: PortfolioRB | null;
  loadingOverride?: boolean;
  errorOverride?: string | null;
  showControlsBar?: boolean;
};

export default function RiskBucketsPortfolio({ dataOverride, loadingOverride, errorOverride, showControlsBar = true }: RiskBucketsPortfolioProps = {}) {
  const { notify } = useNotifications();
  const { setControls } = useControls();
  const [refreshTick, setRefreshTick] = useState<number>(0);
  const rb = useCachedApi<PortfolioRB>(
    `risk_buckets_portfolio_v3_${refreshTick}`,
    "/risk-buckets/portfolio?advanced_v=3",
    60_000
  );
  const data = dataOverride ?? rb.data;
  const loading = loadingOverride ?? rb.loading;
  const error = errorOverride ?? rb.error;
  const ohlcv = useCachedApi<Ohlcv>("nifty_ohlcv_30", "/derivatives/nifty-ohlcv?limit=40", 60_000);
  const settingsCfg = useCachedApi<{ settings: SettingsConfig }>("risk_buckets_settings_config_portfolio", "/risk-buckets/settings/config", 60_000);
  const [breachLimitPct, setBreachLimitPct] = useState<number | null>(null);

  useEffect(() => {
    if (settingsCfg.data?.settings && breachLimitPct == null) {
      setBreachLimitPct(Number(settingsCfg.data.settings.portfolio_es_limit));
    }
  }, [settingsCfg.data, breachLimitPct]);

  useEffect(() => {
    if (!showControlsBar) {
      return () => {};
    }
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Breach Limit % (Total Loss)</span>
          <input
            className="control-input"
            type="number"
            value={breachLimitPct ?? ""}
            onChange={(e) => setBreachLimitPct(Number(e.target.value))}
          />
        </label>
        <button
          className="control-input"
          style={{ background: "linear-gradient(135deg, var(--gs-accent), var(--gs-accent2))", border: "none", color: "#fff" }}
          onClick={saveBreachLimit}
        >
          Apply
        </button>
      </div>
    );
    setControls({
      key: `risk-buckets-portfolio:${breachLimitPct ?? "na"}`,
      title: "Controls",
      summary: (
        <span>
          <span className="controls-summary-key">Breach</span> {breachLimitPct ?? "—"}%
        </span>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, breachLimitPct, showControlsBar]);

  const saveBreachLimit = async () => {
    if (!settingsCfg.data?.settings || breachLimitPct == null) return;
    try {
      await apiFetch("/risk-buckets/settings/config", {
        method: "POST",
        body: { settings: { ...settingsCfg.data.settings, portfolio_es_limit: Number(breachLimitPct) } }
      });
      notify({ type: "success", message: "Breach limit updated." });
      setRefreshTick((v) => v + 1);
    } catch (err) {
      notify({ type: "error", message: String(err) });
    }
  };

  const axisColor = "#b9c4d6";
  const advanced = data?.advanced_simulation || null;
  const overlayPlots = advanced?.overlay_plots || {};
  const rows = (advanced?.summary_rows || []) as Record<string, unknown>[];
  const engineSortKey = (raw: string): { ivOrder: number; base: string } => {
    const parts = String(raw || "").split("|");
    const base = String(parts[0] || raw || "");
    const iv = String(parts[1] || "").toLowerCase();
    const ivOrder = iv === "flat" ? 0 : (iv === "surface" ? 1 : 2);
    return { ivOrder, base };
  };
  const sortedRows = [...rows].sort((a, b) => {
    const ea = engineSortKey(String(a.engine || ""));
    const eb = engineSortKey(String(b.engine || ""));
    if (ea.ivOrder !== eb.ivOrder) return ea.ivOrder - eb.ivOrder;
    const baseCmp = ea.base.localeCompare(eb.base);
    if (baseCmp !== 0) return baseCmp;
    const ma = String(a.mode_variant || a.mode || "");
    const mb = String(b.mode_variant || b.mode || "");
    return ma.localeCompare(mb);
  });
  const pickPreferredModeKey = (modes: Record<string, ModePayload>, base: "repricing" | "greeks"): string => {
    const keys = Object.keys(modes || {});
    if (!keys.length) return "";
    if (base === "repricing") {
      if (keys.includes("repricing_bs")) return "repricing_bs";
      const repricingKeys = keys.filter((k) => k.startsWith("repricing"));
      return repricingKeys[0] || keys[0];
    }
    if (keys.includes("greeks")) return "greeks";
    return keys[0];
  };
  const repricingMedianByEngine = new Map(
    sortedRows
      .filter((r) => String(r.mode || "") === "repricing")
      .map((r) => [String(r.engine || ""), Number(r.median)] as [string, number])
      .filter(([, v]) => Number.isFinite(v))
  );
  const engines = (advanced?.engines || {}) as Record<string, EnginePayload>;
  const engineOptions = Object.keys(engines).sort((a, b) => {
    const ea = engineSortKey(a);
    const eb = engineSortKey(b);
    if (ea.ivOrder !== eb.ivOrder) return ea.ivOrder - eb.ivOrder;
    return ea.base.localeCompare(eb.base);
  });
  const [selectedEngine, setSelectedEngine] = useState<string>("");
  const effectiveEngine = engineOptions.includes(selectedEngine) ? selectedEngine : (engineOptions[0] || "");
  const modeOptions = Object.keys(engines[effectiveEngine]?.modes || {});
  const [selectedMode, setSelectedMode] = useState<string>("");
  const effectiveMode = modeOptions.includes(selectedMode) ? selectedMode : (modeOptions[0] || "");
  const [selectedPathEngine, setSelectedPathEngine] = useState<string>("");
  const [ddView, setDdView] = useState<"terminal" | "drawdown" | "both">("both");
  const effectivePathEngine = engineOptions.includes(selectedPathEngine) ? selectedPathEngine : (engineOptions[0] || "");
  const spotSample = (engines[effectivePathEngine]?.spot_paths_worst || engines[effectivePathEngine]?.spot_paths_sample || {}) as SpotPathsSample;
  const spotWorstAll = (engines[effectivePathEngine]?.spot_paths_worst_all || {}) as SpotPathsSample;
  const spotWorstDD = (engines[effectivePathEngine]?.spot_paths_worst_dd || {}) as SpotPathsSample;
  const pathDays = (spotSample.days || []) as number[];
  const pathSeries = (spotSample.paths || []) as number[][];
  const worstAllSeries = (spotWorstAll.paths || []) as number[][];
  const worstAllPnL = (spotWorstAll.terminal_pnl || []) as number[];
  const worstAllDD = (spotWorstAll.max_drawdown || []) as number[];
  const worstAllDDSeries = (spotWorstAll.drawdown_series || []) as number[][];
  const worstDDSeries = (spotWorstDD.drawdown_series || []) as number[][];
  const modePayload = ((engines[effectiveEngine]?.modes || {})[effectiveMode] || {}) as ModePayload;
  const selectedKpis = (modePayload.kpis || {}) as Record<string, unknown>;
  const selectedFan = (selectedKpis.fan || {}) as Record<string, number[]>;
  const termSample = (modePayload.terminal_pnl_sample || []) as number[];
  const sortedTerm = [...termSample].sort((a, b) => a - b);
  const quantiles = (selectedKpis.quantiles || {}) as Record<string, unknown>;
  const p1 = Number(quantiles.p1);
  const p5 = Number(quantiles.p5);
  const p1Lac = Number.isFinite(p1) ? toLac(p1) : (sortedTerm.length ? toLac(sortedTerm[Math.floor(0.01 * (sortedTerm.length - 1))]) : 0);
  const p5Lac = Number.isFinite(p5) ? toLac(p5) : (sortedTerm.length ? toLac(sortedTerm[Math.floor(0.05 * (sortedTerm.length - 1))]) : 0);
  const histVals = termSample.map((v) => toLac(v)).filter((v) => Number.isFinite(v));
  const bins = 28;
  const minX = histVals.length ? Math.min(...histVals) : 0;
  const maxX = histVals.length ? Math.max(...histVals) : 1;
  const span = Math.max(maxX - minX, 1e-9);
  const width = span / bins;
  const counts = new Array<number>(bins).fill(0);
  for (const v of histVals) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - minX) / width)));
    counts[idx] += 1;
  }
  const barX = counts.map((_, i) => minX + width * (i + 0.5));
  const barText = counts.map((c) => (c > 0 ? String(c) : ""));
  const allModes = Array.from(new Set(sortedRows.map((r) => String(r.mode || "")).filter(Boolean)));
  const effectiveAggMode = allModes.includes("repricing") ? "repricing" : (allModes[0] || "");
  const aggRows = sortedRows.filter((r) => String(r.mode || "") === effectiveAggMode);
  const var99Arr = aggRows.map((r) => Number(r.var99)).filter((v) => Number.isFinite(v));
  const es99Arr = aggRows.map((r) => Number(r.es99)).filter((v) => Number.isFinite(v));
  const p5Arr = aggRows.map((r) => Number(r.p5)).filter((v) => Number.isFinite(v));
  const maxddArr = aggRows.map((r) => Number(r.maxdd_p50)).filter((v) => Number.isFinite(v));
  const maxddP1Arr = aggRows.map((r) => Number(r.maxdd_p1)).filter((v) => Number.isFinite(v));
  const probLossArr = aggRows.map((r) => Number(r.prob_loss)).filter((v) => Number.isFinite(v));
  const probBreachArr = aggRows.map((r) => Number(r.prob_breach_total)).filter((v) => Number.isFinite(v));
  const meanArr = aggRows.map((r) => Number(r.mean)).filter((v) => Number.isFinite(v));
  const var99Median = pct(var99Arr, 50);
  const var99Min = var99Arr.length ? Math.min(...var99Arr) : NaN;
  const var99Max = var99Arr.length ? Math.max(...var99Arr) : NaN;
  const es99Median = pct(es99Arr, 50);
  const es99Min = es99Arr.length ? Math.min(...es99Arr) : NaN;
  const es99Max = es99Arr.length ? Math.max(...es99Arr) : NaN;
  const p5Median = pct(p5Arr, 50);
  const maxddMedian = pct(maxddArr, 50);
  const maxddP1 = pct(maxddP1Arr, 50);
  const probLossMin = probLossArr.length ? Math.min(...probLossArr) : NaN;
  const probLossMed = pct(probLossArr, 50);
  const probLossMax = probLossArr.length ? Math.max(...probLossArr) : NaN;
  const probBreachMin = probBreachArr.length ? Math.min(...probBreachArr) : NaN;
  const probBreachMed = pct(probBreachArr, 50);
  const probBreachMax = probBreachArr.length ? Math.max(...probBreachArr) : NaN;
  const medianArr = aggRows.map((r) => Number(r.median)).filter((v) => Number.isFinite(v));
  const medianMed = pct(medianArr, 50);
  const medianMin = medianArr.length ? Math.min(...medianArr) : NaN;
  const medianMax = medianArr.length ? Math.max(...medianArr) : NaN;
  const accountSize = Number(data?.account_size || 0);
  const marginUsed = Number(data?.margin_used || 0);
  const thetaNorm = Number(data?.zone?.theta_norm || 0);
  const gammaNorm = Number(data?.zone?.gamma_norm || 0);
  const vegaNorm = Number(data?.zone?.vega_norm || 0);
  const horizonDays = Number(advanced?.config?.horizon_days || 10);
  const windowsPerYear = horizonDays > 0 ? (252 / horizonDays) : 0;
  const expectedBreachesPerYear = probBreachMed * windowsPerYear;
  const expectedLossDaysPerYear = probLossMed * windowsPerYear;
  const annualMedianPnl = medianMed * windowsPerYear;
  const annualMedianPct = accountSize > 0 ? (annualMedianPnl / accountSize) * 100 : NaN;
  const annualTailLoss = es99Median * expectedBreachesPerYear;
  const annualTailLossPct = accountSize > 0 ? (annualTailLoss / accountSize) * 100 : NaN;
  const annualThetaPerLac = thetaNorm * 252;
  const annualThetaInr = (annualThetaPerLac / 100000) * (accountSize || 0);
  const annualThetaPct = accountSize > 0 ? (annualThetaInr / accountSize) * 100 : NaN;
  const marginStressProbs = Object.entries(engines).map(([engine, payload]) => {
    const modes = payload?.modes || {};
    const key = pickPreferredModeKey(modes, "repricing");
    const sample = ((modes[key] || {}).terminal_pnl_sample || []) as number[];
    if (!accountSize || !Number.isFinite(marginUsed) || !sample.length) return NaN;
    const stressed = sample.filter((pnl) => {
      const denom = accountSize + Number(pnl);
      if (!Number.isFinite(denom) || denom <= 0) return false;
      return (marginUsed / denom) > 0.8;
    });
    return sample.length ? (stressed.length / sample.length) : NaN;
  }).filter((v) => Number.isFinite(v));
  const probMarginStressMed = pct(marginStressProbs, 50);
  const p99UtilArr = var99Arr.map((v) => {
    const denom = accountSize + v;
    if (!Number.isFinite(denom) || denom <= 0) return NaN;
    return marginUsed / denom;
  }).filter((v) => Number.isFinite(v));
  const p99UtilMed = pct(p99UtilArr, 50);
  const p1Util = (() => {
    const denom = accountSize + var99Median;
    if (!Number.isFinite(denom) || denom <= 0) return NaN;
    return marginUsed / denom;
  })();
  const p5Util = (() => {
    const denom = accountSize + p5Median;
    if (!Number.isFinite(denom) || denom <= 0) return NaN;
    return marginUsed / denom;
  })();
  const p50Util = (() => {
    const denom = accountSize + medianMed;
    if (!Number.isFinite(denom) || denom <= 0) return NaN;
    return marginUsed / denom;
  })();
  const var99Iqr = pct(var99Arr, 75) - pct(var99Arr, 25);
  const var99Range = (var99Arr.length ? Math.max(...var99Arr) : NaN) - (var99Arr.length ? Math.min(...var99Arr) : NaN);
  const meanRange = (meanArr.length ? Math.max(...meanArr) : NaN) - (meanArr.length ? Math.min(...meanArr) : NaN);
  const reprRows = sortedRows.filter((r) => String(r.mode || "") === "repricing");
  const greeksByEngine = new Map(sortedRows.filter((r) => String(r.mode || "") === "greeks").map((r) => [String(r.engine || ""), r]));
  const paired = reprRows
    .map((r) => {
      const g = greeksByEngine.get(String(r.engine || ""));
      if (!g) return null;
      const dVar99 = Number(r.var99) - Number(g.var99);
      const dEs99 = Number(r.es99) - Number(g.es99);
      if (!Number.isFinite(dVar99) || !Number.isFinite(dEs99)) return null;
      return { dVar99, dEs99 };
    })
    .filter(Boolean) as { dVar99: number; dEs99: number }[];
  const modeGapVar99Med = pct(paired.map((p) => p.dVar99), 50);
  const modeGapEs99Med = pct(paired.map((p) => p.dEs99), 50);
  const tailBinCount = 24;
  const buildOverlayDist = (modeName: string, stackId: string) => {
    const samplesByEngine = Object.entries(engines).map(([engine, payload]) => {
      const modes = payload?.modes || {};
      const key =
        modeName === "repricing"
          ? pickPreferredModeKey(modes, "repricing")
          : pickPreferredModeKey(modes, "greeks");
      const sample = ((modes[key] || {}).terminal_pnl_sample || []) as number[];
      return { engine, sample: sample.map((v) => toLac(v)).filter((v) => Number.isFinite(v)) };
    });
    const allVals = samplesByEngine.flatMap((r) => r.sample);
    const minVRaw = allVals.length ? pct(allVals, 0.5) : -1;
    const maxVRaw = allVals.length ? pct(allVals, 99.5) : 1;
    const minV = Number.isFinite(minVRaw) ? minVRaw : -1;
    const maxV = Number.isFinite(maxVRaw) && maxVRaw > minV ? maxVRaw : (minV + 1);
    const spanV = Math.max(maxV - minV, 1e-9);
    const w = spanV / tailBinCount;
    const centers = new Array(tailBinCount).fill(0).map((_, i) => minV + w * (i + 0.5));
    const countsByEngine = samplesByEngine.map(({ sample }) => {
      const counts = new Array<number>(tailBinCount).fill(0);
      for (const v of sample) {
        const idx = Math.min(tailBinCount - 1, Math.max(0, Math.floor((v - minV) / w)));
        counts[idx] += 1;
      }
      return counts;
    });
    const totalCounts = new Array<number>(tailBinCount).fill(0);
    for (const counts of countsByEngine) {
      for (let i = 0; i < tailBinCount; i += 1) totalCounts[i] += counts[i];
    }
    const keepIdx = totalCounts.map((c, i) => ({ c, i })).filter((x) => x.c > 0).map((x) => x.i);
    const xFiltered = keepIdx.length ? keepIdx.map((i) => centers[i]) : centers;
    const traces = samplesByEngine.map(({ engine }, engineIdx) => {
      const full = countsByEngine[engineIdx];
      const yFiltered = keepIdx.length ? keepIdx.map((i) => full[i]) : full;
      return {
        type: "scatter",
        mode: "lines",
        x: xFiltered,
        y: yFiltered,
        name: `${engine.toUpperCase()}`,
        stackgroup: stackId,
        line: { width: 1.8 },
        hovertemplate: `${engine.toUpperCase()}<br>Bin center: ₹%{x:.2f}L<br>Count: %{y}<extra></extra>`,
      } as any;
    });
    return { traces, minV, maxV };
  };
  const repricingOverlay = buildOverlayDist("repricing", "repricing");
  const greeksOverlay = buildOverlayDist("greeks", "greeks");
  const sharedXRange = [-3, 3];
  const breachMed = probBreachMed;
  const baseRiskHigh = Number.isFinite(var99Median) && var99Median <= -300000;
  const baseRiskMed = Number.isFinite(var99Median) && var99Median > -300000 && var99Median <= -150000;
  const baseRiskLabel = baseRiskHigh ? "High Risk" : (baseRiskMed ? "Moderate Risk" : "Lower Risk");
  const stressHigh = Number.isFinite(var99Min) && var99Min <= -350000;
  const stressMed = Number.isFinite(var99Min) && var99Min > -350000 && var99Min <= -200000;
  const stressLabel = stressHigh ? "Severe Stress Loss" : (stressMed ? "Material Stress Loss" : "Contained Stress Loss");
  const breachHigh = Number.isFinite(breachMed) && breachMed >= 0.60;
  const breachMedFlg = Number.isFinite(breachMed) && breachMed >= 0.30 && breachMed < 0.60;
  const breachLabel = breachHigh ? "High Breach Odds" : (breachMedFlg ? "Watch Breach Odds" : "Contained Breach Odds");
  const modelConfidenceLow = (Number.isFinite(var99Range) && var99Range > 150000) || (Number.isFinite(modeGapVar99Med) && Math.abs(modeGapVar99Med) > 80000);
  const modelConfidenceMed = !modelConfidenceLow && ((Number.isFinite(var99Range) && var99Range > 80000) || (Number.isFinite(modeGapVar99Med) && Math.abs(modeGapVar99Med) > 40000));
  const confidenceLabel = modelConfidenceLow ? "Low Confidence" : (modelConfidenceMed ? "Medium Confidence" : "High Confidence");

  return (
    <>
      {error ? <ErrorState message={error} /> : null}

      <SectionCard title="Seller Health">
        {loading || !data ? <LoadingState /> : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12 }}>
            <div className="metric-card">
              <div className="seller-health-label">Expectation</div>
              <div style={{ display: "grid", gap: 6 }}>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Mean of Med | Med of Med
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Mean of per-engine median P&amp;L | Median of per-engine median P&amp;L.</span>
                    </span>
                  </span>
                  <span className={`seller-metric-value${avg(medianArr) < 0 ? " negative" : " positive"}`}>{`${formatInrLac(avg(medianArr))} | ${formatInrLac(medianMed)}`}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Median P&L Range
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Min to max of per-engine median P&amp;L values (range across engines).</span>
                    </span>
                  </span>
                  <span className={`seller-metric-value${medianMed < 0 ? " negative" : " positive"}`}>{`${formatInrLac(medianMin)} - ${formatInrLac(medianMax)}`}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Probability of Profit
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Probability of profit = 1 - prob_loss; median of per-engine chance that terminal P&amp;L is negative.</span>
                    </span>
                  </span>
                  <span className={`seller-metric-value${(1 - probLossMed) * 100 < 50 ? " negative" : ""}`}>{formatPct((1 - probLossMed) * 100)}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    DD P50
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Median of per-engine max drawdown P50 (typical worst dip per path over the horizon).</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{formatInrLac(maxddMedian)}</span>
                </div>
              </div>
            </div>
            <div className="metric-card">
              <div className="seller-health-label">Risk</div>
              <div style={{ display: "grid", gap: 6 }}>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    VaR (99%)
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Median of per-engine 1st percentile terminal P&amp;L (VaR99 threshold).</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{formatInrLac(var99Median)}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Expected Shortfall
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Median of per-engine ES99: average loss of the worst 1% terminal P&amp;L paths.</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{formatInrLac(es99Median)}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    P5 Outcome
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Median of per-engine 5th percentile terminal P&amp;L (P5 outcome).</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{formatInrLac(p5Median)}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Breach Odds
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Median probability that total loss limit is breached at any point in the horizon.</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{formatPct(probBreachMed * 100)}</span>
                </div>
              </div>
            </div>
            <div className="metric-card">
              <div className="seller-health-label">Exposure</div>
              <div style={{ display: "grid", gap: 6 }}>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    DD P1
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Median of per-engine max drawdown P1 (worst 1% drawdown).</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{formatInrLac(maxddP1)}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Margin Stress (&gt;80%)
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Median probability that margin_used / (account_size + terminal P&amp;L) exceeds 80%, plus current utilization.</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">
                    {formatPct(probMarginStressMed * 100)}
                    {" | "}
                    <span className={(marginUsed / (accountSize || 1)) < 0.8 ? "positive" : "negative"}>
                      {formatPct((marginUsed / (accountSize || 1)) * 100)}
                    </span>
                  </span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Margin (P1 | P5 | P50)
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Margin utilization computed at P1 (VaR99), P5, and P50 terminal P&amp;L levels.</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{`${formatPct(p1Util * 100)} | ${formatPct(p5Util * 100)} | ${formatPct(p50Util * 100)}`}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Convexity | Vega
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Portfolio gamma and vega normalized per ₹1L capital (convexity and volatility exposure).</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{`${formatNumber(gammaNorm, 4)} | ${formatNumber(vegaNorm, 0)}`}</span>
                </div>
              </div>
            </div>
            <div className="metric-card">
              <div className="seller-health-label">Annualized Policy</div>
              <div style={{ display: "grid", gap: 6 }}>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Breaches / Year
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Expected breaches per year assuming independent rolling windows: prob_breach × (252 / horizon_days).</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{formatNumber(expectedBreachesPerYear, 2)}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Loss Windows / Year
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Expected loss windows per year: prob_loss × (252 / horizon_days).</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{formatNumber(expectedLossDaysPerYear, 2)}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Annual Tail Loss Budget
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Heuristic: ES99 × expected breaches/year. Assumes independent rolling windows.</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{`${formatInrLac(annualTailLoss)} | ${formatPct(annualTailLossPct)}`}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Annual Median Return
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Annualized median P&amp;L using rolling windows: median P&amp;L × (252 / horizon_days).</span>
                    </span>
                  </span>
                  <span className={`seller-metric-value${annualMedianPnl < 0 ? " negative" : " positive"}`}>{`${formatInrLac(annualMedianPnl)} | ${formatPct(annualMedianPct)}`}</span>
                </div>
                <div className="seller-metric-row" style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                  <span className="seller-metric-label">
                    Annual Theta
                    <span className="metric-help seller-metric-help">
                      ?
                      <span className="metric-popover">Annualized theta based on ₹1L per day × 252 days, scaled to account size.</span>
                    </span>
                  </span>
                  <span className="seller-metric-value">{`${formatInrLac(annualThetaInr)} | ${formatPct(annualThetaPct)}`}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </SectionCard>

      <SectionCard title="Cross-Engine Risk Consensus">
        {loading || !data ? <LoadingState /> : (
          <>
            <div className="metric-grid-row">
              <MetricGrid>
                <MetricCard
                  label={`VaR99 (${baseRiskLabel})`}
                  value={`${formatInrLac(var99Median)} | ${formatInrLac(avg(var99Arr))} | ${formatInrLac(var99Min)} - ${formatInrLac(var99Max)}`}
                />
                <MetricCard
                  label={`ES99 (${stressLabel})`}
                  value={`${formatInrLac(es99Median)} | ${formatInrLac(avg(es99Arr))} | ${formatInrLac(es99Min)} - ${formatInrLac(es99Max)}`}
                />
                <MetricCard
                  label="P&L (Engine Aggregate)"
                  value={`${formatInrLac(medianMed)} | ${formatInrLac(avg(meanArr))} | ${formatInrLac(medianMin)} - ${formatInrLac(medianMax)}`}
                />
              </MetricGrid>
            </div>
            <div className="metric-grid-row">
              <MetricGrid>
                <MetricCard
                  label={`Prob Loss (Median)`}
                  value={`${formatPct(probLossMed * 100)} | Range ${formatPct(probLossMin * 100)}-${formatPct(probLossMax * 100)}`}
                />
                <MetricCard
                  label={`Breach Odds (Median)`}
                  value={`${formatPct(probBreachMed * 100)} | Range ${formatPct(probBreachMin * 100)}-${formatPct(probBreachMax * 100)}`}
                />
                <MetricCard
                  label={`Model Confidence (${confidenceLabel})`}
                  value={`Dispersion ${formatInrLac(var99Range)} | ModeGap ${formatInrLac(modeGapVar99Med)}`}
                />
              </MetricGrid>
            </div>
          </>
        )}
      </SectionCard>

      <SectionCard title="Engine Overlay Empirical CDF">
        {loading || !data ? <LoadingState /> : (
          <Plot
            data={[
              ...((overlayPlots.cdf || []).flatMap((row) => {
                const e = String(row.engine || "").toUpperCase();
                const ek = String(row.engine || "");
                const x = (row.x || []).map((v) => toLac(v || 0));
                const y = (row.cdf || []).map((v) => Number(v || 0));
                const var99 = Number(row.var99 || 0);
                const med = repricingMedianByEngine.get(ek);
                return [
                  { type: "scatter", mode: "lines", x, y, name: `${e} CDF`, line: { width: 2 } },
                  {
                    type: "scatter",
                    mode: "lines",
                    x: [toLac(var99), toLac(var99)],
                    y: [0, 1],
                    name: `${e} VaR99`,
                    line: { width: 1, dash: "dash" },
                    opacity: 0.6
                  },
                  ...(Number.isFinite(med)
                    ? [{
                        type: "scatter",
                        mode: "lines",
                        x: [toLac(med), toLac(med)],
                        y: [0, 1],
                        name: `${e} Median`,
                        line: { width: 1, dash: "dot" },
                        opacity: 0.55
                      }]
                    : [])
                ];
              }) as any[])
            ]}
            layout={{
              height: 360,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 56, r: 20, t: 20, b: 48 },
              xaxis: {
                title: "Terminal P&L (₹ Lacs)",
                automargin: true,
                tickfont: { color: axisColor, size: 10 },
                titlefont: { color: axisColor, size: 11 },
                range: sharedXRange
              },
              yaxis: { title: "CDF", automargin: true, range: [0, 1], tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
              showlegend: false
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
          />
        )}
      </SectionCard>

      <SectionCard title="Engine Overlay Loss Distribution">
        {loading || !data ? <LoadingState /> : (
          <Plot
            data={repricingOverlay.traces}
            layout={{
              height: 360,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 56, r: 20, t: 20, b: 48 },
              xaxis: {
                title: "Terminal P&L Bin Center (₹ Lacs)",
                automargin: true,
                tickfont: { color: axisColor, size: 10 },
                titlefont: { color: axisColor, size: 11 },
                range: sharedXRange || [repricingOverlay.minV, repricingOverlay.maxV]
              },
              yaxis: { title: "Count", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
              showlegend: false
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
          />
        )}
      </SectionCard>

      <SectionCard title="Engine Overlay Loss Distribution (Greeks Mode)">
        {loading || !data ? <LoadingState /> : (
          <Plot
            data={greeksOverlay.traces}
            layout={{
              height: 360,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 56, r: 20, t: 20, b: 48 },
              xaxis: {
                title: "Terminal P&L Bin Center (₹ Lacs)",
                automargin: true,
                tickfont: { color: axisColor, size: 10 },
                titlefont: { color: axisColor, size: 11 },
                range: sharedXRange || [greeksOverlay.minV, greeksOverlay.maxV]
              },
              yaxis: { title: "Count", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
              showlegend: false
            }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
          />
        )}
      </SectionCard>

      <SectionCard title="Advanced Engine Summary">
        {loading || !data ? <LoadingState /> : (
          <>
            <div className="risk-alert" style={{ marginBottom: 8 }}>
              Advanced model version: <strong>{advanced?.model_version || "unknown"}</strong>
            </div>
            <DataTable
              rows={sortedRows}
              maxHeight={380}
              columns={[
                { key: "engine", label: "Engine" },
                { key: "mode", label: "P&L Mode" },
                { key: "mode_variant", label: "Mode Variant" },
                { key: "iv_rule", label: "IV Rule" },
                {
                  key: "pricing_model",
                  label: "Pricing",
                  render: (row) => {
                    const mode = String(row.mode || "");
                    if (mode === "greeks") return "-";
                    const raw = row.pricing_model;
                    return String(raw ?? "—");
                  }
                },
                { key: "mean", label: "Mean (₹ Lacs)", format: formatInrLac },
                { key: "median", label: "Median (₹ Lacs)", format: formatInrLac },
                { key: "var95", label: "VaR95 (P5, ₹ Lacs)", format: formatInrLac },
                { key: "var99", label: "VaR99 (P1, ₹ Lacs)", format: formatInrLac },
                { key: "es95", label: "ES95 (₹ Lacs)", format: formatInrLac },
                { key: "es99", label: "ES99 (₹ Lacs)", format: formatInrLac },
                { key: "prob_loss", label: "Prob Loss", format: (v) => formatPct((Number(v) || 0) * 100) },
                { key: "prob_breach_total", label: "Prob Breach Total", format: (v) => formatPct((Number(v) || 0) * 100) },
                { key: "maxdd_p1", label: "MaxDD P1 (₹ Lacs)", format: formatInrLac },
                { key: "vol_rmse", label: "Vol RMSE", format: (v) => formatNumber(v, 4) },
                { key: "surface_iv_rmse", label: "IV Fit RMSE", format: (v) => formatNumber(v, 4) }
              ]}
            />
          </>
        )}
      </SectionCard>

      <SectionCard title="Engine + Mode View">
        {loading || !data ? <LoadingState /> : (
          <>
            <div style={{ display: "flex", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
              <label className="control-field" style={{ minWidth: 180 }}>
                <span className="control-label">Engine</span>
                <select className="control-input" value={effectiveEngine} onChange={(e) => setSelectedEngine(e.target.value)}>
                  {engineOptions.map((e) => (
                    <option key={e} value={e}>{e.toUpperCase()}</option>
                  ))}
                </select>
              </label>
              <label className="control-field" style={{ minWidth: 180 }}>
                <span className="control-label">P&L Mode</span>
                <select className="control-input" value={effectiveMode} onChange={(e) => setSelectedMode(e.target.value)}>
                  {modeOptions.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </label>
            </div>

            <MetricGrid>
              <MetricCard label="Mean" value={formatInrLac(selectedKpis.mean)} />
              <MetricCard label="Median" value={formatInrLac(selectedKpis.median)} />
              <MetricCard label="VaR99 (P1)" value={formatInrLac(selectedKpis.var99)} />
              <MetricCard label="ES99" value={formatInrLac(selectedKpis.es99)} />
              <MetricCard label="Prob Loss" value={formatPct((Number(selectedKpis.prob_loss) || 0) * 100)} />
            </MetricGrid>

            <div className="chart-panel" style={{ marginTop: 12 }}>
              <h4>Fan Chart ({effectiveEngine.toUpperCase()} · {effectiveMode})</h4>
              <Plot
                data={[
                  { type: "scatter", mode: "lines", x: (selectedFan.p50 || []).map((_, i) => i + 1), y: (selectedFan.p50 || []).map((v) => toLac(v || 0)), name: "P50", line: { color: "#2D7DFF", width: 2 } },
                  { type: "scatter", mode: "lines", x: (selectedFan.p10 || []).map((_, i) => i + 1), y: (selectedFan.p10 || []).map((v) => toLac(v || 0)), name: "P10", line: { color: "#FFB020", dash: "dot" } },
                  { type: "scatter", mode: "lines", x: (selectedFan.p5 || []).map((_, i) => i + 1), y: (selectedFan.p5 || []).map((v) => toLac(v || 0)), name: "P5", line: { color: "#FF7A00", dash: "dot" } },
                  { type: "scatter", mode: "lines", x: (selectedFan.p1 || []).map((_, i) => i + 1), y: (selectedFan.p1 || []).map((v) => toLac(v || 0)), name: "P1", line: { color: "#FF4D4D", dash: "dot" } }
                ]}
                layout={{
                  height: 360,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 56, r: 20, t: 20, b: 48 },
                  xaxis: { title: "Day", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
                  yaxis: { title: "P&L (₹ Lacs)", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
                  legend: { font: { color: axisColor, size: 10 } }
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
              />
            </div>

            <div className="chart-panel" style={{ marginTop: 12 }}>
              <h4>Loss Distribution ({effectiveEngine.toUpperCase()} · {effectiveMode})</h4>
              <Plot
                data={[
                  {
                    type: "bar",
                    x: barX,
                    y: counts,
                    name: "Count",
                    marker: { color: "rgba(45,125,255,0.55)", line: { width: 1, color: "rgba(45,125,255,0.95)" } },
                    text: barText,
                    textposition: "outside",
                    cliponaxis: false,
                    hovertemplate: "P&L bin center: ₹%{x:.2f}L<br>Count: %{y}<extra></extra>"
                  },
                  {
                    type: "scatter",
                    mode: "lines",
                    x: [p5Lac, p5Lac],
                    y: [0, 1],
                    name: "P5",
                    line: { color: "#FF7A00", width: 1.5, dash: "dot" }
                  },
                  {
                    type: "scatter",
                    mode: "lines",
                    x: [p1Lac, p1Lac],
                    y: [0, 1],
                    name: "P1",
                    line: { color: "#FF4D4D", width: 1.5, dash: "dash" }
                  }
                ]}
                layout={{
                  height: 360,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 56, r: 20, t: 20, b: 48 },
                  xaxis: { title: "Terminal P&L (₹ Lacs)", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
                  yaxis: { title: "Count", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
                  legend: { font: { color: axisColor, size: 10 } }
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
              />
            </div>
          </>
        )}
      </SectionCard>

      <SectionCard title="NIFTY Last 30 Days + Simulated Paths">
        {loading || !data || ohlcv.loading || !ohlcv.data ? <LoadingState /> : (
          (() => {
            const last10 = (ohlcv.data.rows || [])
              .filter((r) => Boolean(String(r.date || "")))
              .sort((a, b) => String(a.date || "").localeCompare(String(b.date || "")))
              .slice(-10);
            const histDates = last10.map((r) => String(r.date || ""));
            const histClose = last10.map((r) => Number(r.close || 0));
            const histOpen = last10.map((r) => Number(r.open || 0));
            const histHigh = last10.map((r) => Number(r.high || 0));
            const histLow = last10.map((r) => Number(r.low || 0));
            const lastDateStr = histDates[histDates.length - 1] || "";
            const lastClose = histClose[histClose.length - 1] || 0;
            const baseDate = lastDateStr ? new Date(lastDateStr) : null;
            const simDays = Math.min(pathDays.length, 10);
            const simDates = baseDate && simDays
              ? [baseDate.toISOString().slice(0, 10), ...pathDays.slice(0, simDays).map((_, i) => {
                  const d = new Date(baseDate.getTime());
                  d.setDate(d.getDate() + i + 1);
                  return d.toISOString().slice(0, 10);
                })]
              : [];
            const simSeries = pathSeries.map((series) => [lastClose, ...series.slice(0, simDays)]);
            const worstAllSeriesAdj = worstAllSeries.map((series) => [lastClose, ...series.slice(0, simDays)]);
            const worstAllDDAdj = worstAllDDSeries.map((series) => [0, ...series.slice(0, simDays)]);
            const worstDDAdj = worstDDSeries.map((series) => [0, ...series.slice(0, simDays)]);
            const ddBars = worstAllDD.map((v, idx) => ({ id: idx + 1, dd: v }));
            return (
              <>
                <div style={{ display: "flex", gap: 12, marginBottom: 12, flexWrap: "wrap" }}>
                  <label className="control-field" style={{ minWidth: 200 }}>
                    <span className="control-label">Engine</span>
                    <select className="control-input" value={effectivePathEngine} onChange={(e) => setSelectedPathEngine(e.target.value)}>
                      {engineOptions.map((e) => (
                        <option key={e} value={e}>{e.toUpperCase()}</option>
                      ))}
                    </select>
                  </label>
                </div>
                <Plot
                  data={[
                    {
                      type: "candlestick",
                      x: histDates,
                      open: histOpen,
                      high: histHigh,
                      low: histLow,
                      close: histClose,
                      name: "NIFTY OHLC",
                      increasing: { line: { color: "#2ecc71" } },
                      decreasing: { line: { color: "#ff4d4d" } }
                    },
                    ...simSeries.map((series) => ({
                      type: "scatter",
                      mode: "lines",
                      x: simDates,
                      y: series,
                      line: { width: 1, color: "rgba(108,192,255,0.35)" },
                      hoverinfo: "skip",
                      showlegend: false
                    })),
                    ...worstAllSeriesAdj.map((series, idx) => ({
                      type: "scatter",
                      mode: "lines",
                      x: simDates,
                      y: series,
                      line: { width: 2, color: "rgba(255,77,77,0.75)" },
                      hovertemplate: `Terminal P&L: ₹${formatNumber(worstAllPnL[idx], 0)}<br>Max DD: ₹${formatNumber(worstAllDD[idx], 0)}<extra></extra>`,
                      showlegend: false
                    }))
                  ]}
                  layout={{
                    height: 380,
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    margin: { l: 56, r: 20, t: 20, b: 48 },
                    xaxis: { title: "Date", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 }, rangeslider: { visible: false } },
                    yaxis: { title: "NIFTY Spot", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } }
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                />
                <div style={{ marginTop: 14 }}>
                  <Plot
                    data={[
                      {
                        type: "bar",
                        x: ddBars.map((d) => `Path ${d.id}`),
                        y: ddBars.map((d) => d.dd),
                        marker: { color: "#ff4d4d" },
                        name: "Max Drawdown"
                      }
                    ]}
                    layout={{
                      height: 220,
                      paper_bgcolor: "rgba(0,0,0,0)",
                      plot_bgcolor: "rgba(0,0,0,0)",
                      margin: { l: 56, r: 20, t: 12, b: 48 },
                      xaxis: { title: "Worst 20 Paths", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
                      yaxis: { title: "Max Drawdown (₹)", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
                      showlegend: false
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: "100%", height: "100%" }}
                    useResizeHandler
                  />
                </div>
                <div style={{ display: "flex", gap: 12, marginBottom: 8, flexWrap: "wrap" }}>
                  <label className="control-field" style={{ minWidth: 220 }}>
                    <span className="control-label">Drawdown Paths</span>
                    <select className="control-input" value={ddView} onChange={(e) => setDdView(e.target.value as "terminal" | "drawdown" | "both")}>
                      <option value="both">Worst Terminal + Worst DD</option>
                      <option value="terminal">Worst Terminal Only</option>
                      <option value="drawdown">Worst DD Only</option>
                    </select>
                  </label>
                </div>
                <div style={{ marginTop: 4 }}>
                  <Plot
                    data={[
                      ...(ddView !== "drawdown"
                        ? worstAllDDAdj.map((series) => ({
                            type: "scatter",
                            mode: "lines",
                            x: simDates,
                            y: series,
                            line: { width: 1.4, color: "rgba(255,77,77,0.6)" },
                            hoverinfo: "skip",
                            showlegend: false
                          }))
                        : []),
                      ...(ddView !== "terminal"
                        ? worstDDAdj.map((series) => ({
                            type: "scatter",
                            mode: "lines",
                            x: simDates,
                            y: series,
                            line: { width: 1.4, color: "rgba(46,204,113,0.6)" },
                            hoverinfo: "skip",
                            showlegend: false
                          }))
                        : []),
                    ]}
                    layout={{
                      height: 240,
                      paper_bgcolor: "rgba(0,0,0,0)",
                      plot_bgcolor: "rgba(0,0,0,0)",
                      margin: { l: 56, r: 20, t: 12, b: 48 },
                      xaxis: { title: "Date", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
                      yaxis: { title: "Drawdown (₹)", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
                      showlegend: false
                    }}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: "100%", height: "100%" }}
                    useResizeHandler
                  />
                </div>
              </>
            );
          })()
        )}
      </SectionCard>
    </>
  );
}
