import { useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import DataTable from "../components/DataTable";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { formatPct, formatNumber } from "../components/format";

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

type OverlayCdfRow = { engine?: string; x?: number[]; cdf?: number[]; var99?: number };
type OverlayTailRow = { engine?: string; x?: number[]; cdf?: number[]; ccdf?: number[]; es99?: number };
type ModePayload = { kpis?: Record<string, unknown>; terminal_pnl_sample?: number[] };
type EnginePayload = { fit_kpis?: Record<string, unknown>; modes?: Record<string, ModePayload> };

type PortfolioRB = {
  advanced_simulation?: {
    model_version?: string;
    summary_rows?: Record<string, unknown>[];
    engines?: Record<string, EnginePayload>;
    overlay_plots?: {
      cdf?: OverlayCdfRow[];
      tail?: OverlayTailRow[];
      tail_xlim?: number[] | null;
    };
  } | null;
};

export default function RiskBucketsPortfolio() {
  const { data, error, loading } = useCachedApi<PortfolioRB>(
    "risk_buckets_portfolio_v3",
    "/risk-buckets/portfolio?advanced_v=3",
    5_000
  );

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
  const probLossArr = aggRows.map((r) => Number(r.prob_loss)).filter((v) => Number.isFinite(v));
  const probBreachArr = aggRows.map((r) => Number(r.prob_breach_total)).filter((v) => Number.isFinite(v));
  const meanArr = aggRows.map((r) => Number(r.mean)).filter((v) => Number.isFinite(v));
  const var99Median = pct(var99Arr, 50);
  const var99Min = var99Arr.length ? Math.min(...var99Arr) : NaN;
  const var99Max = var99Arr.length ? Math.max(...var99Arr) : NaN;
  const es99Median = pct(es99Arr, 50);
  const es99Min = es99Arr.length ? Math.min(...es99Arr) : NaN;
  const es99Max = es99Arr.length ? Math.max(...es99Arr) : NaN;
  const probLossMin = probLossArr.length ? Math.min(...probLossArr) : NaN;
  const probLossMed = pct(probLossArr, 50);
  const probLossMax = probLossArr.length ? Math.max(...probLossArr) : NaN;
  const probBreachMin = probBreachArr.length ? Math.min(...probBreachArr) : NaN;
  const probBreachMed = pct(probBreachArr, 50);
  const probBreachMax = probBreachArr.length ? Math.max(...probBreachArr) : NaN;
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

      <SectionCard title="Cross-Engine Risk Consensus">
        {loading || !data ? <LoadingState /> : (
          <>
            <MetricGrid>
              <MetricCard
                label={`Base Risk (${baseRiskLabel})`}
                value={`${formatInrLac(var99Median)} | P(loss) ${formatPct(probLossMed * 100)}`}
              />
              <MetricCard
                label={`Stress Floor (${stressLabel})`}
                value={`${formatInrLac(var99Min)} | ES99 ${formatInrLac(es99Median)}`}
              />
              <MetricCard
                label={`Breach Odds (${breachLabel})`}
                value={`${formatPct(probBreachMed * 100)} | Range ${formatPct(probBreachMin * 100)}-${formatPct(probBreachMax * 100)}`}
              />
              <MetricCard
                label={`Model Confidence (${confidenceLabel})`}
                value={`Dispersion ${formatInrLac(var99Range)} | ModeGap ${formatInrLac(modeGapVar99Med)}`}
              />
              <MetricCard
                label="Consensus Mean P&L"
                value={`${formatInrLac(meanArr.length ? (meanArr.reduce((a, b) => a + b, 0) / meanArr.length) : NaN)} | Band ${formatInrLac(meanArr.length ? Math.min(...meanArr) : NaN)}-${formatInrLac(meanArr.length ? Math.max(...meanArr) : NaN)}`}
              />
              <MetricCard
                label="Consensus Median P&L"
                value={`${formatInrLac(pct(meanArr, 50))} | Band ${formatInrLac(meanArr.length ? Math.min(...meanArr) : NaN)}-${formatInrLac(meanArr.length ? Math.max(...meanArr) : NaN)}`}
              />
            </MetricGrid>
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
              xaxis: { title: "Terminal P&L (₹ Lacs)", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
              yaxis: { title: "CDF", automargin: true, range: [0, 1], tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
              legend: { font: { color: axisColor, size: 10 } }
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
                range: [repricingOverlay.minV, repricingOverlay.maxV]
              },
              yaxis: { title: "Count", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
              legend: { font: { color: axisColor, size: 10 } }
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
                range: [greeksOverlay.minV, greeksOverlay.maxV]
              },
              yaxis: { title: "Count", automargin: true, tickfont: { color: axisColor, size: 10 }, titlefont: { color: axisColor, size: 11 } },
              legend: { font: { color: axisColor, size: 10 } }
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
                { key: "pricing_model", label: "Pricing" },
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
    </>
  );
}
