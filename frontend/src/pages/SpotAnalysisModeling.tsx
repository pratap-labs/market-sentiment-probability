import { useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { API_BASE_URL } from "../api/client";

type ModelHorizonMetric = {
  horizon: number;
  train_rows: number;
  test_rows: number;
  auc: number | null;
  brier: number | null;
  pinball_q10: number;
  pinball_q50: number;
  pinball_q90: number;
  wf_folds?: number;
  wf_auc?: number | null;
  wf_brier?: number | null;
  wf_pinball_q10?: number | null;
  wf_pinball_q50?: number | null;
  wf_pinball_q90?: number | null;
  selected_feature_count?: number;
  selected_features?: string[];
};

type FlowModelSummaryResponse = {
  metrics: {
    test_days: number;
    target_mode?: string;
    rows_after_feature_filter: number;
    features_count: number;
    mode_comparison?: Array<{
      target_mode: string;
      horizons: Array<{
        horizon: number;
        auc: number | null;
        brier: number | null;
        pinball_q10: number;
        pinball_q50: number;
        pinball_q90: number;
        wf_folds?: number;
        wf_auc?: number | null;
        wf_brier?: number | null;
      }>;
    }>;
    horizons: ModelHorizonMetric[];
  };
  diagnostics?: {
    corr_matrix?: { features: string[]; values: number[][] };
    pca?: {
      components: string[];
      explained_variance_ratio: number[];
      cumulative_explained_variance: number[];
    };
    spot_variance_explained?: {
      target: string;
      method: string;
      by_horizon: Record<string, Array<{
        rank: number;
        feature: string;
        individual_r2: number;
        cumulative_r2: number;
        marginal_r2: number;
      }>>;
    };
    top_feature_correlations?: Record<string, {
      target_ret: Array<{ feature: string; corr: number }>;
      target_up: Array<{ feature: string; corr: number }>;
    }>;
  };
  predictions: Array<{
    date: string;
    horizon: number;
    close: number;
    actual_ret: number;
    p_up: number;
    pred_close_q10: number;
    pred_close_q50: number;
    pred_close_q90: number;
  }>;
  live_signal?: {
    rows: Array<{
      horizon: number;
      as_of_date?: string | null;
      close?: number;
      p_up?: number;
      pred_close_q10?: number;
      pred_close_q50?: number;
      pred_close_q90?: number;
      signal?: string;
    }>;
  };
};

type FeatureDiagnosticsResponse = {
  target_mode: string;
  test_days: number;
  rolling_window: number;
  top_n: number;
  corr_threshold: number;
  by_horizon: Record<string, Array<{
    feature: string;
    full_corr: number;
    median_ic: number;
    mean_ic: number;
    ic_std: number;
    ic_ir: number;
    sign_consistency: number;
    coverage: number;
  }>>;
  collinearity_pairs: Array<{
    feature_a: string;
    feature_b: string;
    abs_corr: number;
  }>;
};

function toShortDate(iso: string): string {
  const dt = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(dt.getTime())) return iso;
  return dt.toLocaleDateString("en-GB", { day: "2-digit", month: "short" });
}

function formatNum(value: number | null | undefined, d = 3): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return Number(value).toFixed(d);
}

export default function SpotAnalysisModeling() {
  const [selectedModelHorizon, setSelectedModelHorizon] = useState<number>(1);
  const [refreshToken, setRefreshToken] = useState<number>(Date.now());
  const [retraining, setRetraining] = useState(false);
  const [retrainStatus, setRetrainStatus] = useState<string>("");
  const model = useCachedApi<FlowModelSummaryResponse>(
    `flow_model_summary_${refreshToken}`,
    `/spot-analysis/flow-model-summary?pred_limit=20&t=${refreshToken}`,
    60_000
  );
  const featureDiag = useCachedApi<FeatureDiagnosticsResponse>(
    `flow_feature_diagnostics_${refreshToken}`,
    `/spot-analysis/feature-diagnostics?rolling_window=90&top_n=25&corr_threshold=0.9&t=${refreshToken}`,
    60_000
  );

  const modelHorizons = useMemo(
    () => (model.data?.metrics?.horizons || []).map((h) => h.horizon).sort((a, b) => a - b),
    [model.data]
  );

  const modelPlot = useMemo(() => {
    if (!model.data) return { x: [], q10: [], q50: [], q90: [], actual: [], pUp: [] };
    const rows = (model.data.predictions || [])
      .filter((r) => Number(r.horizon) === Number(selectedModelHorizon))
      .sort((a, b) => String(a.date).localeCompare(String(b.date)));
    const x = rows.map((r) => toShortDate(String(r.date)));
    const q10 = rows.map((r) => Number(r.pred_close_q10 || NaN));
    const q50 = rows.map((r) => Number(r.pred_close_q50 || NaN));
    const q90 = rows.map((r) => Number(r.pred_close_q90 || NaN));
    const actual = rows.map((r) => Number(r.close || 0) * (1 + Number(r.actual_ret || 0)));
    const pUp = rows.map((r) => Number(r.p_up || 0) * 100);
    return { x, q10, q50, q90, actual, pUp };
  }, [model.data, selectedModelHorizon]);

  const topFeatureBars = useMemo(() => {
    const bucket = model.data?.diagnostics?.top_feature_correlations?.[String(selectedModelHorizon)];
    const ret = bucket?.target_ret || [];
    const up = bucket?.target_up || [];
    return { ret, up };
  }, [model.data, selectedModelHorizon]);

  const varianceExplainedRows = useMemo(
    () => model.data?.diagnostics?.spot_variance_explained?.by_horizon?.[String(selectedModelHorizon)] || [],
    [model.data, selectedModelHorizon]
  );

  const liveRows = useMemo(
    () => (model.data?.live_signal?.rows || []).slice().sort((a, b) => Number(a.horizon) - Number(b.horizon)),
    [model.data]
  );
  const stabilityRows = useMemo(
    () => featureDiag.data?.by_horizon?.[String(selectedModelHorizon)] || [],
    [featureDiag.data, selectedModelHorizon]
  );

  return (
    <>
      <SectionCard title="Flow Model Snapshot">
        {model.loading ? (
          <LoadingState />
        ) : model.error || !model.data ? (
          <ErrorState message={String(model.error || "Model summary unavailable")} />
        ) : (
          <>
            <div className="controls-row" style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
              <button
                className="control-input"
                disabled={retraining}
                onClick={async () => {
                  setRetraining(true);
                  setRetrainStatus("");
                  try {
                    const res = await fetch(
                      `${API_BASE_URL}/spot-analysis/flow-model/retrain-latest?timeout_seconds=1200`,
                      { method: "POST" }
                    );
                    if (!res.ok) {
                      const txt = await res.text();
                      throw new Error(txt || "Retrain failed");
                    }
                    const payload = await res.json();
                    const reason = String(payload?.reason || "");
                    setRetrainStatus(
                      reason === "cache_only_retrain"
                        ? "Retrained using local cache only (no fetch)."
                        : "Retrained."
                    );
                    setRefreshToken(Date.now());
                  } catch (err) {
                    setRetrainStatus(`Retrain failed: ${String(err)}`);
                  } finally {
                    setRetraining(false);
                  }
                }}
              >
                {retraining ? "Retraining..." : "Retrain Model Till Latest Data"}
              </button>
              {retrainStatus ? <span style={{ color: "#b9c4d6", fontSize: 12 }}>{retrainStatus}</span> : null}
            </div>

            <div className="chart-panel" style={{ marginBottom: 12 }}>
              <h4>Today Signal (Latest Available)</h4>
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Horizon</th>
                      <th>As Of</th>
                      <th>Close</th>
                      <th>P(up) %</th>
                      <th>Signal</th>
                      <th>Pred q10</th>
                      <th>Pred q50</th>
                      <th>Pred q90</th>
                    </tr>
                  </thead>
                  <tbody>
                    {liveRows.map((r) => (
                      <tr key={`live-${r.horizon}`}>
                        <td>{r.horizon}d</td>
                        <td>{r.as_of_date || "—"}</td>
                        <td>{formatNum(r.close, 2)}</td>
                        <td>{formatNum((Number(r.p_up || 0) * 100), 1)}</td>
                        <td>{r.signal || "—"}</td>
                        <td>{formatNum(r.pred_close_q10, 2)}</td>
                        <td>{formatNum(r.pred_close_q50, 2)}</td>
                        <td>{formatNum(r.pred_close_q90, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <MetricGrid>
              <MetricCard label="Target Mode" value={model.data.metrics.target_mode || "point_return"} />
              <MetricCard label="Feature Rows" value={model.data.metrics.rows_after_feature_filter} />
              <MetricCard label="Features" value={model.data.metrics.features_count} />
              <MetricCard label="Test Window" value={`${model.data.metrics.test_days} days`} />
              <MetricCard label="Horizons" value={(model.data.metrics.horizons || []).map((h) => `${h.horizon}d`).join(", ")} />
            </MetricGrid>

            {!!(model.data.metrics.mode_comparison || []).length && (
              <div className="table-wrap" style={{ marginTop: 12 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Target Mode</th>
                      <th>Horizon</th>
                      <th>AUC</th>
                      <th>Brier</th>
                      <th>WF Folds</th>
                      <th>WF AUC</th>
                      <th>WF Brier</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(model.data.metrics.mode_comparison || []).flatMap((m) =>
                      (m.horizons || []).map((h) => (
                        <tr key={`${m.target_mode}-${h.horizon}`}>
                          <td>{m.target_mode}</td>
                          <td>{h.horizon}d</td>
                          <td>{formatNum(h.auc, 3)}</td>
                          <td>{formatNum(h.brier, 3)}</td>
                          <td>{h.wf_folds ?? "—"}</td>
                          <td>{formatNum(h.wf_auc, 3)}</td>
                          <td>{formatNum(h.wf_brier, 3)}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            )}

            <div className="table-wrap" style={{ marginTop: 12 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Horizon</th>
                    <th>Selected Feats</th>
                    <th>Train</th>
                    <th>Test</th>
                    <th>AUC</th>
                    <th>Brier</th>
                    <th>Pinball q10</th>
                    <th>Pinball q50</th>
                    <th>Pinball q90</th>
                    <th>WF Folds</th>
                    <th>WF AUC</th>
                    <th>WF Brier</th>
                  </tr>
                </thead>
                <tbody>
                  {model.data.metrics.horizons.map((h) => (
                    <tr key={h.horizon}>
                      <td>{h.horizon}d</td>
                      <td>{h.selected_feature_count ?? "—"}</td>
                      <td>{h.train_rows}</td>
                      <td>{h.test_rows}</td>
                      <td>{formatNum(h.auc, 3)}</td>
                      <td>{formatNum(h.brier, 3)}</td>
                      <td>{formatNum(h.pinball_q10, 4)}</td>
                      <td>{formatNum(h.pinball_q50, 4)}</td>
                      <td>{formatNum(h.pinball_q90, 4)}</td>
                      <td>{h.wf_folds ?? "—"}</td>
                      <td>{formatNum(h.wf_auc, 3)}</td>
                      <td>{formatNum(h.wf_brier, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="control-grid" style={{ marginTop: 14, marginBottom: 10 }}>
              <label className="control-field">
                <span className="control-label">Test Horizon</span>
                <select
                  className="control-input"
                  value={selectedModelHorizon}
                  onChange={(e) => setSelectedModelHorizon(Number(e.target.value))}
                >
                  {modelHorizons.map((h) => (
                    <option key={h} value={h}>{h}d</option>
                  ))}
                </select>
              </label>
            </div>

            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "lines",
                  name: "Pred q90",
                  x: modelPlot.x,
                  y: modelPlot.q90,
                  line: { color: "rgba(96, 153, 255, 0.85)", width: 1.5, dash: "dot" }
                },
                {
                  type: "scatter",
                  mode: "lines",
                  name: "Pred q10",
                  x: modelPlot.x,
                  y: modelPlot.q10,
                  fill: "tonexty",
                  fillcolor: "rgba(96, 153, 255, 0.28)",
                  line: { color: "rgba(96, 153, 255, 0.85)", width: 1.5, dash: "dot" }
                },
                {
                  type: "scatter",
                  mode: "lines+markers",
                  name: "Pred q50",
                  x: modelPlot.x,
                  y: modelPlot.q50,
                  line: { color: "#5f9dff", width: 2 },
                  marker: { size: 5, color: "#5f9dff" }
                },
                {
                  type: "scatter",
                  mode: "lines+markers",
                  name: "Actual",
                  x: modelPlot.x,
                  y: modelPlot.actual,
                  line: { color: "#75F37B", width: 2 },
                  marker: { size: 5, color: "#75F37B" }
                },
                {
                  type: "scatter",
                  mode: "lines",
                  name: "P(up) %",
                  x: modelPlot.x,
                  y: modelPlot.pUp,
                  yaxis: "y2",
                  line: { color: "#d058ff", width: 1.8, dash: "dot" }
                }
              ]}
              layout={{
                height: 330,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 40, r: 46, t: 14, b: 36 },
                xaxis: { tickfont: { color: "#9fb0c9", size: 10 } },
                yaxis: { title: "Close", tickfont: { color: "#9fb0c9", size: 10 } },
                yaxis2: {
                  overlaying: "y",
                  side: "right",
                  showgrid: false,
                  title: "P(up) %",
                  tickfont: { color: "#d058ff", size: 10 },
                  titlefont: { color: "#d058ff", size: 10 },
                  range: [0, 100]
                },
                legend: { orientation: "h", y: 1.1, x: 0, font: { color: "#c7d3e6", size: 10 } }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%", marginTop: 6 }}
            />
          </>
        )}
      </SectionCard>

      <SectionCard title="Feature Diagnostics">
        {model.loading ? (
          <LoadingState />
        ) : model.error || !model.data ? (
          <ErrorState message={String(model.error || "Model diagnostics unavailable")} />
        ) : (
          <>
            <div className="chart-panel" style={{ marginBottom: 12 }}>
              <h4>Feature Stability (Rolling IC, {selectedModelHorizon}d)</h4>
              {featureDiag.loading ? (
                <LoadingState />
              ) : featureDiag.error ? (
                <ErrorState message={String(featureDiag.error)} />
              ) : (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Full Corr</th>
                        <th>Median IC</th>
                        <th>Mean IC</th>
                        <th>IC IR</th>
                        <th>Sign Consistency</th>
                        <th>Coverage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stabilityRows.map((r) => (
                        <tr key={r.feature}>
                          <td>{r.feature}</td>
                          <td>{formatNum(r.full_corr, 3)}</td>
                          <td>{formatNum(r.median_ic, 3)}</td>
                          <td>{formatNum(r.mean_ic, 3)}</td>
                          <td>{formatNum(r.ic_ir, 3)}</td>
                          <td>{formatNum((Number(r.sign_consistency || 0) * 100), 1)}%</td>
                          <td>{formatNum((Number(r.coverage || 0) * 100), 1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="chart-panel" style={{ marginBottom: 12 }}>
              <h4>Collinearity Pairs (|corr| ≥ 0.9)</h4>
              {featureDiag.loading ? (
                <LoadingState />
              ) : featureDiag.error ? (
                <ErrorState message={String(featureDiag.error)} />
              ) : (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feature A</th>
                        <th>Feature B</th>
                        <th>|Corr|</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(featureDiag.data?.collinearity_pairs || []).slice(0, 30).map((r) => (
                        <tr key={`${r.feature_a}__${r.feature_b}`}>
                          <td>{r.feature_a}</td>
                          <td>{r.feature_b}</td>
                          <td>{formatNum(r.abs_corr, 3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="chart-panel" style={{ marginBottom: 12 }}>
              <h4>Feature to Spot Variance Explained ({selectedModelHorizon}d)</h4>
              <Plot
                data={[
                  {
                    type: "bar",
                    name: "Individual R² %",
                    x: varianceExplainedRows.map((r) => r.feature),
                    y: varianceExplainedRows.map((r) => Number(r.individual_r2 || 0) * 100),
                    marker: { color: "#2D7DFF" },
                  },
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    name: "Cumulative R² %",
                    x: varianceExplainedRows.map((r) => r.feature),
                    y: varianceExplainedRows.map((r) => Number(r.cumulative_r2 || 0) * 100),
                    yaxis: "y2",
                    line: { color: "#75F37B", width: 2 },
                  },
                ]}
                layout={{
                  height: 340,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 36, r: 36, t: 20, b: 110 },
                  xaxis: { tickangle: -35, tickfont: { color: "#9fb0c9", size: 9 } },
                  yaxis: { title: "Individual R² %", tickfont: { color: "#9fb0c9", size: 10 } },
                  yaxis2: {
                    overlaying: "y",
                    side: "right",
                    title: "Cumulative R² %",
                    tickfont: { color: "#75F37B", size: 10 },
                    titlefont: { color: "#75F37B", size: 10 },
                  },
                  legend: { orientation: "h", y: 1.1, x: 0, font: { color: "#c7d3e6", size: 10 } },
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
              <div className="table-wrap" style={{ marginTop: 10 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Feature</th>
                      <th>Individual R² %</th>
                      <th>Marginal R² %</th>
                      <th>Cumulative R² %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {varianceExplainedRows.map((r) => (
                      <tr key={`${r.rank}-${r.feature}`}>
                        <td>{r.rank}</td>
                        <td>{r.feature}</td>
                        <td>{formatNum(Number(r.individual_r2 || 0) * 100, 2)}</td>
                        <td>{formatNum(Number(r.marginal_r2 || 0) * 100, 2)}</td>
                        <td>{formatNum(Number(r.cumulative_r2 || 0) * 100, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="chart-grid-1x2">
              <div className="chart-panel">
                <h4>PCA Variance Decomposition</h4>
                <Plot
                  data={[
                    {
                      type: "bar",
                      name: "Explained %",
                      x: model.data.diagnostics?.pca?.components || [],
                      y: (model.data.diagnostics?.pca?.explained_variance_ratio || []).map((v) => Number(v) * 100),
                      marker: { color: "#2D7DFF" },
                    },
                    {
                      type: "scatter",
                      mode: "lines+markers",
                      name: "Cumulative %",
                      x: model.data.diagnostics?.pca?.components || [],
                      y: (model.data.diagnostics?.pca?.cumulative_explained_variance || []).map((v) => Number(v) * 100),
                      yaxis: "y2",
                      line: { color: "#75F37B", width: 2 },
                    },
                  ]}
                  layout={{
                    height: 320,
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    margin: { l: 36, r: 36, t: 20, b: 30 },
                    yaxis: { title: "Explained %", tickfont: { color: "#9fb0c9", size: 10 } },
                    yaxis2: {
                      overlaying: "y",
                      side: "right",
                      title: "Cumulative %",
                      tickfont: { color: "#75F37B", size: 10 },
                      titlefont: { color: "#75F37B", size: 10 },
                    },
                    legend: { orientation: "h", y: 1.1, x: 0, font: { color: "#c7d3e6", size: 10 } },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: "100%" }}
                />
              </div>

              <div className="chart-panel">
                <h4>Top Feature Correlations ({selectedModelHorizon}d)</h4>
                <Plot
                  data={[
                    {
                      type: "bar",
                      name: "Corr vs target_ret",
                      x: (topFeatureBars.ret || []).map((r) => r.feature),
                      y: (topFeatureBars.ret || []).map((r) => r.corr),
                      marker: { color: "#5f9dff" },
                    },
                    {
                      type: "bar",
                      name: "Corr vs target_up",
                      x: (topFeatureBars.up || []).map((r) => r.feature),
                      y: (topFeatureBars.up || []).map((r) => r.corr),
                      marker: { color: "#d058ff" },
                    },
                  ]}
                  layout={{
                    height: 320,
                    barmode: "group",
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    margin: { l: 36, r: 20, t: 20, b: 90 },
                    xaxis: { tickangle: -35, tickfont: { color: "#9fb0c9", size: 9 } },
                    yaxis: { tickfont: { color: "#9fb0c9", size: 10 } },
                    legend: { orientation: "h", y: 1.1, x: 0, font: { color: "#c7d3e6", size: 10 } },
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  useResizeHandler
                  style={{ width: "100%" }}
                />
              </div>
            </div>

            <div className="chart-panel" style={{ marginTop: 12 }}>
              <h4>Correlation Matrix (Core Features)</h4>
              <Plot
                data={[
                  {
                    type: "heatmap",
                    x: model.data.diagnostics?.corr_matrix?.features || [],
                    y: model.data.diagnostics?.corr_matrix?.features || [],
                    z: model.data.diagnostics?.corr_matrix?.values || [],
                    colorscale: "RdBu",
                    zmid: 0,
                    reversescale: true,
                    colorbar: { title: "Corr" },
                  },
                ]}
                layout={{
                  height: 520,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 140, r: 20, t: 20, b: 120 },
                  xaxis: { tickangle: -45, tickfont: { color: "#9fb0c9", size: 9 } },
                  yaxis: { tickfont: { color: "#9fb0c9", size: 9 } },
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            </div>
          </>
        )}
      </SectionCard>
    </>
  );
}
