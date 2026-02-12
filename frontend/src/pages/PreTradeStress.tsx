import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import DataTable from "../components/DataTable";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { formatInr, formatPct, formatNumber } from "../components/format";
import { useControls } from "../state/ControlsContext";
import { usePortfolio } from "../state/PortfolioContext";

type PreTrade = {
  spot: number;
  capital: number;
  greeks: Record<string, number>;
  normalized: Record<string, number>;
  iv_regime: string;
  zone: { num: number; name: string; color: string; message: string };
  comparison_table: Record<string, unknown>[];
  bucket_probs: Record<string, number>;
  bucket_rows: Record<string, unknown>[];
  threshold_rows: Record<string, unknown>[];
  repriced_rows: Record<string, unknown>[];
  scenario_table: Record<string, unknown>[];
};

export default function PreTradeStress() {
  const { summary, positions } = usePortfolio();
  const { setControls } = useControls();
  const [draft, setDraft] = useState<Record<string, number>>({});
  const [applied, setApplied] = useState<Record<string, number>>({});

  const defaults = useMemo(() => {
    const s = summary as any;
    const defaultSpot = positions?.current_spot ?? s?.current_spot ?? null;
    const defaultTheta = s?.greeks?.net_theta ?? null;
    const defaultGamma = s?.greeks?.net_gamma ?? null;
    const defaultVega = s?.greeks?.net_vega ?? null;
    const defaultCapital = s?.margin_used ?? s?.account_size ?? null;
    const out: Record<string, number> = {};
    if (defaultSpot != null) out.spot = Math.round(Number(defaultSpot));
    if (defaultTheta != null) out.theta = Math.round(Number(defaultTheta));
    if (defaultGamma != null) out.gamma = Number(Number(defaultGamma).toFixed(3));
    if (defaultVega != null) out.vega = Math.round(Number(defaultVega));
    if (defaultCapital != null) out.capital = Number((Number(defaultCapital) / 100000).toFixed(1));
    return out;
  }, [summary, positions]);

  useEffect(() => {
    if (!Object.keys(defaults).length) return;
    if (!Object.keys(draft).length) setDraft(defaults);
    if (!Object.keys(applied).length) setApplied(defaults);
  }, [defaults, draft, applied]);

  const normalizeCapital = (value?: number) => {
    if (value == null) return value;
    const v = Number(value);
    if (!Number.isFinite(v)) return value;
    return v > 1000 ? v / 100000 : v;
  };

  const query = useMemo(() => {
    const params = new URLSearchParams();
    if (applied.spot != null) params.set("spot", String(applied.spot));
    if (applied.theta != null) params.set("theta", String(applied.theta));
    if (applied.gamma != null) params.set("gamma", String(applied.gamma));
    if (applied.vega != null) params.set("vega", String(applied.vega));
    if (applied.capital != null) params.set("capital", String(applied.capital * 100000));
    const qs = params.toString();
    return qs ? `?${qs}` : "";
  }, [applied]);

  const { data, error, loading } = useCachedApi<PreTrade>(
    `pre_trade${query}`,
    `/pre-trade/analysis${query}`,
    60_000
  );

  const comparisonMap = useMemo(() => {
    const rows = data?.comparison_table || [];
    const map: Record<string, string> = {};
    rows.forEach((r) => {
      const key = String((r as any)["Metric"] || "").toLowerCase();
      const value = (r as any)["Your Position"];
      if (!key) return;
      map[key] = value != null ? String(value) : "";
    });
    return map;
  }, [data]);

  const scenarioStats = useMemo(() => {
    const probMap = new Map<string, number>();
    (data?.scenario_table || []).forEach((r: any) => {
      const name = String(r["Scenario"] || "");
      const p = Number(String(r["Probability"] || "0").replace("%", ""));
      if (name) probMap.set(name, Number.isFinite(p) ? p / 100 : 0);
    });
    const rows = (data?.repriced_rows || []).map((r) => {
      const scenario = String(r["Scenario"]);
      const pnl = Number(String(r["Repriced P&L (₹)"] || "0").replace(/[₹,]/g, ""));
      const prob = probMap.get(scenario) ?? 0;
      return { scenario, pnl, prob };
    }).filter((r) => Number.isFinite(r.pnl) && Number.isFinite(r.prob));

    const totalProb = rows.reduce((acc, r) => acc + r.prob, 0);
    if (!rows.length || totalProb <= 0) return null;

    const mean = rows.reduce((acc, r) => acc + r.pnl * r.prob, 0) / totalProb;
    const sorted = [...rows].sort((a, b) => a.pnl - b.pnl);

    const esAt = (tail: number) => {
      let cum = 0;
      let sum = 0;
      let taken = 0;
      for (const r of sorted) {
        if (cum >= tail) break;
        const take = Math.min(r.prob, tail - cum);
        sum += r.pnl * take;
        taken += take;
        cum += take;
      }
      return taken > 0 ? sum / taken : 0;
    };

    return {
      mean,
      es95: esAt(0.05),
      es99: esAt(0.01)
    };
  }, [data]);

  const pickComparison = (needle: string) => {
    const key = Object.keys(comparisonMap).find((k) => k.includes(needle));
    return key ? comparisonMap[key] : "";
  };

  const toneFromComparison = (value: string, fallback: "neutral" | "info" = "neutral") => {
    if (!value) return fallback;
    if (value.includes("🟢")) return "positive";
    if (value.includes("🟡")) return "warning";
    if (value.includes("🔴")) return "negative";
    if (value.includes("⚠️")) return "warning";
    return fallback;
  };

  const emojiOnly = (value: string) => {
    if (!value) return "";
    const match = value.match(/(🟢|🟡|🔴|⚠️)/);
    return match ? match[1] : "";
  };

  useEffect(() => {
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Spot</span>
          <input
            className="control-input"
            type="number"
            step="1"
            value={draft.spot ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, spot: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Capital (₹ Lacs)</span>
          <input
            className="control-input"
            type="number"
            step="0.1"
            value={draft.capital ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, capital: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Theta</span>
          <input
            className="control-input"
            type="number"
            step="1"
            value={draft.theta ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, theta: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Gamma</span>
          <input
            className="control-input"
            type="number"
            step="0.001"
            value={draft.gamma ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, gamma: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Vega</span>
          <input
            className="control-input"
            type="number"
            step="1"
            value={draft.vega ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, vega: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field control-inline" style={{ alignSelf: "end" }}>
          <button
            className="control-input"
            onClick={() => {
              setApplied({
                spot: draft.spot != null ? Math.round(draft.spot) : draft.spot,
                theta: draft.theta != null ? Math.round(draft.theta) : draft.theta,
                gamma: draft.gamma != null ? Number(Number(draft.gamma).toFixed(3)) : draft.gamma,
                vega: draft.vega != null ? Math.round(draft.vega) : draft.vega,
                capital: draft.capital != null ? Number(normalizeCapital(draft.capital)?.toFixed(1)) : draft.capital
              });
            }}
          >
            Apply
          </button>
        </label>
      </div>
    );
    setControls({
      title: "Controls",
      summary: (
        <>
          <span><span className="controls-summary-key">Spot</span> {draft.spot != null ? Math.round(draft.spot) : "—"}</span>
          <span><span className="controls-summary-key">Theta</span> {draft.theta != null ? Math.round(draft.theta) : "—"}</span>
          <span><span className="controls-summary-key">Gamma</span> {draft.gamma != null ? Number(draft.gamma).toFixed(3) : "—"}</span>
          <span><span className="controls-summary-key">Vega</span> {draft.vega != null ? Math.round(draft.vega) : "—"}</span>
          <span><span className="controls-summary-key">Capital</span> {draft.capital != null ? `${Number(draft.capital).toFixed(1)} lacs` : "—"}</span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, draft]);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      <SectionCard title="Stress Testing">
        {loading || !data ? <LoadingState /> : (
          <MetricGrid>
            <MetricCard label="Spot" value={formatInr(data.spot)} />
            <MetricCard label="Capital" value={formatInr(data.capital)} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Scenario KPIs">
        {loading ? <LoadingState /> : (
          <MetricGrid>
            <MetricCard label="Mean P&L" value={scenarioStats ? formatInr(scenarioStats.mean) : "—"} />
            <MetricCard label="ES95" value={scenarioStats ? formatInr(scenarioStats.es95) : "—"} />
            <MetricCard label="ES99" value={scenarioStats ? formatInr(scenarioStats.es99) : "—"} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Normalized Greeks (per ₹1L)">
        {loading || !data ? <LoadingState /> : (
          <MetricGrid>
            <MetricCard
              label="Delta"
              value={formatNumber(data.normalized.delta, 2)}
              delta={pickComparison("delta")}
              tone="info"
            />
            <MetricCard
              label="Theta"
              value={formatNumber(data.normalized.theta, 0)}
              delta={pickComparison("theta")}
              tone={toneFromComparison(pickComparison("theta"))}
            />
            <MetricCard
              label="Gamma"
              value={formatNumber(data.normalized.gamma, 4)}
              delta={emojiOnly(pickComparison("gamma"))}
              tone={toneFromComparison(pickComparison("gamma"))}
            />
            <MetricCard
              label="Vega"
              value={formatNumber(data.normalized.vega, 0)}
              delta={emojiOnly(pickComparison("vega"))}
              tone={toneFromComparison(pickComparison("vega"))}
            />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Scenario Repricing">
        {loading || !data ? <LoadingState /> : (
          <>
            {(() => {
              const probMap = new Map<string, string>();
              (data.scenario_table || []).forEach((r: any) => {
                const name = String(r["Scenario"] || "");
                const p = String(r["Probability"] || "");
                if (name) probMap.set(name, p);
              });
              const rows = (data.repriced_rows || []).map((r) => {
                const status = String(r["Breach"] || r["Status"] || r["status"] || "");
                const lossPct = Number(String(r["Loss % Capital"] || "0").replace("%", ""));
                const thresholdPct = Number(String(r["Threshold % NAV"] || "0").replace("%", ""));
                const isBad =
                  (Number.isFinite(lossPct) && Number.isFinite(thresholdPct) && lossPct > thresholdPct) ||
                  /fail|breach|red/i.test(status);
                const bubble = isBad ? "🔴" : "🟢";
                return { ...r, Probability: probMap.get(String(r["Scenario"])) || "—", Breach: `${bubble} ${status}`.trim() };
              });
              return (
            <DataTable
              columns={[
                { key: "Scenario", label: "Scenario" },
                { key: "dS% / dIV", label: "dS% / dIV" },
                { key: "Repriced P&L (₹)", label: "Repriced P&L" },
                { key: "Loss % Capital", label: "Loss % NAV" },
                { key: "Threshold % NAV", label: "Threshold % NAV" },
                { key: "Probability", label: "Prob %" },
                { key: "Breach", label: "Status" }
              ]}
              rows={rows}
            />
              );
            })()}
          </>
        )}
      </SectionCard>

      <SectionCard title="Scenario Loss Distribution">
        {loading || !data ? <LoadingState /> : (
          (() => {
            const probMap = new Map<string, number>();
            (data.scenario_table || []).forEach((r: any) => {
              const name = String(r["Scenario"] || "");
              const p = Number(String(r["Probability"] || "0").replace("%", ""));
              if (name) probMap.set(name, Number.isFinite(p) ? p : 0);
            });
            const rows = (data.repriced_rows || []).map((r) => {
              const lossPct = Number(String(r["Loss % Capital"] || "0").replace("%", ""));
              const pnl = Number(String(r["Repriced P&L (₹)"] || "0").replace(/[₹,]/g, ""));
              return {
                scenario: String(r["Scenario"]),
                lossPct,
                pnl,
                prob: probMap.get(String(r["Scenario"])) ?? 0
              };
            }).sort((a, b) => a.pnl - b.pnl);
            let cum = 0;
            const totalProb = rows.reduce((acc, r) => acc + (Number.isFinite(r.prob) ? r.prob : 0), 0);
            const cumProb = rows.map((r) => {
              const p = Number.isFinite(r.prob) ? r.prob : 0;
              cum += p;
              return cum;
            });
            return (
              <Plot
                data={[
                  {
                    type: "bar",
                    x: rows.map((r) => r.scenario),
                    y: rows.map((r) => r.pnl),
                    marker: {
                      color: rows.map((r) => (r.pnl < 0 ? "#ff4d4d" : "#2ecc71"))
                    },
                    name: "P&L (₹)"
                  },
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    x: rows.map((r) => r.scenario),
                    y: cumProb,
                    yaxis: "y2",
                    line: { color: "#F5B041", width: 2 },
                    marker: { color: "#F5B041", size: 6 },
                    name: "Cumulative %"
                  }
                ]}
                layout={{
                  height: 360,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 40, r: 40, t: 20, b: 70 },
                  xaxis: { tickangle: -30, tickfont: { color: "#b9c4d6", size: 10 }, automargin: true },
                  yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, title: "P&L (₹)" },
                  yaxis2: {
                    overlaying: "y",
                    side: "right",
                    range: [0, 100],
                    tickfont: { color: "#b9c4d6", size: 10 },
                    title: "Cum %"
                  },
                  legend: { orientation: "h", x: 0, y: -0.2, font: { color: "#b9c4d6", size: 10 } }
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            );
          })()
        )}
      </SectionCard>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">Zone Comparison Table (Debug)</div>
        </summary>
        <div className="section-body">
          {loading || !data ? <LoadingState /> : (
            <DataTable
              columns={[
                { key: "Metric", label: "Metric" },
                { key: "Zone 1 Range", label: "Zone 1" },
                { key: "Zone 2 Range", label: "Zone 2" },
                { key: "Zone 3 Range", label: "Zone 3" },
                { key: "Your Position", label: "Your Position" }
              ]}
              rows={data.comparison_table}
            />
          )}
        </div>
      </details>
    </>
  );
}
