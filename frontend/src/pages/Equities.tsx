import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import Plot from "../components/Plot";
import { formatInr, formatPct } from "../components/format";
import { useCachedApi } from "../hooks/useCachedApi";
import { useControls } from "../state/ControlsContext";

type EquitySummary = {
  equity_sleeve_value: number | null;
  allocation_pct: number | null;
  allocation_denom?: string;
  sleeve_drawdown_pct: number | null;
  stress_loss_inr: number | null;
  stress_loss_pct: number | null;
  equity_es99_contrib_pct: number | null;
  pct_underwater: number | null;
  weighted_time_under_water_days: number | null;
  capital_base?: number;
  capital_base_label?: string;
  warnings?: { history_unavailable?: boolean; drawdown_history_unavailable?: boolean; es99_available?: boolean };
  shock_losses?: { level: number; loss: number }[];
  alloc_series?: { symbol: string; value: number }[];
  top_stress?: { symbol: string; value: number }[];
  risk_concentration?: { top1_pct: number; top5_pct: number; top1_stress_pct: number; top5_stress_pct: number };
  rows: Record<string, unknown>[];
};

export default function Equities() {
  const [shockLevels, setShockLevels] = useState<string>("-5,-10,-20,-30");
  const [currentShock, setCurrentShock] = useState<string>("-10");
  const [lookbackDays, setLookbackDays] = useState<string>("365");
  const [capitalBase, setCapitalBase] = useState<string>("auto");
  const [useCache, setUseCache] = useState<boolean>(true);
  const { setControls } = useControls();

  const query = `?shock_levels=${encodeURIComponent(shockLevels)}&current_shock=${encodeURIComponent(currentShock)}&lookback_days=${encodeURIComponent(lookbackDays)}&capital_base=${encodeURIComponent(capitalBase)}&use_cache=${useCache}`;
  const { data, error, loading } = useCachedApi<EquitySummary>(
    "equities_summary",
    `/equities/summary${query}`,
    60_000
  );

  useEffect(() => {
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Shock levels (%)</span>
          <input className="control-input" value={shockLevels} onChange={(e) => setShockLevels(e.target.value)} />
        </label>
        <label className="control-field">
          <span className="control-label">Current shock</span>
          <input className="control-input" value={currentShock} onChange={(e) => setCurrentShock(e.target.value)} />
        </label>
        <label className="control-field">
          <span className="control-label">History lookback (days)</span>
          <input className="control-input" value={lookbackDays} onChange={(e) => setLookbackDays(e.target.value)} />
        </label>
        <label className="control-field">
          <span className="control-label">Equity sleeve capital base</span>
          <select className="control-input" value={capitalBase} onChange={(e) => setCapitalBase(e.target.value)}>
            <option value="auto">Auto (cost if available)</option>
            <option value="cost_basis">Cost basis</option>
            <option value="market_value">Market value</option>
          </select>
        </label>
        <label className="control-field control-inline">
          <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} />
          <span className="control-label">Use Cache</span>
        </label>
      </div>
    );
    setControls({
      key: `equities:${shockLevels}:${currentShock}:${lookbackDays}:${capitalBase}:${useCache}`,
      title: "Controls",
      summary: (
        <>
          <span>
            <span className="controls-summary-key">Shock</span> {currentShock}%
          </span>
          <span>
            <span className="controls-summary-key">Lookback</span> {lookbackDays}d
          </span>
          <span>
            <span className="controls-summary-key">Base</span> {capitalBase}
          </span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, shockLevels, currentShock, lookbackDays, capitalBase, useCache]);

  const rows = data?.rows || [];

  const charts = useMemo(() => {
    const byValue = [...rows].sort((a, b) => Number(b["market_value"] || 0) - Number(a["market_value"] || 0)).slice(0, 10);
    const byStress = [...rows].sort((a, b) => Number(a["stress_loss"] || 0) - Number(b["stress_loss"] || 0)).slice(0, 10);
    const byEs99 = [...rows].filter((r) => Number(r["es99_inr"]) > 0).sort((a, b) => Number(b["es99_inr"] || 0) - Number(a["es99_inr"] || 0)).slice(0, 10);
    return { byValue, byStress, byEs99 };
  }, [rows]);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}

      {/* Controls moved to sticky ControlsBar */}

      <SectionCard title="Equity Sleeve KPIs">
        {loading || !data ? (
          <LoadingState />
        ) : (
          <MetricGrid>
            <MetricCard label="Equity Sleeve Value" value={formatInr(data.equity_sleeve_value)} />
            <MetricCard label="Allocation %" value={formatPct(data.allocation_pct)} delta={data.allocation_denom ? `Denominator: ${data.allocation_denom}` : undefined} />
            <MetricCard
              label="Sleeve Drawdown %"
              value={formatPct(data.sleeve_drawdown_pct)}
              tone={(data.sleeve_drawdown_pct ?? 0) < 0 ? "negative" : "neutral"}
            />
            <MetricCard
              label="Stress Loss"
              value={formatInr(data.stress_loss_inr)}
              delta={formatPct(data.stress_loss_pct)}
              tone={(data.stress_loss_inr ?? 0) < 0 ? "negative" : "neutral"}
            />
            <MetricCard label="ES99 Contribution %" value={formatPct(data.equity_es99_contrib_pct)} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Under-water Summary">
        {loading || !data ? (
          <LoadingState />
        ) : (
          <MetricGrid>
            <MetricCard label="Holdings Under Water" value={formatPct(data.pct_underwater)} />
            <MetricCard label="Weighted Avg Time Under Water" value={data.weighted_time_under_water_days ? `${Math.round(data.weighted_time_under_water_days)}d` : "N/A"} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Risk Concentration KPIs">
        {loading || !data ? <LoadingState /> : (
          <MetricGrid>
            <MetricCard label="Top 1 Holding %" value={formatPct(data.risk_concentration?.top1_pct)} />
            <MetricCard label="Top 5 Holdings %" value={formatPct(data.risk_concentration?.top5_pct)} />
            <MetricCard label="Top 1 Stress Contrib %" value={formatPct(data.risk_concentration?.top1_stress_pct)} />
            <MetricCard label="Top 5 Stress Contrib %" value={formatPct(data.risk_concentration?.top5_stress_pct)} />
          </MetricGrid>
        )}
      </SectionCard>

      {data?.warnings?.history_unavailable ? (
        <SectionCard>
          <div>Time under water: Kite history unavailable (check login/credentials).</div>
        </SectionCard>
      ) : null}
      {data?.warnings?.drawdown_history_unavailable ? (
        <SectionCard>
          <div>Drawdown needs Kite history; currently unavailable.</div>
        </SectionCard>
      ) : null}
      {data && data.warnings && data.warnings.es99_available === false ? (
        <SectionCard>
          <div>ES99 contribution not available for equities (unable to compute per-symbol ES99).</div>
        </SectionCard>
      ) : null}

      <SectionCard title="Equity Scenario Table">
        {loading || !data ? (
          <LoadingState />
        ) : (
          <DataTable
            columns={[
              { key: "symbol", label: "Symbol" },
              { key: "qty", label: "Qty" },
              { key: "avg_cost", label: "Avg Cost", format: (v) => formatInr(v) },
              { key: "ltp", label: "LTP", format: (v) => formatInr(v) },
              { key: "market_value", label: "Market Value", format: (v) => formatInr(v) },
              { key: "unreal_pnl", label: "Unreal PnL", format: (v) => formatInr(v) },
              { key: "unreal_pnl_pct", label: "Unreal PnL %", format: (v) => formatPct(v) },
              { key: "return_vs_cost_pct", label: "Return vs Cost %", format: (v) => formatPct(v) },
              { key: "drawdown_pct", label: "Drawdown %", format: (v) => formatPct(v) },
              { key: "time_under_water", label: "Time Under Water" },
              { key: "stress_loss", label: "Stress Loss", format: (v) => formatInr(v) },
              { key: "es99_inr", label: "ES99", format: (v) => formatInr(v) },
              { key: "es99_pct", label: "ES99 %", format: (v) => formatPct(v) }
            ]}
            rows={rows}
          />
        )}
      </SectionCard>

      <div className="chart-grid-1x2">
        <SectionCard title="Scenario Comparison (Sleeve Stress Loss)">
          {!rows.length ? <LoadingState /> : (
            <Plot
              data={[{
                type: "bar",
                x: (data?.shock_losses || []).map((r) => `${r.level}%`),
                y: (data?.shock_losses || []).map((r) => Number(r.loss || 0)),
                marker: { color: "#4FBFD6" },
              }]}
              layout={{
                height: 300,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 30, r: 12, t: 20, b: 40 },
                bargap: 0.35,
                xaxis: { tickfont: { size: 10, color: "#9fb2c7" } },
                yaxis: { tickfont: { size: 10, color: "#9fb2c7" }, title: "₹" }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          )}
        </SectionCard>

        <SectionCard title="Allocation by Holding">
          {!rows.length ? <LoadingState /> : (
            <Plot
              data={[
                {
                  type: "bar",
                  x: charts.byValue.map((r) => r.symbol as string),
                  y: charts.byValue.map((r) => Number(r.market_value || 0)),
                  marker: { color: "#2D7DFF" },
                  width: 0.45
                }
              ]}
              layout={{
                height: 300,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 30, r: 12, t: 20, b: 40 },
                bargap: 0.35,
                xaxis: { tickangle: -30, tickfont: { size: 10, color: "#9fb2c7" } },
                yaxis: { title: "₹", tickfont: { size: 10, color: "#9fb2c7" } }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          )}
        </SectionCard>
      </div>

      <div className="chart-grid-1x2" style={{ marginTop: "12px" }}>
        <SectionCard title="Top Stress Contributors">
          {!rows.length ? <LoadingState /> : (
            <Plot
              data={[
                {
                  type: "bar",
                  x: charts.byStress.map((r) => r.symbol as string),
                  y: charts.byStress.map((r) => Number(r.stress_loss || 0)),
                  marker: { color: "#FF4D4D" },
                  width: 0.45
                }
              ]}
              layout={{
                height: 300,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 30, r: 12, t: 20, b: 40 },
                bargap: 0.35,
                xaxis: { tickangle: -30, tickfont: { size: 10, color: "#9fb2c7" } },
                yaxis: { title: "₹", tickfont: { size: 10, color: "#9fb2c7" } }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          )}
        </SectionCard>

        {charts.byEs99.length ? (
          <SectionCard title="ES99 Contribution by Holding">
            <Plot
              data={[
                {
                  type: "bar",
                  x: charts.byEs99.map((r) => r.symbol as string),
                  y: charts.byEs99.map((r) => Number(r.es99_inr || 0)),
                  marker: { color: "#FFB020" },
                  width: 0.45
                }
              ]}
              layout={{
                height: 300,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 30, r: 12, t: 20, b: 40 },
                bargap: 0.35,
                xaxis: { tickangle: -30, tickfont: { size: 10, color: "#9fb2c7" } },
                yaxis: { title: "₹", tickfont: { size: 10, color: "#9fb2c7" } }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </SectionCard>
        ) : (
          <div />
        )}
      </div>

    </>
  );
}
