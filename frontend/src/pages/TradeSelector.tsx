import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { formatInr, formatNumber } from "../components/format";
import { useControls } from "../state/ControlsContext";

type TradeSelector = { rows: Record<string, unknown>[]; underlying: string; expiry: string };

export default function TradeSelector() {
  const { data, error, loading } = useCachedApi<TradeSelector>("trade_selector", "/trade-selector/run", 60_000);
  const rows = data?.rows || [];
  const { setControls } = useControls();
  const [filters, setFilters] = useState({ minRr: 0, minPop: 0, minWidth: 200, maxWidth: 600 });

  useEffect(() => {
    if (!rows.length) return;
    const widths = rows.map((r) => Math.abs(Number(r.long_strike || 0) - Number(r.short_strike || 0))).filter((v) => Number.isFinite(v));
    if (!widths.length) return;
    const max = Math.max(...widths);
    setFilters((prev) => ({
      ...prev,
      maxWidth: Number.isFinite(prev.maxWidth) && prev.maxWidth > 0 ? prev.maxWidth : max
    }));
  }, [rows]);

  useEffect(() => {
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Min RR</span>
          <input
            className="control-input"
            type="number"
            step="0.1"
            value={filters.minRr}
            onChange={(e) => setFilters((prev) => ({ ...prev, minRr: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Min PoP %</span>
          <input
            className="control-input"
            type="number"
            step="1"
            value={filters.minPop}
            onChange={(e) => setFilters((prev) => ({ ...prev, minPop: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Min Width</span>
          <input
            className="control-input"
            type="number"
            step="50"
            value={filters.minWidth}
            onChange={(e) => setFilters((prev) => ({ ...prev, minWidth: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Max Width</span>
          <input
            className="control-input"
            type="number"
            step="50"
            value={filters.maxWidth}
            onChange={(e) => setFilters((prev) => ({ ...prev, maxWidth: Number(e.target.value) }))}
          />
        </label>
      </div>
    );
    setControls({
      title: "Controls",
      summary: (
        <>
          <span><span className="controls-summary-key">Min RR</span> {filters.minRr}</span>
          <span><span className="controls-summary-key">Min PoP</span> {filters.minPop}%</span>
          <span><span className="controls-summary-key">Min Width</span> {filters.minWidth}</span>
          <span><span className="controls-summary-key">Max Width</span> {filters.maxWidth}</span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, filters]);

  const filteredRows = useMemo(() => {
    return rows.filter((r) => {
      const rr = Number(r.rr || 0);
      const pop = Number(r.pop || 0) * 100;
      const width = Math.abs(Number(r.long_strike || 0) - Number(r.short_strike || 0));
      if (rr < filters.minRr) return false;
      if (pop < filters.minPop) return false;
      if (Number.isFinite(width) && width < filters.minWidth) return false;
      if (Number.isFinite(width) && width > filters.maxWidth) return false;
      return true;
    });
  }, [rows, filters]);

  const chartRows = filteredRows.filter((r) => Number(r.pop || 0) < 1.0);
  const xVals = chartRows.map((r) => Number(r.expected_pnl || 0));
  const yVals = chartRows.map((r) => Number(r.tail_loss || 0));
  const xMax = Math.max(1, ...xVals);
  const yMax = Math.max(1, ...yVals);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      <SectionCard title="Top Candidates">
        {loading || !data ? <LoadingState /> : (
          <DataTable
            maxHeight={400}
            columns={[
              { key: "strategy", label: "Strategy" },
              { key: "expiry", label: "Expiry" },
              { key: "short_strike", label: "Short" },
              { key: "long_strike", label: "Long" },
              { key: "premium", label: "Premium", format: (v) => formatInr(v) },
              { key: "max_profit", label: "Max Profit", format: (v) => formatInr(v) },
              { key: "max_loss", label: "Max Loss", format: (v) => formatInr(v) },
              { key: "tail_loss", label: "Tail Loss", format: (v) => formatInr(v) },
              { key: "rr", label: "RR", format: (v) => formatNumber(v, 2) },
              { key: "pop", label: "PoP", format: (v) => formatNumber((Number(v) || 0) * 100, 1) + "%" },
              { key: "expected_pnl", label: "Expected PnL", format: (v) => formatInr(v) }
            ]}
            rows={filteredRows.slice(0, 10)}
          />
        )}
      </SectionCard>

      <SectionCard title="Tail Loss vs Mean PnL">
        {loading || !data ? <LoadingState /> : (
          <Plot
            data={[
              {
                type: "scatter",
                mode: "markers",
                x: xVals,
                y: yVals,
                marker: {
                  color: chartRows.map((r) => Number(r.rr || 0)),
                  cmin: 0,
                  cmax: 2,
                  colorscale: [
                    [0, "#e57373"],
                    [0.5, "#f5b041"],
                    [1, "#2ecc71"]
                  ],
                  showscale: false,
                  size: 8,
                  opacity: 0.85,
                  line: { color: "rgba(0,0,0,0.25)", width: 1 }
                }
              },
              ...[0.15, 0.3, 0.5].map((rr) => ({
                type: "scatter",
                mode: "lines",
                x: [0, xMax],
                y: [0, xMax / rr],
                line: { dash: "dot", color: "#6C7A89" },
                name: `RR ${rr}`
              }))
            ]}
              layout={{
                height: 420,
                autosize: true,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 50, r: 30, t: 10, b: 36 },
                xaxis: { title: "Expected PnL", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                yaxis: { title: "Tail Loss (CVaR)", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                legend: { orientation: "h", x: 0, y: -0.2, font: { color: "#b9c4d6", size: 10 } },
                shapes: [
                {
                  type: "rect",
                  x0: 0,
                  x1: xMax,
                  y0: xMax / 0.15,
                  y1: yMax * 1.2,
                  fillcolor: "rgba(217,83,79,0.08)",
                  line: { width: 0 }
                },
                {
                  type: "rect",
                  x0: 0,
                  x1: xMax,
                  y0: xMax / 0.3,
                  y1: xMax / 0.15,
                  fillcolor: "rgba(255,176,32,0.06)",
                  line: { width: 0 }
                },
                {
                  type: "rect",
                  x0: 0,
                  x1: xMax,
                  y0: xMax / 0.5,
                  y1: xMax / 0.3,
                  fillcolor: "rgba(46,196,182,0.08)",
                  line: { width: 0 }
                },
                {
                  type: "rect",
                  x0: 0,
                  x1: xMax,
                  y0: 0,
                  y1: xMax / 0.5,
                  fillcolor: "rgba(45,125,255,0.06)",
                  line: { width: 0 }
                }
              ]
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <SectionCard title="All Candidates">
        {loading || !data ? <LoadingState /> : (
          <DataTable
            maxHeight={400}
            columns={[
              { key: "strategy", label: "Strategy" },
              { key: "expiry", label: "Expiry" },
              { key: "short_strike", label: "Short" },
              { key: "long_strike", label: "Long" },
              { key: "premium", label: "Premium", format: (v) => formatInr(v) },
              { key: "max_profit", label: "Max Profit", format: (v) => formatInr(v) },
              { key: "max_loss", label: "Max Loss", format: (v) => formatInr(v) },
              { key: "tail_loss", label: "Tail Loss", format: (v) => formatInr(v) },
              { key: "rr", label: "RR", format: (v) => formatNumber(v, 2) },
              { key: "pop", label: "PoP", format: (v) => formatNumber((Number(v) || 0) * 100, 1) + "%" },
              { key: "expected_pnl", label: "Expected PnL", format: (v) => formatInr(v) }
            ]}
            rows={filteredRows}
          />
        )}
      </SectionCard>
    </>
  );
}
