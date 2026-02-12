import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { formatInr, formatPct, formatNumber } from "../components/format";
import { apiFetch } from "../api/client";
import { useControls } from "../state/ControlsContext";

type TradeResponse = { rows: Record<string, unknown>[] };

type TradeDetail = {
  scenario_rows: Record<string, unknown>[];
  payoff: Record<string, unknown>[];
  expiry_payoff: Record<string, unknown>[];
  legs: Record<string, unknown>[];
  iv_debug: Record<string, unknown>[];
  scenario_dist: { scenario: string; pnl_inr: number; prob_pct: number; cum_prob: number }[];
};
type TradeDetailsResponse = { details: Record<string, TradeDetail>; trade_ids: string[] };

export default function RiskBucketsTrade() {
  const { data, error, loading } = useCachedApi<TradeResponse>(
    "risk_buckets_trades",
    "/risk-buckets/trades",
    60_000
  );
  const details = useCachedApi<TradeDetailsResponse>(
    "risk_buckets_trade_details",
    "/risk-buckets/trade-details",
    60_000
  );
  const [selectedTrade, setSelectedTrade] = useState<string>("");
  const [showGroup, setShowGroup] = useState<boolean>(true);
  const { setControls } = useControls();

  const scatter = useMemo(() => {
    const rows = data?.rows || [];
    return {
      x: rows.map((r) => Number(r.expected_pnl_inr || 0)),
      y: rows.map((r) => Number(r.tail_loss_inr || 0)),
      text: rows.map((r) => String(r.trade_id || ""))
    };
  }, [data]);

  const filteredRows = useMemo(() => {
    const rows = data?.rows || [];
    return rows.filter((r) => {
      const isGroup = Boolean(r.is_group_trade);
      if (!showGroup && isGroup) return false;
      return true;
    });
  }, [data, showGroup]);

  const xMax = Math.max(1, ...filteredRows.map((r) => Number(r.expected_pnl_inr || 0)));
  const yMax = Math.max(1, ...filteredRows.map((r) => Number(r.tail_loss_inr || 0)));

  const probPct = useMemo(() => {
    return filteredRows.map((r) => {
      const direct = Number((r as any).prob_profit ?? (r as any).pop_pct ?? (r as any).probability ?? (r as any).prob_pct);
      if (Number.isFinite(direct) && direct > 0) return direct;
      const lossPct = Number((r as any).prob_loss_pct ?? (r as any).prob_loss);
      if (Number.isFinite(lossPct) && lossPct > 0) return Math.max(0, Math.min(100, 100 - lossPct));
      return 50;
    });
  }, [filteredRows]);

  const bubbleSizes = useMemo(() => {
    return probPct.map((p) => {
      if (p >= 75) return 16;
      if (p >= 50) return 12;
      return 9;
    });
  }, [probPct]);

  const bubbleColors = useMemo(() => {
    return filteredRows.map((r) => {
      const rr = Number(r.risk_reward || 0);
      if (rr >= 1.0) return "#2ecc71";
      if (rr >= 0.5) return "#f5b041";
      return "#e57373";
    });
  }, [filteredRows]);

  const firstTrade = filteredRows?.[0]?.trade_id as string | undefined;

  useEffect(() => {
    if (!selectedTrade && firstTrade) {
      setSelectedTrade(firstTrade);
    }
  }, [firstTrade, selectedTrade]);

  const detail = selectedTrade ? details.data?.details?.[selectedTrade] ?? null : null;

  const payoff = useMemo(() => {
    const rows = detail?.payoff || [];
    const expiryRows = detail?.expiry_payoff || [];
    const x = rows.map((r) => Number(r.spot || 0));
    const y = rows.map((r) => Number(r.pnl || 0));
    const yPos = y.map((v) => (v >= 0 ? v : null));
    const yNeg = y.map((v) => (v < 0 ? v : null));
    const expiryX = expiryRows.map((r) => Number(r.spot || 0));
    const expiryY = expiryRows.map((r) => Number(r.pnl || 0));
    const breakevens: number[] = [];
    for (let i = 0; i < expiryY.length - 1; i++) {
      const y0 = expiryY[i];
      const y1 = expiryY[i + 1];
      if (y0 === 0) breakevens.push(expiryX[i]);
      if (y0 === 0 || y1 === 0) continue;
      if ((y0 > 0 && y1 < 0) || (y0 < 0 && y1 > 0)) {
        const t = Math.abs(y0) / (Math.abs(y0) + Math.abs(y1));
        breakevens.push(expiryX[i] + (expiryX[i + 1] - expiryX[i]) * t);
      }
    }
    const beLow = breakevens.length ? Math.min(...breakevens) : null;
    const beHigh = breakevens.length ? Math.max(...breakevens) : null;
    const greenFill = breakevens.length === 1
      ? expiryY.map((v) => (v > 0 ? v : null))
      : breakevens.length
        ? expiryX.map((xi, idx) => (xi >= (beLow as number) && xi <= (beHigh as number) && expiryY[idx] > 0 ? expiryY[idx] : null))
        : expiryX.map(() => null);
    const redFill = breakevens.length === 1
      ? expiryY.map((v) => (v < 0 ? v : null))
      : breakevens.length
        ? expiryX.map((xi, idx) => ((xi < (beLow as number) || xi > (beHigh as number)) && expiryY[idx] < 0 ? expiryY[idx] : null))
        : expiryX.map(() => null);
    const bePoints = breakevens.map((b) => ({ x: b, y: 0 }));
    return { x, y, yPos, yNeg, expiryX, expiryY, beLow, beHigh, greenFill, redFill, bePoints, hasPositiveExpiry: expiryY.some((v) => v > 0) };
  }, [detail]);
  const beLow = payoff.beLow ?? NaN;
  const beHigh = payoff.beHigh ?? NaN;

  const convexity = useMemo(() => {
    const x = payoff.x;
    const y = payoff.y;
    if (x.length < 3) return { x: [], c: [] as number[] };
    const c = new Array(y.length).fill(0);
    for (let i = 1; i < y.length - 1; i++) {
      const step = x[i + 1] - x[i] || 1;
      c[i] = Math.abs((y[i + 1] - 2 * y[i] + y[i - 1]) / (step * step));
    }
    const max = Math.max(...c, 1);
    return { x, c: c.map((v) => v / max) };
  }, [payoff]);

  const payoffXRange = useMemo(() => {
    if (!payoff.expiryX.length) return null;
    const min = Math.min(...payoff.expiryX);
    const max = Math.max(...payoff.expiryX);
    return [min, max] as [number, number];
  }, [payoff.expiryX]);

  const currentSpot = useMemo(() => {
    const legs = detail?.legs || [];
    for (const leg of legs) {
      const spot = Number((leg as any).spot_price || (leg as any).underlying_value || 0);
      if (Number.isFinite(spot) && spot > 0) return spot;
    }
    return payoff.x.length ? payoff.x[Math.floor(payoff.x.length / 2)] : 0;
  }, [detail, payoff.x]);

  useEffect(() => {
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Trade</span>
          <select className="control-input" value={selectedTrade} onChange={(e) => setSelectedTrade(e.target.value)}>
            {(details.data?.trade_ids || filteredRows.map((r) => String(r.trade_id))).map((id) => (
              <option key={id} value={id}>{id}</option>
            ))}
          </select>
        </label>
        <label className="control-field control-inline">
          <input type="checkbox" checked={showGroup} onChange={(e) => setShowGroup(e.target.checked)} />
          <span className="control-label">Include Group Trades</span>
        </label>
      </div>
    );
    setControls({
      title: "Controls",
      summary: (
        <>
          <span>
            <span className="controls-summary-key">Trade</span> {selectedTrade || "—"}
          </span>
          <span>
            <span className="controls-summary-key">Group</span> {showGroup ? "On" : "Off"}
          </span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, selectedTrade, showGroup, details.data, filteredRows]);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      {/* Controls moved to sticky ControlsBar */}

      <SectionCard title="Tail Loss vs Mean PnL">
        {loading || !data ? <LoadingState /> : (
          <div style={{ width: "100%" }}>
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "markers",
                  x: filteredRows.map((r) => Number(r.expected_pnl_inr || 0)),
                  y: filteredRows.map((r) => Number(r.tail_loss_inr || 0)),
                  text: filteredRows.map((r) => String(r.trade_id || "")),
                  customdata: filteredRows.map((r, idx) => [Number(r.risk_reward || 0), probPct[idx]]),
                  hovertemplate: "<b>%{text}</b><br>Mean PnL: %{x:.0f}<br>Tail Loss: %{y:.0f}<br>RR: %{customdata[0]:.2f}<br>Prob: %{customdata[1]:.0f}%<extra></extra>",
                  marker: {
                    color: filteredRows.map((r) => Number(r.risk_reward || 0)),
                    cmin: 0,
                    cmax: 2,
                    colorscale: [
                      [0, "#e57373"],
                      [0.5, "#f5b041"],
                      [1, "#2ecc71"]
                    ],
                    showscale: true,
                    colorbar: { title: "RR", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                    size: bubbleSizes,
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
                height: 460,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                xaxis: { title: "Mean PnL", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                yaxis: { title: "Tail Loss", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                margin: { l: 30, r: 12, t: 10, b: 40 },
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
          </div>
        )}
      </SectionCard>

      <SectionCard title="Payoff Graph">
        {details.error ? <ErrorState message={String(details.error)} /> : null}
        {detail ? (
          <div className="chart-stack">
            <div className="chart-panel">
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines",
                    x: payoff.expiryX,
                    y: payoff.greenFill,
                    line: { color: "rgba(0,0,0,0)" },
                    fill: "tozeroy",
                    fillcolor: "rgba(46, 204, 113, 0.22)",
                    name: "Profit Zone"
                  },
                  {
                    type: "scatter",
                    mode: "lines",
                    x: payoff.expiryX,
                    y: payoff.redFill,
                    line: { color: "rgba(0,0,0,0)" },
                    fill: "tozeroy",
                    fillcolor: "rgba(231, 76, 60, 0.12)",
                    name: "Loss Zone"
                  },
                  {
                    type: "scatter",
                    mode: "markers",
                    x: payoff.bePoints.map((p) => p.x),
                    y: payoff.bePoints.map((p) => p.y),
                    marker: { color: "#f5b041", size: 6 },
                    name: "Breakeven"
                  },
                  { type: "scatter", mode: "lines", x: payoff.expiryX, y: payoff.expiryY, line: { color: "#2ecc71", width: 2 }, name: "On Expiry" },
                  { type: "scatter", mode: "lines", x: payoff.x, y: payoff.y, line: { color: "#6CC0FF", width: 2 }, name: "On Target" }
                ]}
                layout={{
                  height: 360,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 20, r: 6, t: 6, b: 54 },
                  xaxis: { title: "Spot", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                  yaxis: { title: "P&L", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                  legend: { orientation: "h", x: 0, y: -0.2, font: { color: "#b9c4d6", size: 10 } },
                  shapes: [
                    {
                      type: "line",
                      x0: payoff.x[0],
                      x1: payoff.x[payoff.x.length - 1],
                      y0: 0,
                      y1: 0,
                      line: { color: "rgba(148,163,184,0.6)", width: 1 }
                    },
                    ...(Number.isFinite(beLow) && Number.isFinite(beHigh) && beLow <= beHigh ? [
                      {
                        type: "line",
                        x0: beLow,
                        x1: beLow,
                        y0: Math.min(...(payoff.expiryY as number[]), -1),
                        y1: Math.max(...(payoff.expiryY as number[]), 1),
                        line: { color: "rgba(125, 140, 160, 0.5)", width: 1, dash: "dot" }
                      },
                      {
                        type: "line",
                        x0: beHigh,
                        x1: beHigh,
                        y0: Math.min(...(payoff.expiryY as number[]), -1),
                        y1: Math.max(...(payoff.expiryY as number[]), 1),
                        line: { color: "rgba(125, 140, 160, 0.5)", width: 1, dash: "dot" }
                      }
                    ] : []),
                    {
                      type: "line",
                      x0: currentSpot,
                      x1: currentSpot,
                      y0: Math.min(...(payoff.expiryY as number[]), -1),
                      y1: Math.max(...(payoff.expiryY as number[]), 1),
                      line: { color: "rgba(46,204,113,0.7)", width: 1, dash: "dot" }
                    },
                  ]
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            </div>
            <div className="chart-panel">
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines",
                    x: payoff.expiryX,
                    y: payoff.greenFill,
                    line: { color: "rgba(0,0,0,0)" },
                    fill: "tozeroy",
                    fillcolor: "rgba(46, 204, 113, 0.22)",
                    name: "Profit Zone"
                  },
                  {
                    type: "scatter",
                    mode: "markers",
                    x: payoff.bePoints.map((p) => p.x),
                    y: payoff.bePoints.map((p) => p.y),
                    marker: { color: "#f5b041", size: 6 },
                    name: "Breakeven"
                  },
                  { type: "scatter", mode: "lines", x: payoff.expiryX, y: payoff.expiryY, line: { color: "#2ecc71", width: 2 }, name: "On Expiry" },
                  { type: "scatter", mode: "lines", x: payoff.x, y: payoff.y, line: { color: "#6CC0FF", width: 2 }, name: "On Target" },
                ]}
                layout={{
                  height: 360,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 20, r: 30, t: 6, b: 54 },
                  xaxis: {
                    title: "Spot",
                    tickfont: { color: "#b9c4d6", size: 10 },
                    titlefont: { color: "#b9c4d6", size: 11 },
                    range: payoffXRange || undefined
                  },
                  yaxis: { title: "P&L", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                  legend: { orientation: "h", x: 0, y: -0.2, font: { color: "#b9c4d6", size: 10 } },
                  shapes: [
                    {
                      type: "line",
                      x0: payoff.x[0],
                      x1: payoff.x[payoff.x.length - 1],
                      y0: 0,
                      y1: 0,
                      line: { color: "rgba(148,163,184,0.6)", width: 1 }
                    },
                    ...(Number.isFinite(beLow) && Number.isFinite(beHigh) && beLow <= beHigh ? [
                      {
                        type: "line",
                        x0: beLow,
                        x1: beLow,
                        y0: Math.min(...(payoff.expiryY as number[]), -1),
                        y1: Math.max(...(payoff.expiryY as number[]), 1),
                        line: { color: "rgba(125, 140, 160, 0.5)", width: 1, dash: "dot" }
                      },
                      {
                        type: "line",
                        x0: beHigh,
                        x1: beHigh,
                        y0: Math.min(...(payoff.expiryY as number[]), -1),
                        y1: Math.max(...(payoff.expiryY as number[]), 1),
                        line: { color: "rgba(125, 140, 160, 0.5)", width: 1, dash: "dot" }
                      }
                    ] : []),
                    {
                      type: "line",
                      x0: currentSpot,
                      x1: currentSpot,
                      y0: Math.min(...(payoff.expiryY as number[]), -1),
                      y1: Math.max(...(payoff.expiryY as number[]), 1),
                      line: { color: "rgba(46,204,113,0.7)", width: 1, dash: "dot" }
                    },
                    {
                      type: "rect",
                      x0: payoff.expiryX[0],
                      x1: payoff.expiryX[payoff.expiryX.length - 1],
                      y0: Math.min(...(payoff.expiryY as number[]), -1),
                      y1: 0,
                      fillcolor: "rgba(231,76,60,0.12)",
                      line: { width: 0 }
                    }
                  ]
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            </div>
            <div className="chart-panel">
              <Plot
                data={[
                  {
                    type: "scatter",
                    mode: "lines",
                    x: convexity.x,
                    y: convexity.c,
                    line: { color: "#F4A261", width: 2 },
                    name: "Convexity"
                  }
                ]}
                layout={{
                  height: 240,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 20, r: 12, t: 6, b: 36 },
                  xaxis: {
                    title: "Spot",
                    tickfont: { color: "#b9c4d6", size: 10 },
                    titlefont: { color: "#b9c4d6", size: 11 },
                    range: payoffXRange || undefined
                  },
                  yaxis: { title: "Convexity", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                  legend: { orientation: "h", x: 0, y: -0.2, font: { color: "#b9c4d6", size: 10 } }
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            </div>
          </div>
        ) : null}
      </SectionCard>

      <SectionCard title="Scenario Distribution">
        {details.error ? <ErrorState message={String(details.error)} /> : null}
        {detail ? (
          <div className="chart-panel">
            <Plot
              data={[
                {
                  type: "bar",
                  x: detail.scenario_dist.map((r) => r.scenario),
                  y: detail.scenario_dist.map((r) => r.pnl_inr),
                  marker: { color: detail.scenario_dist.map((r) => (r.pnl_inr < 0 ? "#E57373" : "#58D68D")) }
                },
                {
                  type: "scatter",
                  mode: "lines+markers",
                  x: detail.scenario_dist.map((r) => r.scenario),
                  y: detail.scenario_dist.map((r) => r.cum_prob),
                  yaxis: "y2",
                  line: { color: "#F5B041", width: 2 }
                }
              ]}
              layout={{
                height: 340,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 20, r: 6, t: 6, b: 40 },
                xaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 }, tickangle: -25, automargin: true },
                yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                yaxis2: {
                  overlaying: "y",
                  side: "right",
                  title: "Cum Prob %",
                  tickfont: { color: "#b9c4d6", size: 10 },
                  titlefont: { color: "#b9c4d6", size: 11 }
                }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>
        ) : null}
      </SectionCard>

      <SectionCard title="Trade Level Risk">
        {loading || !data ? <LoadingState /> : (
          <DataTable
            columns={[
              { key: "trade_id", label: "Trade" },
              { key: "bucket", label: "Bucket" },
              { key: "trade_es99_inr", label: "ES99 (₹)", format: (v) => formatInr(v) },
              { key: "trade_es99_pct", label: "ES99 (%)", format: (v) => formatPct(v) },
              { key: "tail_loss_inr", label: "Tail Loss", format: (v) => formatInr(v), tone: "negative" },
              { key: "expected_pnl_inr", label: "Mean PnL", format: (v) => formatInr(v) },
              { key: "mean_loss_inr", label: "Mean Loss", format: (v) => formatInr(v) },
              { key: "risk_reward", label: "RR", format: (v) => formatNumber(v, 2) },
              { key: "theta_carry_inr", label: "Theta Carry", format: (v) => formatInr(v) },
              { key: "premium_received_inr", label: "Premium", format: (v) => formatInr(v) },
              { key: "theta_per_lakh", label: "Theta/1L", format: (v) => formatNumber(v, 1) },
              { key: "gamma_per_lakh", label: "Gamma/1L", format: (v) => formatNumber(v, 4) },
              { key: "vega_per_lakh", label: "Vega/1L", format: (v) => formatNumber(v, 1) },
              { key: "zone_label", label: "Zone" },
              { key: "mtd_pnl", label: "MTD P&L", format: (v) => formatInr(v) }
            ]}
            rows={filteredRows}
          />
        )}
      </SectionCard>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">Debug Tables</div>
        </summary>
        <div className="section-body">
          {detail ? (
            <>
              <DataTable
                columns={[
                  { key: "symbol", label: "Symbol" },
                  { key: "qty", label: "Qty" },
                  { key: "strike", label: "Strike" },
                  { key: "type", label: "Type" },
                  { key: "price", label: "Price" },
                  { key: "expiry", label: "Expiry" },
                  { key: "dte", label: "DTE" },
                  { key: "tte", label: "TTE" },
                  { key: "iv", label: "IV" }
                ]}
                rows={detail.iv_debug || []}
              />
              <div style={{ marginTop: "12px" }}>
                <DataTable
                  columns={[
                    { key: "Scenario", label: "Scenario" },
                    { key: "Bucket", label: "Bucket" },
                    { key: "Repriced P&L (₹)", label: "Repriced P&L" },
                    { key: "Loss % NAV", label: "Loss % NAV" },
                    { key: "Threshold % NAV", label: "Threshold % NAV" },
                    { key: "Probability", label: "Probability" }
                  ]}
                  rows={detail.scenario_rows || []}
                />
              </div>
            </>
          ) : null}
        </div>
      </details>
    </>
  );
}
