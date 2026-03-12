import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { apiFetch } from "../api/client";

type ConstituentRow = {
  rank: number;
  symbol: string;
  company?: string;
  final_composite_score: number | null;
  weight: number | null;
  weight_pct: number | null;
  date: string | null;
  close: number | null;
  volume: number | null;
};

type ConstituentsResponse = {
  rows: ConstituentRow[];
  count: number;
};

type OhlcvRow = {
  symbol: string;
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
};

type OhlcvResponse = {
  rows: OhlcvRow[];
  count: number;
};

type ContributionEntry = {
  symbol: string;
  weight_pct: number | null;
  ltp: number | null;
  change_pct: number | null;
  volume: number | null;
  contribution_points: number | null;
};

type ContributionDay = {
  date: string;
  nifty_close: number | null;
  nifty_change_points: number | null;
  nifty_change_pct: number | null;
  positive_count: number;
  negative_count: number;
  positive_total: number | null;
  negative_total: number | null;
  net_total: number | null;
  constituents: ContributionEntry[];
};

type ContributionTimeseriesResponse = {
  rows: ContributionDay[];
  count: number;
  constituents: string[];
};

function fmtNum(value: unknown, digits = 2): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(digits);
}

function fmtPct(value: unknown, digits = 2): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return `${n.toFixed(digits)}%`;
}

function fmtVolume(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return n.toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

function fmtSliderDate(value: string): string {
  const dt = new Date(`${value}T00:00:00`);
  if (Number.isNaN(dt.getTime())) return value;
  return dt.toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
  });
}

export default function SpotAnalysisConstituents() {
  const contributions = useCachedApi<ContributionTimeseriesResponse>(
    "spot_constituent_contribution_ts",
    "/long-term/constituents-contribution-timeseries?top_n=20&days=180&use_cache=true",
    60_000
  );
  const constituents = useCachedApi<ConstituentsResponse>(
    "spot_constituents_top20",
    "/long-term/constituents?top_n=20&lookback_days=90&use_cache=true",
    60_000
  );
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");
  const [selectedDateIdx, setSelectedDateIdx] = useState<number>(0);
  const [symbolOhlcv, setSymbolOhlcv] = useState<OhlcvResponse | null>(null);
  const [symbolLoading, setSymbolLoading] = useState<boolean>(false);
  const [symbolError, setSymbolError] = useState<string | null>(null);

  useEffect(() => {
    const rows = constituents.data?.rows || [];
    if (!rows.length) return;
    const exists = rows.some((r) => String(r.symbol) === selectedSymbol);
    if (!selectedSymbol || !exists) {
      setSelectedSymbol(String(rows[0].symbol || ""));
    }
  }, [constituents.data, selectedSymbol]);

  useEffect(() => {
    const rows = contributions.data?.rows || [];
    if (!rows.length) return;
    setSelectedDateIdx(rows.length - 1);
  }, [contributions.data]);

  useEffect(() => {
    if (!selectedSymbol) return;
    let mounted = true;
    setSymbolLoading(true);
    setSymbolError(null);
    apiFetch<OhlcvResponse>(
      `/long-term/ohlcv?symbol=${encodeURIComponent(selectedSymbol)}&days=120&limit=500&use_cache=true`
    )
      .then((payload) => {
        if (!mounted) return;
        setSymbolOhlcv(payload);
      })
      .catch((err) => {
        if (!mounted) return;
        setSymbolError(String(err));
      })
      .finally(() => {
        if (!mounted) return;
        setSymbolLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [selectedSymbol]);

  const chartSeries = useMemo(() => {
    const rows = (symbolOhlcv?.rows || [])
      .slice()
      .sort((a, b) => String(a.date).localeCompare(String(b.date)));
    return {
      x: rows.map((r) => String(r.date)),
      open: rows.map((r) => Number(r.open ?? NaN)),
      high: rows.map((r) => Number(r.high ?? NaN)),
      low: rows.map((r) => Number(r.low ?? NaN)),
      close: rows.map((r) => Number(r.close ?? NaN)),
      volume: rows.map((r) => Number(r.volume ?? 0)),
    };
  }, [symbolOhlcv]);

  const selectedDay = useMemo(() => {
    const rows = contributions.data?.rows || [];
    if (!rows.length) return null;
    const idx = Math.max(0, Math.min(selectedDateIdx, rows.length - 1));
    return rows[idx];
  }, [contributions.data, selectedDateIdx]);

  const positiveRows = useMemo(() => {
    const rows = selectedDay?.constituents || [];
    return rows
      .filter((r) => Number(r.contribution_points || 0) > 0)
      .sort((a, b) => Number(b.contribution_points || 0) - Number(a.contribution_points || 0));
  }, [selectedDay]);

  const negativeRows = useMemo(() => {
    const rows = selectedDay?.constituents || [];
    return rows
      .filter((r) => Number(r.contribution_points || 0) < 0)
      .sort((a, b) => Number(a.contribution_points || 0) - Number(b.contribution_points || 0));
  }, [selectedDay]);

  const barPairs = useMemo(() => {
    const maxLen = Math.max(positiveRows.length, negativeRows.length);
    return Array.from({ length: maxLen }, (_, i) => ({
      pos: positiveRows[i] || null,
      neg: negativeRows[i] || null,
    }));
  }, [positiveRows, negativeRows]);

  const maxAbsContribution = useMemo(() => {
    const vals = (contributions.data?.rows || [])
      .flatMap((d) => d.constituents || [])
      .map((r) => Math.abs(Number(r.contribution_points || 0)))
      .filter((v) => Number.isFinite(v) && v > 0);
    return vals.length ? Math.max(...vals) : 1;
  }, [contributions.data]);

  return (
    <>
      {constituents.error ? <ErrorState message={String(constituents.error)} /> : null}
      {contributions.error ? <ErrorState message={String(contributions.error)} /> : null}

      <SectionCard title="NIFTY Contributors Timeline (Top 20 Constituents)">
        {contributions.loading || !contributions.data || !selectedDay ? (
          <LoadingState />
        ) : (
          <>
            <div style={{ marginBottom: 12, display: "flex", justifyContent: "center" }}>
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 10,
                  border: "1px solid rgba(90,112,146,0.45)",
                  background: "rgba(12,18,30,0.7)",
                  borderRadius: 8,
                  padding: "8px 10px",
                }}
              >
                <button
                  className="control-input"
                  type="button"
                  onClick={() => setSelectedDateIdx((prev) => Math.max(0, prev - 1))}
                  disabled={selectedDateIdx <= 0}
                  style={{ width: 30, minWidth: 30, padding: 0, height: 30 }}
                >
                  ‹
                </button>
                <div style={{ minWidth: 160, textAlign: "center", color: "#d2dced", fontWeight: 600 }}>
                  {fmtSliderDate(selectedDay.date)}
                </div>
                <button
                  className="control-input"
                  type="button"
                  onClick={() =>
                    setSelectedDateIdx((prev) =>
                      Math.min((contributions.data?.rows || []).length - 1, prev + 1)
                    )
                  }
                  disabled={selectedDateIdx >= (contributions.data?.rows || []).length - 1}
                  style={{ width: 30, minWidth: 30, padding: 0, height: 30 }}
                >
                  ›
                </button>
              </div>
            </div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 12, color: "#b9c4d6" }}>
              <span>Date: {selectedDay.date}</span>
              <span>NIFTY Change: {fmtNum(selectedDay.nifty_change_points, 2)} ({fmtPct(selectedDay.nifty_change_pct, 2)})</span>
              <span>Net Contribution: {fmtNum(selectedDay.net_total, 2)}</span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "420px 560px 420px", gap: 12, justifyContent: "space-between" }}>
              <div className="chart-panel" style={{ width: 420 }}>
                <h4 style={{ color: "#8ec7a2" }}>
                  Positive Contributors ({selectedDay.positive_count}) · {fmtNum(selectedDay.positive_total, 2)}
                </h4>
                <div className="table-wrap" style={{ maxHeight: 420, overflow: "auto" }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>LTP</th>
                        <th>Contribution</th>
                        <th>Volume</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positiveRows.map((r) => (
                        <tr key={`p-${r.symbol}`}>
                          <td>{r.symbol}</td>
                          <td>{fmtNum(r.ltp, 0)} ({fmtPct(r.change_pct, 2)})</td>
                          <td className="positive">{fmtNum(r.contribution_points, 2)}</td>
                          <td>{fmtVolume(r.volume)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="chart-panel" style={{ width: 560 }}>
                <h4>NIFTY Contributors</h4>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 420, overflow: "auto", paddingRight: 4 }}>
                  {barPairs.map((pair, idx) => {
                    const pWidth = pair.pos ? `${(Math.abs(Number(pair.pos.contribution_points || 0)) / maxAbsContribution) * 100}%` : "0%";
                    const nWidth = pair.neg ? `${(Math.abs(Number(pair.neg.contribution_points || 0)) / maxAbsContribution) * 100}%` : "0%";
                    return (
                      <div key={`barpair-${idx}`} style={{ display: "grid", gridTemplateColumns: "1fr 1px 1fr", alignItems: "center", gap: 8 }}>
                        <div style={{ display: "grid", gridTemplateColumns: "84px 130px 52px", alignItems: "center", justifyContent: "end", gap: 6 }}>
                          <span style={{ color: "#9fc2ec", textAlign: "right", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                            {pair.pos ? pair.pos.symbol : ""}
                          </span>
                          <div style={{ width: 130, height: 14, background: pair.pos ? "rgba(46,196,182,0.15)" : "transparent", position: "relative" }}>
                            <div style={{ width: pair.pos ? pWidth : "0%", height: "100%", background: "rgba(46,196,182,0.6)", marginLeft: "auto" }} />
                          </div>
                          <span className="positive" style={{ width: 52, textAlign: "right", whiteSpace: "nowrap" }}>
                            {pair.pos ? `+${fmtNum(pair.pos.contribution_points, 2)}` : ""}
                          </span>
                        </div>
                        <div style={{ width: 1, height: 22, background: "rgba(160,180,210,0.35)" }} />
                        <div style={{ display: "grid", gridTemplateColumns: "52px 130px 84px", alignItems: "center", gap: 6 }}>
                          <span className="negative" style={{ width: 52, textAlign: "left", whiteSpace: "nowrap" }}>
                            {pair.neg ? fmtNum(pair.neg.contribution_points, 2) : ""}
                          </span>
                          <div style={{ width: 130, height: 14, background: pair.neg ? "rgba(233,103,103,0.12)" : "transparent", position: "relative" }}>
                            <div style={{ width: pair.neg ? nWidth : "0%", height: "100%", background: "rgba(233,103,103,0.55)" }} />
                          </div>
                          <span style={{ color: "#9fc2ec", textAlign: "left", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                            {pair.neg ? pair.neg.symbol : ""}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              <div className="chart-panel" style={{ width: 420 }}>
                <h4 style={{ color: "#e5a0ab" }}>
                  Negative Contributors ({selectedDay.negative_count}) · {fmtNum(selectedDay.negative_total, 2)}
                </h4>
                <div className="table-wrap" style={{ maxHeight: 420, overflow: "auto" }}>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Name</th>
                        <th>LTP</th>
                        <th>Contribution</th>
                        <th>Volume</th>
                      </tr>
                    </thead>
                    <tbody>
                      {negativeRows.map((r) => (
                        <tr key={`n-${r.symbol}`}>
                          <td>{r.symbol}</td>
                          <td>{fmtNum(r.ltp, 0)} ({fmtPct(r.change_pct, 2)})</td>
                          <td className="negative">{fmtNum(r.contribution_points, 2)}</td>
                          <td>{fmtVolume(r.volume)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </>
        )}
      </SectionCard>

      <SectionCard title="Constituent Candle + Volume">
        {constituents.loading || !constituents.data ? (
          <LoadingState />
        ) : (
          <>
            <div className="control-grid" style={{ marginBottom: 12 }}>
              <label className="control-field">
                <span className="control-label">Select Stock</span>
                <select
                  className="control-input"
                  value={selectedSymbol}
                  onChange={(e) => setSelectedSymbol(e.target.value)}
                >
                  {(constituents.data.rows || []).map((row) => (
                    <option key={row.symbol} value={row.symbol}>
                      {row.symbol}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {symbolError ? (
              <ErrorState message={String(symbolError)} />
            ) : symbolLoading || !symbolOhlcv ? (
              <LoadingState />
            ) : (
              <Plot
                data={[
                  {
                    type: "candlestick",
                    x: chartSeries.x,
                    open: chartSeries.open,
                    high: chartSeries.high,
                    low: chartSeries.low,
                    close: chartSeries.close,
                    name: `${selectedSymbol} OHLC`,
                    increasing: { line: { color: "#2ecc71" } },
                    decreasing: { line: { color: "#ff4d4d" } },
                    yaxis: "y"
                  },
                  {
                    type: "bar",
                    x: chartSeries.x,
                    y: chartSeries.volume,
                    name: "Volume",
                    marker: { color: "rgba(45,125,255,0.45)" },
                    yaxis: "y2"
                  }
                ]}
                layout={{
                  height: 520,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 56, r: 28, t: 18, b: 42 },
                  xaxis: {
                    type: "date",
                    tickfont: { color: "#b9c4d6", size: 10 },
                    rangeslider: { visible: false },
                  },
                  yaxis: {
                    domain: [0.30, 1],
                    title: "Price",
                    tickfont: { color: "#b9c4d6", size: 10 },
                    titlefont: { color: "#b9c4d6", size: 10 },
                  },
                  yaxis2: {
                    domain: [0, 0.24],
                    anchor: "x",
                    showgrid: false,
                    showticklabels: false,
                    ticks: "",
                    zeroline: false,
                    showline: false,
                  },
                  legend: { orientation: "h", y: 1.08, x: 0, font: { color: "#b9c4d6", size: 10 } },
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            )}
          </>
        )}
      </SectionCard>

      <SectionCard title="Top 20 Constituents by Weight (Saved NIFTY50 Weights)">
        {constituents.loading || !constituents.data ? (
          <LoadingState />
        ) : (
          <>
            <div style={{ marginBottom: 12, color: "#b9c4d6" }}>
              Rows: {constituents.data.count ?? constituents.data.rows.length}
            </div>
            <DataTable
              columns={[
                { key: "rank", label: "Rank" },
                { key: "symbol", label: "Symbol" },
                { key: "company", label: "Company" },
                { key: "weight_pct", label: "Weight %", format: (v) => fmtPct(v, 2) },
                { key: "final_composite_score", label: "Final Score", format: (v) => fmtNum(v, 1) },
                { key: "close", label: "Close", format: (v) => fmtNum(v, 2) },
                { key: "volume", label: "Volume", format: (v) => fmtVolume(v) },
                { key: "date", label: "Date" },
              ]}
              rows={constituents.data.rows || []}
              maxHeight={620}
            />
          </>
        )}
      </SectionCard>
    </>
  );
}
