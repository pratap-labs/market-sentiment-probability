import { useMemo, useState } from "react";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import Plot from "../components/Plot";
import SectionCard from "../components/SectionCard";
import { useCachedApi } from "../hooks/useCachedApi";

type WorldIndexRow = {
  country: string;
  symbol: string;
  name: string;
  last_date?: string;
  close?: number;
  prev_close?: number;
  day_change_pct?: number;
  ret_5d_pct?: number;
  ret_1m_pct?: number;
  ret_3m_pct?: number;
  points_count?: number;
  error?: string;
};

type WorldIndexesResponse = {
  generated_at: string;
  source: string;
  rows: WorldIndexRow[];
  nifty_relationship?: {
    error?: string;
    samples?: number;
    date_start?: string;
    date_end?: string;
    target?: string;
    feature_count?: number;
    correlation_by_symbol?: Array<{ symbol: string; corr_with_nifty: number | null }>;
    model?: {
      method?: string;
      intercept?: number;
      r2?: number;
      betas?: Array<{ symbol: string; beta: number }>;
      error?: string;
    };
  };
  history_by_symbol: Record<string, Array<{
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
  }>>;
  db: {
    snapshot_path: string;
    history_path: string;
    symbols_count: number;
  };
};

function fmtNum(v: number | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return Number(v).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

function fmtPct(v: number | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${Number(v).toFixed(digits)}%`;
}

function fmtFloat(v: number | undefined, digits = 4): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return Number(v).toFixed(digits);
}

export default function SpotAnalysisWorldIndexes() {
  const [refreshToken, setRefreshToken] = useState<number>(Date.now());
  const data = useCachedApi<WorldIndexesResponse>(
    `spot_world_indexes_${refreshToken}`,
    `/spot-analysis/world-indexes?t=${refreshToken}`,
    0
  );

  const payload = data.data;
  const rows = useMemo(() => data.data?.rows || [], [data.data]);
  const okRows = useMemo(() => rows.filter((r) => !r.error), [rows]);
  const chartRows = useMemo(
    () =>
      okRows
        .slice()
        .sort((a, b) => `${a.country}-${a.name}`.localeCompare(`${b.country}-${b.name}`)),
    [okRows]
  );
  const byCountry = useMemo(() => {
    const out: Record<string, WorldIndexRow[]> = {};
    rows.forEach((r) => {
      const key = String(r.country || "Other");
      if (!out[key]) out[key] = [];
      out[key].push(r);
    });
    Object.keys(out).forEach((k) => {
      out[k] = out[k].slice().sort((a, b) => String(a.name || "").localeCompare(String(b.name || "")));
    });
    return out;
  }, [rows]);

  return (
    <SectionCard title="World Indexes">
      {data.loading ? (
        <LoadingState />
      ) : data.error || !payload ? (
        <ErrorState message={String(data.error || "World indexes unavailable")} />
      ) : (
        <>
          <div className="controls-row" style={{ marginBottom: 12, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <button className="control-input" onClick={() => setRefreshToken(Date.now())}>Refresh from Yahoo</button>
            <span style={{ color: "#9fb0c9", fontSize: 12 }}>
              Generated: {payload.generated_at} | Source: {payload.source}
            </span>
          </div>

          <div style={{ color: "#9fb0c9", fontSize: 12, marginBottom: 12 }}>
            DB: {payload.db.history_path}
          </div>

          <div className="table-wrap" style={{ marginTop: 8 }}>
            <div style={{ color: "#d7dfec", fontWeight: 700, marginBottom: 8 }}>
              NIFTY as Output | World Indexes as Inputs
            </div>
            {payload.nifty_relationship?.error ? (
              <div style={{ color: "#d45f5f", fontSize: 12 }}>{payload.nifty_relationship.error}</div>
            ) : (
              <>
                <div style={{ color: "#9fb0c9", fontSize: 12, marginBottom: 8 }}>
                  Samples: {payload.nifty_relationship?.samples ?? "—"} | Range: {payload.nifty_relationship?.date_start || "—"} to {payload.nifty_relationship?.date_end || "—"} | R²: {fmtFloat(payload.nifty_relationship?.model?.r2, 4)}
                </div>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Corr with NIFTY Return</th>
                      <th>OLS Beta (NIFTY target)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(payload.nifty_relationship?.correlation_by_symbol || []).map((row) => {
                      const beta = (payload.nifty_relationship?.model?.betas || []).find((b) => b.symbol === row.symbol);
                      return (
                        <tr key={`corr-${row.symbol}`}>
                          <td>{row.symbol}</td>
                          <td>{fmtFloat(row.corr_with_nifty == null ? undefined : row.corr_with_nifty, 4)}</td>
                          <td>{fmtFloat(beta?.beta, 4)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </>
            )}
          </div>

          {Object.keys(byCountry).sort().map((country) => (
            <div key={country} className="table-wrap" style={{ marginTop: 14 }}>
              <div style={{ color: "#d7dfec", fontWeight: 700, marginBottom: 8 }}>{country}</div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Index</th>
                    <th>Symbol</th>
                    <th>Last Date</th>
                    <th>Close</th>
                    <th>Day %</th>
                    <th>5D %</th>
                    <th>1M %</th>
                    <th>3M %</th>
                    <th>Points</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {byCountry[country].map((r) => (
                    <tr key={`${country}-${r.symbol}`}>
                      <td>{r.name}</td>
                      <td>{r.symbol}</td>
                      <td>{r.last_date || "—"}</td>
                      <td>{fmtNum(r.close, 2)}</td>
                      <td>{fmtPct(r.day_change_pct, 2)}</td>
                      <td>{fmtPct(r.ret_5d_pct, 2)}</td>
                      <td>{fmtPct(r.ret_1m_pct, 2)}</td>
                      <td>{fmtPct(r.ret_3m_pct, 2)}</td>
                      <td>{r.points_count ?? "—"}</td>
                      <td>{r.error ? `Error: ${r.error}` : "OK"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}

          <div style={{ marginTop: 18, color: "#d7dfec", fontWeight: 700 }}>Candles (All Indexes)</div>
          <div
            style={{
              marginTop: 10,
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: 12,
            }}
          >
            {chartRows.map((r) => {
              const hist = payload.history_by_symbol?.[r.symbol] || [];
              const x = hist.map((h) => h.date);
              const open = hist.map((h) => Number(h.open));
              const high = hist.map((h) => Number(h.high));
              const low = hist.map((h) => Number(h.low));
              const close = hist.map((h) => Number(h.close));
              return (
                <div key={`chart-${r.symbol}`} className="chart-panel">
                  <div style={{ color: "#d7dfec", fontWeight: 600, marginBottom: 6 }}>
                    {r.country} | {r.name} ({r.symbol})
                  </div>
                  <Plot
                    data={[
                      {
                        type: "candlestick",
                        x,
                        open,
                        high,
                        low,
                        close,
                        increasing: { line: { color: "#30b878" } },
                        decreasing: { line: { color: "#d45f5f" } },
                        whiskerwidth: 0.3,
                      } as any,
                    ]}
                    layout={{
                      height: 320,
                      margin: { l: 46, r: 18, t: 12, b: 42 },
                      paper_bgcolor: "rgba(0,0,0,0)",
                      plot_bgcolor: "rgba(0,0,0,0)",
                      xaxis: { type: "date", rangeslider: { visible: false }, tickfont: { color: "#9fb0c9", size: 10 } },
                      yaxis: { tickfont: { color: "#9fb0c9", size: 10 } },
                      showlegend: false,
                    } as any}
                    config={{ displayModeBar: false, responsive: true }}
                    style={{ width: "100%" }}
                  />
                </div>
              );
            })}
          </div>
        </>
      )}
    </SectionCard>
  );
}
