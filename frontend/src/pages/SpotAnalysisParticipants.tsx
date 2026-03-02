import { useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import { useCachedApi } from "../hooks/useCachedApi";
import Plot from "../components/Plot";

type Futures = { rows: Record<string, unknown>[] };
type Options = { ce: Record<string, unknown>[]; pe: Record<string, unknown>[] };

const EXPIRY_PALETTE = [
  "#2D7DFF",
  "#FF9F1C",
  "#2EC4B6",
  "#E96767",
  "#9B5DE5",
  "#75F37B",
  "#F15BB5",
  "#00BBF9"
];

type ParticipantMetrics = {
  futures_index_net_oi: number;
  futures_index_net_volume: number;
  futures_index_oi_change: number | null;
  option_index_call_net_oi: number;
  option_index_put_net_oi: number;
  option_index_call_oi_change: number | null;
  option_index_put_oi_change: number | null;
  option_index_call_net_volume: number;
  option_index_put_net_volume: number;
  signal: "BULLISH" | "BEARISH";
  total_net_oi: number;
};

type SpotAnalysisRow = {
  date: string;
  nifty_close: number | null;
  nifty_change_pct: number | null;
  fii_buy_sell_amt_cr: number | null;
  fii_cash_cr: number | null;
  dii_cash_cr: number | null;
  participants: {
    fii: ParticipantMetrics;
    dii: ParticipantMetrics;
    pro: ParticipantMetrics;
    client: ParticipantMetrics;
  };
};

type SpotParticipantsResponse = {
  rows: SpotAnalysisRow[];
};

function formatSigned(value: unknown): string {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return num.toLocaleString("en-IN", {
    maximumFractionDigits: 0,
    minimumFractionDigits: 0,
  });
}

function formatNifty(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toLocaleString("en-IN", { maximumFractionDigits: 0 });
}

function formatPct(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function formatCr(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const sign = value < 0 ? "-" : "";
  return `${sign}${Math.abs(value).toLocaleString("en-IN", { maximumFractionDigits: 0 })} Cr`;
}

function formatLakhs(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const lakh = value / 100000;
  const sign = lakh < 0 ? "-" : "";
  return `${sign}${Math.abs(lakh).toFixed(2)}L`;
}

function Pill({ label }: { label: string }) {
  const bearish = label.toUpperCase() === "BEARISH";
  return <span className={`participant-pill ${bearish ? "bearish" : "bullish"}`}>{label}</span>;
}

export default function SpotAnalysisParticipants() {
  const [selectedParticipant, setSelectedParticipant] = useState<"fii" | "dii" | "pro" | "client">("fii");
  const futures = useCachedApi<Futures>("spot_tab_nifty_futures", "/derivatives/futures?limit=6000", 60_000);
  const options = useCachedApi<Options>("spot_tab_nifty_options", "/derivatives/options?limit=25000", 60_000);
  const data = useCachedApi<SpotParticipantsResponse>(
    "spot_participants",
    "/spot-analysis/participants?limit=0",
    60_000
  );
  const rowsAsc = useMemo(() => {
    const rows = data.data?.rows || [];
    return [...rows].sort((a, b) => a.date.localeCompare(b.date));
  }, [data.data]);

  const futuresSeries = useMemo(() => {
    const rows = futures.data?.rows || [];
    const byExpiry: Record<string, { x: string[]; oi: number[] }> = {};
    rows.forEach((r) => {
      const expiry = String(r.expiry_date || r.expiry || "");
      if (!expiry) return;
      if (expiry.startsWith("2025-10")) return;
      if (!byExpiry[expiry]) byExpiry[expiry] = { x: [], oi: [] };
      byExpiry[expiry].x.push(String(r.date || ""));
      byExpiry[expiry].oi.push(Number(r.open_interest || 0));
    });
    return byExpiry;
  }, [futures.data]);

  const expiryColorMap = useMemo(() => {
    const set = new Set<string>();
    Object.keys(futuresSeries).forEach((e) => set.add(e));
    (options.data?.ce || []).forEach((r) => {
      const e = String(r.expiry_date || r.expiry || "");
      if (e) set.add(e);
    });
    (options.data?.pe || []).forEach((r) => {
      const e = String(r.expiry_date || r.expiry || "");
      if (e) set.add(e);
    });
    const all = Array.from(set).sort();
    const out: Record<string, string> = {};
    all.forEach((e, i) => {
      out[e] = EXPIRY_PALETTE[i % EXPIRY_PALETTE.length];
    });
    return out;
  }, [futuresSeries, options.data]);

  const optionsOiTrend = useMemo(() => {
    const ce = options.data?.ce || [];
    const pe = options.data?.pe || [];
    const rows: Array<Record<string, unknown> & { __side: "CE" | "PE" }> = [
      ...ce.map((r) => ({ ...r, __side: "CE" as const })),
      ...pe.map((r) => ({ ...r, __side: "PE" as const }))
    ];
    if (!rows.length) return { traces: [] as any[] };

    const allDates = Array.from(
      new Set(rows.map((r) => String(r["date"] || "")).filter(Boolean))
    ).sort();
    const dateIndex = new Map<string, number>(allDates.map((d, i) => [d, i]));
    const byExpiry: Record<string, { CE: (number | null)[]; PE: (number | null)[] }> = {};

    rows.forEach((r) => {
      const expiry = String(r["expiry_date"] || r["expiry"] || "");
      const date = String(r["date"] || "");
      if (!expiry || !date || !dateIndex.has(date)) return;
      if (!byExpiry[expiry]) {
        byExpiry[expiry] = { CE: Array(allDates.length).fill(null), PE: Array(allDates.length).fill(null) };
      }
      const side = r.__side;
      const idx = dateIndex.get(date)!;
      const oi = Number(r["open_int"] || r["open_interest"] || 0);
      const prev = byExpiry[expiry][side][idx];
      byExpiry[expiry][side][idx] = (prev ?? 0) + oi;
    });

    const traces: any[] = [];
    Object.keys(byExpiry).sort().forEach((expiry) => {
      const color = expiryColorMap[expiry] || "#2D7DFF";
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `${expiry} CE`,
        x: allDates,
        y: byExpiry[expiry].CE,
        line: { color, width: 2 }
      });
      traces.push({
        type: "scatter",
        mode: "lines",
        name: `${expiry} PE`,
        x: allDates,
        y: byExpiry[expiry].PE,
        line: { color, width: 1.6, dash: "dot" }
      });
    });
    return { traces };
  }, [options.data, expiryColorMap]);

  const sharedOiXRange = useMemo(() => {
    const fixedStart = "2025-10-01";
    const futuresDates: string[] = [];
    Object.values(futuresSeries).forEach((s) => {
      s.x.forEach((d) => {
        if (d) futuresDates.push(String(d));
      });
    });
    const optionsDates = [
      ...(options.data?.ce || []).map((r) => String(r.date || "")),
      ...(options.data?.pe || []).map((r) => String(r.date || "")),
    ].filter(Boolean);
    const all = [...futuresDates, ...optionsDates].map((d) => String(d)).filter(Boolean).sort();
    if (!all.length) return [fixedStart, fixedStart];
    return [fixedStart, all[all.length - 1]];
  }, [futuresSeries, options.data]);

  const series = useMemo(() => {
    const x = rowsAsc.map((r) => String(r.date));
    const nifty = rowsAsc.map((r) => Number(r.nifty_close ?? NaN));
    const p = rowsAsc.map((r) => r.participants[selectedParticipant]);
    return {
      x,
      nifty,
      callVol: p.map((v) => Number(v.option_index_call_net_volume || 0)),
      putVol: p.map((v) => Number(v.option_index_put_net_volume || 0)),
      futOiChg: p.map((v) => Number(v.futures_index_oi_change || 0)),
      callOi: p.map((v) => Number(v.option_index_call_net_oi || 0)),
      putOi: p.map((v) => Number(v.option_index_put_net_oi || 0)),
      futOi: p.map((v) => Number(v.futures_index_net_oi || 0)),
    };
  }, [rowsAsc, selectedParticipant]);

  const pLabel = selectedParticipant.toUpperCase();
  const chartLayout = {
    height: 330,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 52, r: 52, t: 24, b: 48 },
    xaxis: {
      type: "date" as const,
      tickformat: "%d %b",
      tickfont: { color: "#9fb0c9", size: 10 }
    },
    yaxis: { zeroline: true, zerolinecolor: "rgba(156,173,196,0.3)", tickfont: { color: "#9fb0c9", size: 10 } },
    yaxis2: {
      overlaying: "y" as const,
      side: "right" as const,
      showgrid: false,
      title: "NIFTY",
      tickfont: { color: "#c264ff", size: 10 },
      titlefont: { color: "#c264ff", size: 10 },
    },
    legend: { orientation: "h" as const, y: 1.12, x: 0, font: { color: "#c7d3e6", size: 10 } },
  };

  return (
    <>
      <SectionCard title="FII Flow History">
        {data.error ? <ErrorState message={String(data.error)} /> : null}
        {data.loading || !data.data ? (
          <LoadingState />
        ) : (
          <div className="table-wrap participant-table-wrap" style={{ maxHeight: 720, overflow: "auto" }}>
            <table className="data-table participant-table">
              <thead>
                <tr>
                  <th rowSpan={3}>Date</th>
                  <th rowSpan={3}>NIFTY</th>
                  <th colSpan={2} className="participant-group-sep-right">Options</th>
                  <th colSpan={4} className="participant-group-sep-right">Futures</th>
                  <th colSpan={2}>Cash</th>
                </tr>
                <tr>
                  <th colSpan={2} className="participant-group-sep-right"></th>
                  <th colSpan={4} className="participant-group-sep-right">FII Index Futures</th>
                  <th colSpan={2}></th>
                </tr>
                <tr>
                  <th>FII Call OI Chg</th>
                  <th className="participant-group-sep-right">FII Put OI Chg</th>
                  <th>Buy/Sell(Amt)</th>
                  <th>OI Change (Qty)</th>
                  <th>View</th>
                  <th className="participant-group-sep-right">OI</th>
                  <th>FII Cash</th>
                  <th>DII Cash</th>
                </tr>
              </thead>
              <tbody>
                {data.data.rows.map((row) => (
                  <tr key={row.date}>
                    <td>{row.date}</td>
                    <td>
                      <span>{formatNifty(row.nifty_close)}</span>
                      <span
                        className={
                          row.nifty_change_pct == null
                            ? "participant-change"
                            : row.nifty_change_pct >= 0
                              ? "participant-change positive"
                              : "participant-change negative"
                        }
                    >
                      {formatPct(row.nifty_change_pct)}
                    </span>
                  </td>

                    <td>{formatSigned(row.participants.fii.option_index_call_oi_change)}</td>
                    <td>{formatSigned(row.participants.fii.option_index_put_oi_change)}</td>
                    <td>{formatCr(row.fii_buy_sell_amt_cr)}</td>
                    <td>{formatSigned(row.participants.fii.futures_index_oi_change)}</td>
                    <td><Pill label={row.participants.fii.signal} /></td>
                    <td>{formatLakhs(row.participants.fii.futures_index_net_oi)}</td>
                    <td>{formatCr(row.fii_cash_cr)}</td>
                    <td>{formatCr(row.dii_cash_cr)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </SectionCard>

      <SectionCard title="Futures OI Trend">
        {futures.loading ? <LoadingState /> : Object.keys(futuresSeries).length === 0 ? (
          <div style={{ color: "var(--gs-muted)" }}>No futures OI data available.</div>
        ) : (
          <Plot
            data={Object.entries(futuresSeries).map(([expiry, s]) => ({
              type: "scatter",
              mode: "lines",
              name: expiry,
              x: s.x,
              y: s.oi,
              line: { color: expiryColorMap[expiry] || "#2D7DFF", width: 2 }
            }))}
            layout={{
              height: 320,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 30, t: 30, b: 30 },
              xaxis: { tickfont: { color: "#b9c4d6", size: 10 }, range: sharedOiXRange },
              yaxis: { tickfont: { color: "#b9c4d6", size: 10 } },
              showlegend: false
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <SectionCard title="Options OI Trend">
        {options.loading || !options.data ? <LoadingState /> : (
          <Plot
            data={optionsOiTrend.traces}
            layout={{
              height: 300,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 30, t: 30, b: 80 },
              xaxis: { tickfont: { color: "#b9c4d6", size: 10 }, range: sharedOiXRange },
              yaxis: { tickfont: { color: "#b9c4d6", size: 10 } },
              legend: { orientation: "h", y: -0.22, x: 0, xanchor: "left", yanchor: "top" }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <SectionCard title="Participant Charts">
        <div className="control-grid" style={{ marginBottom: 12 }}>
          <label className="control-field">
            <span className="control-label">Participant</span>
            <select
              className="control-input"
              value={selectedParticipant}
              onChange={(e) => setSelectedParticipant(e.target.value as "fii" | "dii" | "pro" | "client")}
            >
              <option value="fii">FII</option>
              <option value="dii">DII</option>
              <option value="pro">PRO</option>
              <option value="client">CLIENT</option>
            </select>
          </label>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12 }}>
          <div className="chart-panel">
            <h4>Daily {pLabel} Buy/Sell in Index Options</h4>
            <Plot
              data={[
                { type: "bar", name: `${pLabel} Call Vol Chg`, x: series.x, y: series.callVol, marker: { color: "#6be779" } },
                { type: "bar", name: `${pLabel} Put Vol Chg`, x: series.x, y: series.putVol, marker: { color: "#f06a6a" } },
                { type: "scatter", mode: "lines", name: "NIFTY", x: series.x, y: series.nifty, yaxis: "y2", line: { color: "#d058ff", width: 2 } },
              ]}
              layout={{
                ...chartLayout,
                barmode: "group",
                yaxis2: {
                  overlaying: "y",
                  side: "right",
                  showgrid: false,
                  title: "NIFTY",
                  tickfont: { color: "#d058ff", size: 10 },
                  titlefont: { color: "#d058ff", size: 10 },
                },
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>

          <div className="chart-panel">
            <h4>Daily {pLabel} Buy/Sell in Index Futures</h4>
            <Plot
              data={[
                { type: "bar", name: `${pLabel} Fut OI Chg`, x: series.x, y: series.futOiChg, marker: { color: "#6f76ff" } },
                { type: "scatter", mode: "lines", name: "NIFTY", x: series.x, y: series.nifty, yaxis: "y2", line: { color: "#d058ff", width: 2 } },
              ]}
              layout={{
                ...chartLayout,
                barmode: "group",
                yaxis2: {
                  overlaying: "y",
                  side: "right",
                  showgrid: false,
                  title: "NIFTY",
                  tickfont: { color: "#d058ff", size: 10 },
                  titlefont: { color: "#d058ff", size: 10 },
                },
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>

          <div className="chart-panel">
            <h4>{pLabel} OI in Index Options</h4>
            <Plot
              data={[
                { type: "bar", name: `${pLabel} Call OI`, x: series.x, y: series.callOi, marker: { color: "#6be779" } },
                { type: "bar", name: `${pLabel} Put OI`, x: series.x, y: series.putOi, marker: { color: "#f06a6a" } },
                { type: "scatter", mode: "lines", name: "NIFTY", x: series.x, y: series.nifty, yaxis: "y2", line: { color: "#d058ff", width: 2 } },
              ]}
              layout={{ ...chartLayout, barmode: "group" }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>

          <div className="chart-panel">
            <h4>{pLabel} OI in Index Futures</h4>
            <Plot
              data={[
                { type: "bar", name: `${pLabel} Futures OI`, x: series.x, y: series.futOi, marker: { color: "#4bb0ff" } },
                { type: "scatter", mode: "lines", name: "NIFTY", x: series.x, y: series.nifty, yaxis: "y2", line: { color: "#d058ff", width: 2 } },
              ]}
              layout={{ ...chartLayout, barmode: "group" }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </div>
        </div>
      </SectionCard>
    </>
  );
}
