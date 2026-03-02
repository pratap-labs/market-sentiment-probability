import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { useControls } from "../state/ControlsContext";

type History = { drift_counts: { drift: string; count: number }[]; ohlc: Record<string, unknown>[] };
type Ohlcv = { rows: Record<string, unknown>[]; summary: Record<string, unknown> };
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

export default function RiskBucketsSpotAnalysis() {
  const history = useCachedApi<History>(
    "risk_buckets_history",
    "/risk-buckets/history",
    60_000
  );
  const ohlcv = useCachedApi<Ohlcv>("nifty_ohlcv", "/derivatives/nifty-ohlcv?limit=400", 60_000);
  const futures = useCachedApi<Futures>("nifty_futures", "/derivatives/futures?limit=600", 60_000);
  const options = useCachedApi<Options>("nifty_options", "/derivatives/options?limit=25000", 60_000);
  const { setControls } = useControls();
  const [selectedExpiry, setSelectedExpiry] = useState<string>("All");

  const ohlc = useMemo(() => {
    const rows = history.data?.ohlc || ohlcv.data?.rows || [];
    return {
      x: rows.map((r) => String(r.date || "")),
      open: rows.map((r) => Number(r.open || 0)),
      high: rows.map((r) => Number(r.high || 0)),
      low: rows.map((r) => Number(r.low || 0)),
      close: rows.map((r) => Number(r.close || 0))
    };
  }, [history.data, ohlcv.data]);

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
    const expirySet = new Set<string>();
    Object.keys(futuresSeries).forEach((e) => expirySet.add(e));
    (options.data?.ce || []).forEach((r) => {
      const e = String(r.expiry_date || r.expiry || "");
      if (e) expirySet.add(e);
    });
    (options.data?.pe || []).forEach((r) => {
      const e = String(r.expiry_date || r.expiry || "");
      if (e) expirySet.add(e);
    });
    const all = Array.from(expirySet).sort();
    const out: Record<string, string> = {};
    all.forEach((e, i) => {
      out[e] = EXPIRY_PALETTE[i % EXPIRY_PALETTE.length];
    });
    return out;
  }, [futuresSeries, options.data]);

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
    const all = [...futuresDates, ...optionsDates]
      .map((d) => String(d))
      .filter(Boolean)
      .sort();
    if (!all.length) return [fixedStart, fixedStart];
    return [fixedStart, all[all.length - 1]];
  }, [futuresSeries, options.data]);

  const expiries = useMemo(() => {
    const ce = options.data?.ce || [];
    const pe = options.data?.pe || [];
    const all = [...ce, ...pe];
    const list = Array.from(new Set(all.map((r) => String(r.expiry_date || r.expiry || "")).filter(Boolean))).sort();
    return list;
  }, [options.data]);

  useEffect(() => {
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Options Expiry</span>
          <select
            className="control-input"
            value={selectedExpiry}
            onChange={(e) => setSelectedExpiry(e.target.value)}
          >
            <option value="All">All</option>
            {expiries.map((e) => (
              <option key={e} value={e}>{e}</option>
            ))}
          </select>
        </label>
      </div>
    );
    setControls({
      title: "Controls",
      summary: (
        <>
          <span>
            <span className="controls-summary-key">Expiry</span> {selectedExpiry}
          </span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, selectedExpiry, expiries]);

  const optionsOiTrend = useMemo(() => {
    const ce = options.data?.ce || [];
    const pe = options.data?.pe || [];
    const rows: Array<Record<string, unknown> & { __side: "CE" | "PE" }> = [
      ...ce.map((r) => ({ ...r, __side: "CE" as const })),
      ...pe.map((r) => ({ ...r, __side: "PE" as const }))
    ];
    if (!rows.length) return { dates: [] as string[], traces: [] as any[] };

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
        byExpiry[expiry] = {
          CE: Array(allDates.length).fill(null),
          PE: Array(allDates.length).fill(null)
        };
      }
      const side = r.__side;
      const idx = dateIndex.get(date)!;
      const oi = Number(r["open_int"] || r["open_interest"] || 0);
      const prev = byExpiry[expiry][side][idx];
      byExpiry[expiry][side][idx] = (prev ?? 0) + oi;
    });

    const expiryList = Object.keys(byExpiry).sort();
    const traces: any[] = [];
    expiryList.forEach((expiry) => {
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

    return { dates: allDates, traces };
  }, [options.data, expiryColorMap]);

  const perStrike = useMemo(() => {
    const ce = options.data?.ce || [];
    const pe = options.data?.pe || [];
    const all = [...ce, ...pe];
    if (!all.length) return { expiry: "", date: "", rows: [] as { strike: number; ce: number; pe: number }[] };
    const expiry = selectedExpiry === "All" ? (expiries[expiries.length - 1] || "") : selectedExpiry;
    const filtered = all.filter((r) => String(r.expiry_date || r.expiry || "") === expiry);
    const dates = Array.from(new Set(filtered.map((r) => String(r.date || "")).filter(Boolean))).sort();
    const date = dates[dates.length - 1] || "";
    const byStrike: Record<string, { strike: number; ce: number; pe: number }> = {};
    filtered.filter((r) => String(r.date || "") === date).forEach((r) => {
      const strike = Number(r.strike_price || r.strike || 0);
      if (!strike) return;
      const optType = String(r.option_type || "").toUpperCase();
      if (!byStrike[strike]) byStrike[strike] = { strike, ce: 0, pe: 0 };
      const oi = Number(r.open_int || r.open_interest || 0);
      if (optType === "CE") byStrike[strike].ce += oi;
      if (optType === "PE") byStrike[strike].pe += oi;
    });
    const rows = Object.values(byStrike).sort((a, b) => a.strike - b.strike);
    return { expiry, date, rows };
  }, [options.data, selectedExpiry, expiries]);

  return (
    <>
      {history.error || ohlcv.error || futures.error || options.error ? (
        <ErrorState message={String(history.error || ohlcv.error || futures.error || options.error)} />
      ) : null}

      <SectionCard title="NIFTY OHLCV">
        {history.loading || ohlcv.loading ? (
          <LoadingState />
        ) : (
          <Plot
            data={[
              {
                type: "candlestick",
                x: ohlc.x,
                open: ohlc.open,
                high: ohlc.high,
                low: ohlc.low,
                close: ohlc.close,
                increasing: { line: { color: "#2ecc71" } },
                decreasing: { line: { color: "#ff4d4d" } }
              }
            ]}
            layout={{
              height: 280,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 30, t: 30, b: 30 },
              xaxis: { tickfont: { color: "#b9c4d6", size: 10 } },
              yaxis: { tickfont: { color: "#b9c4d6", size: 10 } }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <SectionCard title="NIFTY DRIFT CHART">
        {history.loading || ohlcv.loading ? (
          <LoadingState />
        ) : (
          <Plot
            data={[
              {
                type: "bar",
                x: (history.data?.drift_counts || []).map((d) => d.drift),
                y: (history.data?.drift_counts || []).map((d) => d.count),
                marker: { color: "#2EC4B6" },
                text: (() => {
                  const counts = (history.data?.drift_counts || []).map((d) => d.count);
                  const total = counts.reduce((a, b) => a + b, 0) || 1;
                  return counts.map((c) => `${((c / total) * 100).toFixed(1)}%`);
                })(),
                textposition: "outside"
              }
            ]}
            layout={{
              height: 260,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 30, t: 30, b: 30 },
              xaxis: { tickangle: -45, tickfont: { color: "#b9c4d6", size: 10 } },
              yaxis: { tickfont: { color: "#b9c4d6", size: 10 } }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
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

      <SectionCard title="Per-Strike OI (Selected Expiry)">
        {options.loading || !options.data ? <LoadingState /> : (
          <Plot
            data={[
              {
                type: "bar",
                name: "CE OI",
                x: perStrike.rows.map((r) => r.strike),
                y: perStrike.rows.map((r) => r.ce),
                marker: { color: "#E96767" }
              },
              {
                type: "bar",
                name: "PE OI",
                x: perStrike.rows.map((r) => r.strike),
                y: perStrike.rows.map((r) => r.pe),
                marker: { color: "#75F37B" }
              }
            ]}
            layout={{
              height: 320,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              barmode: "group",
              margin: { l: 30, r: 30, t: 30, b: 30 },
              xaxis: { tickfont: { color: "#b9c4d6", size: 10 } },
              yaxis: { tickfont: { color: "#b9c4d6", size: 10 } }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>
    </>
  );
}
