import { useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import Plot from "../components/Plot";
import { formatInr, formatNumber } from "../components/format";
import { useCachedApi } from "../hooks/useCachedApi";

type Ohlcv = { rows: Record<string, unknown>[]; summary: Record<string, unknown> };

type Futures = { rows: Record<string, unknown>[] };

type Options = { ce: Record<string, unknown>[]; pe: Record<string, unknown>[] };

export default function DerivativesData() {
  const ohlcv = useCachedApi<Ohlcv>("nifty_ohlcv", "/derivatives/nifty-ohlcv?limit=400", 60_000);
  const futures = useCachedApi<Futures>("nifty_futures", "/derivatives/futures?limit=600", 60_000);
  const options = useCachedApi<Options>("nifty_options", "/derivatives/options?limit=600", 60_000);
  const [selectedDate, setSelectedDate] = useState<string>("");

  const futuresSeries = useMemo(() => {
    const rows = futures.data?.rows || [];
    const byExpiry: Record<string, { x: string[]; oi: number[]; change: number[] }> = {};
    rows.forEach((r) => {
      const expiry = String(r.expiry_date || r.expiry || "");
      if (!expiry) return;
      if (!byExpiry[expiry]) byExpiry[expiry] = { x: [], oi: [], change: [] };
      byExpiry[expiry].x.push(String(r.date || ""));
      byExpiry[expiry].oi.push(Number(r.open_interest || 0));
      byExpiry[expiry].change.push(Number(r.change_in_oi || 0));
    });
    return byExpiry;
  }, [futures]);

  const closeMap = useMemo(() => {
    const rows = ohlcv.data?.rows || [];
    const map: Record<string, number> = {};
    rows.forEach((r) => {
      const key = String(r.date || "");
      if (!key) return;
      map[key] = Number(r.close || 0);
    });
    return map;
  }, [ohlcv.data]);

  const optionsSeries = useMemo(() => {
    const build = (rows: Record<string, unknown>[]) => {
      const byExpiry: Record<string, { x: string[]; oi: number[] }> = {};
      rows.forEach((r) => {
        const expiry = String(r.expiry_date || r.expiry || "");
        if (!expiry) return;
        if (!byExpiry[expiry]) byExpiry[expiry] = { x: [], oi: [] };
        byExpiry[expiry].x.push(String(r.date || ""));
        byExpiry[expiry].oi.push(Number(r.open_interest || 0));
      });
      return byExpiry;
    };
    return {
      ce: build(options.data?.ce || []),
      pe: build(options.data?.pe || [])
    };
  }, [options.data]);

  const perStrike = useMemo(() => {
    const ce = options.data?.ce || [];
    const pe = options.data?.pe || [];
    const all = [...ce, ...pe];
    const dates = Array.from(new Set(all.map((r) => String(r.date || "")).filter(Boolean))).sort();
    const date = selectedDate || dates[dates.length - 1] || "";
    const filtered = all.filter((r) => String(r.date || "") === date);
    const byStrike: Record<string, { strike: number; ce: number; pe: number }> = {};
    filtered.forEach((r) => {
      const strike = Number(r.strike_price || r.strike || 0);
      if (!strike) return;
      const optType = String(r.option_type || "").toUpperCase();
      if (!byStrike[strike]) byStrike[strike] = { strike, ce: 0, pe: 0 };
      const oi = Number(r.open_int || r.open_interest || 0);
      if (optType === "CE") byStrike[strike].ce += oi;
      if (optType === "PE") byStrike[strike].pe += oi;
    });
    const rows = Object.values(byStrike)
      .filter((r) => r.strike >= 23000 && r.strike <= 27000)
      .sort((a, b) => a.strike - b.strike);
    return { date, dates, rows };
  }, [options.data, selectedDate]);

  return (
    <>
      {ohlcv.error || futures.error || options.error ? <ErrorState message={String(ohlcv.error || futures.error || options.error)} /> : null}

      <SectionCard title="NIFTY OHLCV Summary">
        {ohlcv.loading || !ohlcv.data ? (
          <LoadingState />
        ) : (
          <MetricGrid>
            <MetricCard label="Latest Close" value={formatInr(ohlcv.data.summary.latest_close)} />
            <MetricCard label="2Y High" value={formatInr(ohlcv.data.summary.high_2y)} />
            <MetricCard label="2Y Low" value={formatInr(ohlcv.data.summary.low_2y)} />
            <MetricCard label="Avg Volume" value={formatNumber(ohlcv.data.summary.avg_volume)} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Futures OI Trends">
        {futures.loading || !futures.data ? <LoadingState /> : (
          <Plot
            data={[
              ...Object.entries(futuresSeries).map(([expiry, s]) => ({
                type: "scatter",
                mode: "lines",
                name: expiry,
                x: s.x,
                y: s.oi
              })),
              {
                type: "scatter",
                mode: "lines",
                name: "NIFTY Close",
                x: Object.values(futuresSeries)[0]?.x || [],
                y: (Object.values(futuresSeries)[0]?.x || []).map((d) => closeMap[String(d)] || null),
                yaxis: "y2",
                line: { color: "#E8EFE8" }
              }
            ]}
            layout={{
              height: 320,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 40, r: 40, t: 20, b: 40 },
              yaxis2: { overlaying: "y", side: "right", title: "NIFTY Close" }
            }}
            config={{ displayModeBar: false, responsive: true }}
          />
        )}
      </SectionCard>

      <SectionCard title="Futures Change in OI">
        {futures.loading || !futures.data ? <LoadingState /> : (
          <Plot
            data={Object.entries(futuresSeries).map(([expiry, s]) => ({
              type: "scatter",
              mode: "lines",
              name: expiry,
              x: s.x,
              y: s.change
            }))}
            layout={{ height: 280, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 40, r: 20, t: 20, b: 40 } }}
            config={{ displayModeBar: false, responsive: true }}
          />
        )}
      </SectionCard>

      <SectionCard title="Options OI Trends (CE)">
        {options.loading || !options.data ? <LoadingState /> : (
          <Plot
            data={[
              ...Object.entries(optionsSeries.ce).map(([expiry, s]) => ({
                type: "scatter",
                mode: "lines",
                name: expiry,
                x: s.x,
                y: s.oi
              })),
              {
                type: "scatter",
                mode: "lines",
                name: "NIFTY Close",
                x: Object.values(optionsSeries.ce)[0]?.x || [],
                y: (Object.values(optionsSeries.ce)[0]?.x || []).map((d) => closeMap[String(d)] || null),
                yaxis: "y2",
                line: { color: "#E8EFE8" }
              }
            ]}
            layout={{
              height: 280,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 40, r: 40, t: 20, b: 40 },
              yaxis2: { overlaying: "y", side: "right", title: "NIFTY Close" }
            }}
            config={{ displayModeBar: false, responsive: true }}
          />
        )}
      </SectionCard>

      <SectionCard title="Options OI Trends (PE)">
        {options.loading || !options.data ? <LoadingState /> : (
          <Plot
            data={[
              ...Object.entries(optionsSeries.pe).map(([expiry, s]) => ({
                type: "scatter",
                mode: "lines",
                name: expiry,
                x: s.x,
                y: s.oi
              })),
              {
                type: "scatter",
                mode: "lines",
                name: "NIFTY Close",
                x: Object.values(optionsSeries.pe)[0]?.x || [],
                y: (Object.values(optionsSeries.pe)[0]?.x || []).map((d) => closeMap[String(d)] || null),
                yaxis: "y2",
                line: { color: "#E8EFE8" }
              }
            ]}
            layout={{
              height: 280,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 40, r: 40, t: 20, b: 40 },
              yaxis2: { overlaying: "y", side: "right", title: "NIFTY Close" }
            }}
            config={{ displayModeBar: false, responsive: true }}
          />
        )}
      </SectionCard>

      <SectionCard title="Per-Strike OI (Selected Date)">
        {options.loading || !options.data ? <LoadingState /> : (
          <>
            <div style={{ marginBottom: "12px" }}>
              <label style={{ marginRight: "8px" }}>Date</label>
              <select
                value={perStrike.date}
                onChange={(e) => setSelectedDate(e.target.value)}
                style={{ background: "var(--gs-surface)", color: "var(--gs-text)", border: "1px solid var(--gs-border)", padding: "6px 10px", borderRadius: "6px" }}
              >
                {perStrike.dates.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </div>
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
                margin: { l: 40, r: 20, t: 20, b: 40 }
              }}
              config={{ displayModeBar: false, responsive: true }}
            />
          </>
        )}
      </SectionCard>

      <SectionCard title="NIFTY OHLCV Table">
        {ohlcv.loading || !ohlcv.data ? (
          <LoadingState />
        ) : (
          <DataTable
            columns={[
              { key: "date", label: "Date" },
              { key: "open", label: "Open" },
              { key: "high", label: "High" },
              { key: "low", label: "Low" },
              { key: "close", label: "Close" },
              { key: "volume", label: "Volume" }
            ]}
            rows={ohlcv.data.rows}
          />
        )}
      </SectionCard>
    </>
  );
}
