import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { formatInr } from "../components/format";
import { API_BASE_URL } from "../api/client";
import { useNotifications } from "../state/NotificationContext";
import { useControls } from "../state/ControlsContext";

type Historical = {
  warnings: string[];
  trades: Record<string, unknown>[];
  monthly: Record<string, unknown>[];
  summary: Record<string, unknown>;
  by_type: Record<string, unknown>[];
  by_expiry_type: Record<string, unknown>[];
  by_strike: Record<string, unknown>[];
  pnl_pct: number[];
  win_loss: { wins: number; losses: number };
  top_contributors: Record<string, unknown>[];
  bottom_contributors: Record<string, unknown>[];
  daily_margin: Record<string, unknown>[];
  weekly_margin_pnl: Record<string, unknown>[];
  monthly_margin_pnl: Record<string, unknown>[];
  daily_es99: Record<string, unknown>[];
};

type HistoricalRow = Record<string, unknown>;

type DailyMarginRow = HistoricalRow & {
  date: string;
  dateIso: string;
  gross_blocked?: number | string;
};

type DailyEs99Row = HistoricalRow & {
  date: string;
  dateIso: string;
  es99_inr?: number | string;
  blocked_margin?: number | string;
};

type PeriodMarginRow = HistoricalRow & {
  periodStart: string;
  periodLabel: string;
  total_pnl?: number | string;
  avg_margin_blocked?: number | string;
};

type PeriodRoiRow = PeriodMarginRow & {
  roiPct: number;
  stackTop: number;
};

const MIN_DISPLAY_DATE = "2025-06-01";

export default function HistoricalPerformance() {
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [ledgerFile, setLedgerFile] = useState<File | null>(null);
  const [foTradebookFile, setFoTradebookFile] = useState<File | null>(null);
  const [refreshKey, setRefreshKey] = useState<number>(0);
  const [marginView, setMarginView] = useState<"week" | "month">("week");
  const { notify } = useNotifications();
  const { setControls } = useControls();
  const { data, error, loading } = useCachedApi<Historical>(
    `historical_summary_${refreshKey}`,
    `/historical/summary${refreshKey ? `?t=${refreshKey}` : ""}`,
    60_000
  );

  const trades = data?.trades || [];
  const monthly = data?.monthly || [];
  const dailyMargin = data?.daily_margin || [];
  const weeklyMarginPnl = data?.weekly_margin_pnl || [];
  const monthlyMarginPnl = data?.monthly_margin_pnl || [];
  const dailyEs99 = data?.daily_es99 || [];
  const dailyMarginSeries = useMemo<DailyMarginRow[]>(() => {
    return [...dailyMargin]
      .map((row) => {
        const date = String(row.date || "");
        return {
          ...row,
          date,
          dateIso: date ? `${date}T00:00:00` : ""
        } as DailyMarginRow;
      })
      .filter((row) => row.dateIso)
      .filter((row) => row.date >= MIN_DISPLAY_DATE)
      .sort((a, b) => a.dateIso.localeCompare(b.dateIso));
  }, [dailyMargin]);
  const dailyEs99Series = useMemo<DailyEs99Row[]>(() => {
    return [...dailyEs99]
      .map((row) => {
        const date = String(row.date || "");
        return {
          ...row,
          date,
          dateIso: date ? `${date}T00:00:00` : ""
        } as DailyEs99Row;
      })
      .filter((row) => row.dateIso)
      .filter((row) => row.date >= MIN_DISPLAY_DATE)
      .sort((a, b) => a.dateIso.localeCompare(b.dateIso));
  }, [dailyEs99]);
  const weeklyMarginSeries = useMemo<PeriodMarginRow[]>(() => {
    return [...weeklyMarginPnl]
      .map((row) => ({
        ...row,
        periodStart: String(row.period_start || ""),
        periodLabel: String(row.period_label || "")
      }) as PeriodMarginRow)
      .filter((row) => row.periodStart)
      .filter((row) => row.periodStart >= MIN_DISPLAY_DATE)
      .sort((a, b) => a.periodStart.localeCompare(b.periodStart));
  }, [weeklyMarginPnl]);
  const monthlyMarginSeries = useMemo<PeriodMarginRow[]>(() => {
    return [...monthlyMarginPnl]
      .map((row) => ({
        ...row,
        periodStart: String(row.period_start || ""),
        periodLabel: String(row.period_label || "")
      }) as PeriodMarginRow)
      .filter((row) => row.periodStart)
      .filter((row) => row.periodStart >= MIN_DISPLAY_DATE)
      .sort((a, b) => a.periodStart.localeCompare(b.periodStart));
  }, [monthlyMarginPnl]);
  const periodMarginSeries = marginView === "week" ? weeklyMarginSeries : monthlyMarginSeries;
  const periodRoiSeries = useMemo<PeriodRoiRow[]>(() => {
    return periodMarginSeries.map((row) => {
      const pnl = Number(row.total_pnl || 0);
      const avgMargin = Number(row.avg_margin_blocked || 0);
      const roiPct = avgMargin > 0 ? (pnl / avgMargin) * 100 : 0;
      const stackTop = Number(row.avg_margin_blocked || 0) / 100000 + Math.max(Number(row.total_pnl || 0), 0) / 100000;
      return {
        ...row,
        roiPct,
        stackTop,
      };
    });
  }, [periodMarginSeries]);
  const xRange = useMemo(() => {
    if (!trades.length) return null;
    const dates = trades
      .map((t) => new Date(String(t.expiry_date || "")))
      .filter((d) => !Number.isNaN(d.getTime()));
    if (!dates.length) return null;
    const start = new Date(Math.min(...dates.map((d) => d.getTime())));
    const end = new Date();
    end.setMonth(end.getMonth() + 3);
    return [start.toISOString(), end.toISOString()] as [string, string];
  }, [trades]);

  useEffect(() => {
    if (!refreshKey) setRefreshKey(Date.now());
  }, [refreshKey]);

  useEffect(() => {
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Tradebook CSV</span>
          <input
            className="control-input"
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Ledger CSV</span>
          <input
            className="control-input"
            type="file"
            accept=".csv"
            onChange={(e) => setLedgerFile(e.target.files?.[0] || null)}
          />
        </label>
        <label className="control-field">
          <span className="control-label">FO Tradebook CSV</span>
          <input
            className="control-input"
            type="file"
            accept=".csv"
            onChange={(e) => setFoTradebookFile(e.target.files?.[0] || null)}
          />
        </label>
        <label className="control-field control-inline" style={{ alignSelf: "end" }}>
          <button
            className="control-input"
            disabled={(!file && !ledgerFile && !foTradebookFile) || uploading}
            onClick={async () => {
              if (!file && !ledgerFile && !foTradebookFile) return;
              setUploading(true);
              try {
                if (file) {
                  const tradebookForm = new FormData();
                  tradebookForm.append("file", file);
                  const tradebookRes = await fetch(`${API_BASE_URL}/historical/tradebook`, {
                    method: "POST",
                    body: tradebookForm
                  });
                  if (!tradebookRes.ok) {
                    const txt = await tradebookRes.text();
                    throw new Error(txt || "Tradebook upload failed");
                  }
                }
                if (ledgerFile) {
                  const ledgerForm = new FormData();
                  ledgerForm.append("file", ledgerFile);
                  const ledgerRes = await fetch(`${API_BASE_URL}/historical/ledger`, {
                    method: "POST",
                    body: ledgerForm
                  });
                  if (!ledgerRes.ok) {
                    const txt = await ledgerRes.text();
                    throw new Error(txt || "Ledger upload failed");
                  }
                }
                if (foTradebookFile) {
                  const foTradebookForm = new FormData();
                  foTradebookForm.append("file", foTradebookFile);
                  const foTradebookRes = await fetch(`${API_BASE_URL}/historical/tradebook-fo`, {
                    method: "POST",
                    body: foTradebookForm
                  });
                  if (!foTradebookRes.ok) {
                    const txt = await foTradebookRes.text();
                    throw new Error(txt || "FO tradebook upload failed");
                  }
                }
                notify({
                  type: "success",
                  title: "Uploaded",
                  message: [file?.name, ledgerFile?.name, foTradebookFile?.name].filter(Boolean).join(" + ")
                });
                setRefreshKey(Date.now());
                setFile(null);
                setLedgerFile(null);
                setFoTradebookFile(null);
              } catch (err) {
                notify({ type: "error", title: "Upload failed", message: String(err) });
              } finally {
                setUploading(false);
              }
            }}
          >
            {uploading ? "Uploading..." : "Upload"}
          </button>
        </label>
        <label className="control-field control-inline" style={{ alignSelf: "end" }}>
          <button
            className="control-input"
            onClick={() => setRefreshKey(Date.now())}
          >
            Load Default
          </button>
        </label>
      </div>
    );
    setControls({
      title: "Controls",
      summary: (
        <>
          <span>
            <span className="controls-summary-key">Default</span> `database/tradebook.csv`
          </span>
          <span>
            <span className="controls-summary-key">File</span> {file?.name || "—"}
          </span>
          <span>
            <span className="controls-summary-key">Ledger</span> {ledgerFile?.name || "`database/ledger.csv`"}
          </span>
          <span>
            <span className="controls-summary-key">FO Booked P&L</span> {foTradebookFile?.name || "`database/tradebook_fo.csv`"}
          </span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, file, ledgerFile, foTradebookFile, uploading, notify]);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      <SectionCard title="Monthly Performance">
        {loading || !data ? <LoadingState /> : (
          <Plot
            data={[
              {
                type: "bar",
                x: monthly.map((m) => m.month_label as string),
                y: monthly.map((m) => Number(m.total_pnl || 0)),
                name: "Monthly P&L",
                marker: {
                  color: monthly.map((m) => (Number(m.total_pnl || 0) >= 0 ? "#2ecc71" : "#ff4d4d"))
                }
              },
              {
                type: "scatter",
                mode: "lines+markers",
                x: monthly.map((m) => m.month_label as string),
                y: monthly.map((m) => Number(m.total_margin_est || 0)),
                name: "Estimated Margin",
                yaxis: "y2",
                line: { color: "#4f8cff", width: 2 }
              },
              {
                type: "scatter",
                mode: "lines+markers",
                x: monthly.map((m) => m.month_label as string),
                y: monthly.map((m) => Number(m.monthly_return_pct || 0)),
                name: "Monthly Return %",
                yaxis: "y3",
                line: { color: "#ff9f1a", width: 2, dash: "dot" }
              }
            ]}
            layout={{
              height: 360,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 20, t: 20, b: 30 },
              xaxis: { tickangle: -20, tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
              yaxis: { title: "P&L (₹)", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
              yaxis2: {
                title: "Margin (₹)",
                overlaying: "y",
                side: "right",
                tickfont: { color: "#95b8ff", size: 10 },
                titlefont: { color: "#95b8ff", size: 11 }
              },
              yaxis3: {
                title: "Return (%)",
                overlaying: "y",
                side: "right",
                anchor: "free",
                position: 0.97,
                tickfont: { color: "#ffcf8a", size: 10 },
                titlefont: { color: "#ffcf8a", size: 11 }
              },
              legend: { orientation: "h", y: 1.12 }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <SectionCard title="P&L Over Time">
        {loading || !data ? <LoadingState /> : (
          <Plot
            data={[{
              type: "scatter",
              mode: "lines",
              x: trades.map((t) => String(t.expiry_date || "")),
              y: trades.reduce<number[]>((acc, t) => {
                const last = acc[acc.length - 1] || 0;
                acc.push(last + Number(t.realized_pnl || 0));
                return acc;
              }, []),
              line: { color: "#2EC4B6" }
            }]}
            layout={{
              height: 300,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 20, t: 20, b: 30 },
              xaxis: { type: "date", range: xRange || undefined, tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
              yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <SectionCard title="Daily F&O Margin Blocked">
        {loading || !data ? <LoadingState /> : dailyMarginSeries.length === 0 ? (
          <ErrorState message="No ledger margin data found. Upload a ledger CSV or place it at `database/ledger.csv`." />
        ) : (
          <Plot
            data={[
              {
                type: "bar",
                x: dailyMarginSeries.map((row) => row.dateIso),
                y: dailyMarginSeries.map((row) => Number(row.gross_blocked || 0) / 100000),
                name: "Margin Blocked",
                marker: { color: "#4f8cff" }
              }
            ]}
            layout={{
              height: 320,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 20, t: 20, b: 30 },
              xaxis: {
                type: "date",
                tickformat: "%Y-%m-%d",
                tickfont: { color: "#b9c4d6", size: 10 },
                titlefont: { color: "#b9c4d6", size: 11 }
              },
              yaxis: { title: "Margin Blocked (Lakh)", tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
              legend: { orientation: "h", y: 1.12 }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <SectionCard title="Daily Portfolio ES99">
        {loading || !data ? <LoadingState /> : dailyEs99Series.length === 0 ? (
          <ErrorState message="No daily ES99 series available for the current NIFTY cache window." />
        ) : (
          <Plot
            data={[
              {
                type: "scatter",
                mode: "lines+markers",
                x: dailyEs99Series.map((row) => row.dateIso),
                y: dailyEs99Series.map((row) => Number(row.es99_inr || 0) / 100000),
                name: "ES99",
                line: { color: "#ff6b6b", width: 2 }
              },
              {
                type: "scatter",
                mode: "lines",
                x: dailyEs99Series.map((row) => row.dateIso),
                y: dailyEs99Series.map((row) => Number(row.blocked_margin || 0) / 100000),
                name: "Blocked Margin",
                yaxis: "y2",
                line: { color: "#4f8cff", width: 2, dash: "dot" }
              }
            ]}
            layout={{
              height: 320,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 20, t: 20, b: 30 },
              xaxis: {
                type: "date",
                tickformat: "%Y-%m-%d",
                tickfont: { color: "#b9c4d6", size: 10 },
                titlefont: { color: "#b9c4d6", size: 11 }
              },
              yaxis: { title: "ES99 (Lakh)", tickfont: { color: "#ffb3b3", size: 10 }, titlefont: { color: "#ffb3b3", size: 11 } },
              yaxis2: {
                title: "Margin (Lakh)",
                overlaying: "y",
                side: "right",
                tickfont: { color: "#95b8ff", size: 10 },
                titlefont: { color: "#95b8ff", size: 11 }
              },
              legend: { orientation: "h", y: 1.12 }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <SectionCard title="P&L and Avg Margin Blocked">
        {loading || !data ? <LoadingState /> : periodMarginSeries.length === 0 ? (
          <ErrorState message={`No ${marginView}ly margin data available.`} />
        ) : (
          <>
            <div className="control-grid" style={{ marginBottom: 12 }}>
              <label className="control-field control-inline" style={{ maxWidth: 220 }}>
                <span className="control-label">View</span>
                <select
                  className="control-input"
                  value={marginView}
                  onChange={(e) => setMarginView(e.target.value as "week" | "month")}
                >
                  <option value="week">Week</option>
                  <option value="month">Month</option>
                </select>
              </label>
            </div>
            <Plot
              data={[
                {
                  type: "bar",
                  x: periodRoiSeries.map((row) => String(row.periodLabel || "")),
                  y: periodRoiSeries.map((row) => Number(row.avg_margin_blocked || 0) / 100000),
                  name: "Avg Margin Blocked",
                  marker: { color: "#4f8cff" }
                },
                {
                  type: "bar",
                  x: periodRoiSeries.map((row) => String(row.periodLabel || "")),
                  y: periodRoiSeries.map((row) => Number(row.total_pnl || 0) / 100000),
                  name: "P&L",
                  marker: {
                    color: periodRoiSeries.map((row) => (Number(row.total_pnl || 0) >= 0 ? "#2ecc71" : "#ff4d4d"))
                  }
                },
                {
                  type: "scatter",
                  mode: "text",
                  x: periodRoiSeries.map((row) => String(row.periodLabel || "")),
                  y: periodRoiSeries.map((row) => Number(row.stackTop || 0)),
                  text: periodRoiSeries.map((row) => `${Number(row.roiPct || 0).toFixed(1)}%`),
                  textposition: "top center",
                  textfont: { color: "#e7eefb", size: 11 },
                  hoverinfo: "skip",
                  showlegend: false
                }
              ]}
              layout={{
                barmode: "relative",
                height: 320,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 30, r: 20, t: 20, b: 50 },
                xaxis: {
                  tickangle: marginView === "week" ? -35 : 0,
                  tickfont: { color: "#b9c4d6", size: 10 },
                  titlefont: { color: "#b9c4d6", size: 11 }
                },
                yaxis: {
                  title: "Lakh",
                  tickfont: { color: "#b9c4d6", size: 10 },
                  titlefont: { color: "#b9c4d6", size: 11 }
                },
                legend: { orientation: "h", y: 1.12 }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          </>
        )}
      </SectionCard>

      <SectionCard title="Trade P&L by Expiry">
        {loading || !data ? <LoadingState /> : (
          <Plot
            data={[{
              type: "bar",
              x: trades.map((t) => String(t.expiry_date || "")),
              y: trades.map((t) => Number(t.realized_pnl || 0)),
              marker: {
                color: trades.map((t) => (Number(t.realized_pnl || 0) >= 0 ? "#2ecc71" : "#ff4d4d"))
              }
            }]}
            layout={{
              height: 280,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 20, t: 20, b: 30 },
              xaxis: { type: "date", range: xRange || undefined, tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
              yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <div className="chart-grid-1x2">
        <SectionCard title="P&L % Distribution">
          {loading || !data ? <LoadingState /> : (
            <Plot
              data={[{
                type: "histogram",
                x: data.pnl_pct || [],
                marker: { color: "#2EC4B6" }
              }]}
              layout={{
                height: 260,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 30, r: 20, t: 20, b: 30 },
                xaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          )}
        </SectionCard>

        <SectionCard title="Win/Loss Distribution">
          {loading || !data ? <LoadingState /> : (
            <Plot
              data={[{
                type: "bar",
                x: ["Wins", "Losses"],
                y: [data.win_loss?.wins || 0, data.win_loss?.losses || 0],
                marker: { color: ["#2ecc71", "#ff4d4d"] }
              }]}
              layout={{
                height: 260,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 30, r: 20, t: 20, b: 30 },
                xaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          )}
        </SectionCard>
      </div>

      <SectionCard title="Top Contributors">
        {loading || !data ? <LoadingState /> : (
          <DataTable
            maxHeight={400}
            columns={[
              { key: "symbol", label: "Symbol" },
              { key: "expiry_date", label: "Expiry" },
              { key: "option_type", label: "Type" },
              { key: "realized_pnl", label: "Realized PnL", format: (v) => formatInr(v) }
            ]}
            rows={data.top_contributors || []}
          />
        )}
      </SectionCard>

      <SectionCard title="Bottom Contributors">
        {loading || !data ? <LoadingState /> : (
          <DataTable
            maxHeight={400}
            columns={[
              { key: "symbol", label: "Symbol" },
              { key: "expiry_date", label: "Expiry" },
              { key: "option_type", label: "Type" },
              { key: "realized_pnl", label: "Realized PnL", format: (v) => formatInr(v) }
            ]}
            rows={data.bottom_contributors || []}
          />
        )}
      </SectionCard>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">Trade Table (Debug)</div>
        </summary>
        <div className="section-body">
          {loading || !data ? <LoadingState /> : (
            <DataTable
              maxHeight={400}
              columns={[
                { key: "symbol", label: "Symbol" },
                { key: "expiry_date", label: "Expiry" },
                { key: "option_type", label: "Type" },
                { key: "expiry_type", label: "Expiry Type" },
                { key: "strike", label: "Strike" },
                { key: "realized_pnl", label: "Realized PnL", format: (v) => formatInr(v) }
              ]}
              rows={trades}
            />
          )}
        </div>
      </details>
    </>
  );
}
