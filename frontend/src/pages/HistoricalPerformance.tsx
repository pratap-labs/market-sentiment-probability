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
};

export default function HistoricalPerformance() {
  const [uploading, setUploading] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [refreshKey, setRefreshKey] = useState<number>(0);
  const { notify } = useNotifications();
  const { setControls } = useControls();
  const { data, error, loading } = useCachedApi<Historical>(
    `historical_summary_${refreshKey}`,
    `/historical/summary${refreshKey ? `?t=${refreshKey}` : ""}`,
    60_000
  );

  const trades = data?.trades || [];
  const monthly = data?.monthly || [];
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
        <label className="control-field control-inline" style={{ alignSelf: "end" }}>
          <button
            className="control-input"
            disabled={!file || uploading}
            onClick={async () => {
              if (!file) return;
              setUploading(true);
              try {
                const form = new FormData();
                form.append("file", file);
                const res = await fetch(`${API_BASE_URL}/historical/tradebook`, {
                  method: "POST",
                  body: form
                });
                if (!res.ok) {
                  const txt = await res.text();
                  throw new Error(txt || "Upload failed");
                }
                notify({ type: "success", title: "Uploaded", message: file.name });
                setRefreshKey(Date.now());
                setFile(null);
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
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, file, uploading, notify]);

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
