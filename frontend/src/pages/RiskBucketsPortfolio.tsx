import { useMemo } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import DataTable from "../components/DataTable";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { formatInr, formatPct, formatNumber } from "../components/format";
import { usePortfolio } from "../state/PortfolioContext";

type PortfolioRB = {
  sim_summary: Record<string, number> | null;
  sim_fan: { Day?: number; P1?: number; P10?: number; P50?: number }[];
  portfolio_es99_inr: number;
  portfolio_es99_pct: number;
  margin_used_pct: number | null;
  top_underlying: Record<string, unknown>[];
  by_bucket: Record<string, unknown>[];
  by_week: Record<string, unknown>[];
  zone: Record<string, unknown>;
};

export default function RiskBucketsPortfolio() {
  const { data, error, loading } = useCachedApi<PortfolioRB>(
    "risk_buckets_portfolio",
    "/risk-buckets/portfolio",
    60_000
  );
  const { summary, positions } = usePortfolio();
  const portfolioSummary = summary as any;

  const s = data?.sim_summary || {};

  const riskAlerts = useMemo(() => {
    if (!portfolioSummary || !positions?.positions) return [];
    const alerts: string[] = [];
    const totalDelta = Number(portfolioSummary.greeks?.net_delta || 0);
    const totalGamma = Number(portfolioSummary.greeks?.net_gamma || 0);
    const marginPct = Number(portfolioSummary.margin_pct || 0);
    const totalPnl = Number(portfolioSummary.total_pnl || 0);
    const totalTheta = Number(portfolioSummary.theta_day || 0);
    const avgDte = positions.positions.length
      ? positions.positions.reduce((acc, p) => acc + Number(p.dte || 0), 0) / positions.positions.length
      : 30;
    const daysToRecover = totalTheta ? Math.abs(totalPnl / totalTheta) : 999;
    const thetaEfficiency = totalTheta ? (totalPnl / totalTheta) * 100 : 0;

    if (Math.abs(totalDelta) > 100) alerts.push("🔴 CRITICAL: Net Delta > ±100 - High directional risk");
    else if (Math.abs(totalDelta) > 40) alerts.push("🟡 WARNING: Net Delta > ±40 - Monitor directional exposure");
    else alerts.push("🟢 OK: Delta is neutral");

    if (marginPct > 80) alerts.push("🔴 CRITICAL: Margin utilization > 70% - Limited room for adjustments");
    else if (marginPct > 50) alerts.push("🟡 WARNING: Margin utilization > 50%");
    else alerts.push("🟢 OK: Margin utilization healthy");

    if (daysToRecover > avgDte) alerts.push("🔴 CRITICAL: Cannot recover losses by expiry with current theta");
    else if (daysToRecover > avgDte * 0.7) alerts.push("🟡 WARNING: Tight timeline to recover losses");
    else alerts.push("🟢 OK: Recovery timeline manageable");

    if (thetaEfficiency < -200) alerts.push("🔴 CRITICAL: Theta efficiency < -200% - Directional problem, not time decay");
    else if (thetaEfficiency < -100) alerts.push("🟡 WARNING: Theta efficiency negative");

    if (totalGamma < -0.5 && avgDte < 7) alerts.push("🔴 CRITICAL: High negative gamma near expiry - Risk of rapid delta changes");

    return alerts;
  }, [portfolioSummary, positions]);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      <SectionCard title="Forward Simulation KPIs">
        {loading || !data ? <LoadingState /> : (
          <MetricGrid>
            <MetricCard label="Mean" value={formatInr(s.mean)} />
            <MetricCard label="Median" value={formatInr(s.median)} />
            <MetricCard label="P5" value={formatInr(s.p5)} />
            <MetricCard label="P1" value={formatInr(s.p1)} />
            <MetricCard label="Prob. Loss" value={formatPct((Number(s.prob_loss) || 0) * 100)} />
            <MetricCard label="Prob. Breach" value={s.prob_breach != null ? formatPct((Number(s.prob_breach) || 0) * 100) : "N/A"} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Forward Simulation Fan Chart">
        {loading || !data ? <LoadingState /> : (
          <>
            <MetricGrid>
              <MetricCard label="ES99 (₹)" value={formatInr(data.portfolio_es99_inr)} />
              <MetricCard label="ES99 (%)" value={formatPct(data.portfolio_es99_pct)} />
              <MetricCard label="Margin Used %" value={data.margin_used_pct != null ? formatPct(data.margin_used_pct) : "N/A"} />
            </MetricGrid>
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "lines",
                  x: (data.sim_fan || []).map((r) => r.Day),
                  y: (data.sim_fan || []).map((r) => Number(r.P50 || 0)),
                  name: "P50",
                  line: { color: "#2D7DFF", width: 2 }
                },
                {
                  type: "scatter",
                  mode: "lines",
                  x: (data.sim_fan || []).map((r) => r.Day),
                  y: (data.sim_fan || []).map((r) => Number(r.P10 || 0)),
                  name: "P10",
                  line: { color: "#FFB020", dash: "dot" }
                },
                {
                  type: "scatter",
                  mode: "lines",
                  x: (data.sim_fan || []).map((r) => r.Day),
                  y: (data.sim_fan || []).map((r) => Number(r.P1 || 0)),
                  name: "P1",
                  line: { color: "#FF4D4D", dash: "dot" }
                }
              ]}
              layout={{ height: 360, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 40, r: 20, t: 20, b: 40 }, xaxis: { title: "Day" }, yaxis: { title: "P&L (₹)" } }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
            />
          </>
        )}
      </SectionCard>

      <SectionCard title="Zone & Greek Risk Summary">
        {loading || !data ? <LoadingState /> : (
          <>
            <MetricGrid>
              <MetricCard label="IV Regime" value={String(data.zone?.iv_regime || "—")} />
              <MetricCard label="Theta / 1L" value={formatNumber(data.zone?.theta_norm, 1)} />
              <MetricCard label="Gamma / 1L" value={formatNumber(data.zone?.gamma_norm, 4)} />
              <MetricCard label="Vega / 1L" value={formatNumber(data.zone?.vega_norm, 1)} />
              <MetricCard label="Zone" value={`Z${data.zone?.zone_num} ${data.zone?.zone_name || ""}`} />
            </MetricGrid>
            {!portfolioSummary || !positions?.positions ? (
              <LoadingState />
            ) : (
              <div className="risk-alerts">
                {riskAlerts.map((a) => (
                  <div key={a} className="risk-alert">
                    {a}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </SectionCard>
    </>
  );
}
