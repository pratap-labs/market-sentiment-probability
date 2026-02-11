import { useEffect, useMemo, useState } from "react";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import DataTable from "../components/DataTable";
import SectionCard from "../components/SectionCard";
import { formatInr, formatPct, formatNumber } from "../components/format";
import { usePortfolio } from "../state/PortfolioContext";
import { useControls } from "../state/ControlsContext";
import { useCachedApi } from "../hooks/useCachedApi";

type PortfolioRisk = {
  sim_summary: Record<string, number> | null;
  portfolio_es99_inr: number;
  portfolio_es99_pct: number;
};

export default function PortfolioDashboard() {
  const { summary, positions, loading, error, refreshWithSpot } = usePortfolio();
  const portfolioRisk = useCachedApi<PortfolioRisk>("risk_buckets_portfolio", "/risk-buckets/portfolio", 60_000);
  const s = summary as any;
  const [spotOverride, setSpotOverride] = useState<number | null>(null);
  const [expiryFilter, setExpiryFilter] = useState<string>("All");
  const { setControls } = useControls();

  const expiryOptions = useMemo(() => {
    const rows = positions?.positions || [];
    const dates = rows
      .map((p) => {
        const exp = p.expiry as any;
        if (!exp) return "";
        if (typeof exp === "string") return exp.slice(0, 10);
        return String(exp);
      })
      .filter(Boolean);
    return ["All", ...Array.from(new Set(dates)).sort()];
  }, [positions]);

  const filteredPositions = useMemo(() => {
    const rows = positions?.positions || [];
    if (expiryFilter === "All") return rows;
    return rows.filter((p) => {
      const exp = p.expiry as any;
      const val = typeof exp === "string" ? exp.slice(0, 10) : String(exp || "");
      return val === expiryFilter;
    });
  }, [positions, expiryFilter]);

  const riskAlerts = useMemo(() => {
    if (!s || !positions?.positions) return [];
    const alerts: string[] = [];
    const totalDelta = Number(s.greeks?.net_delta || 0);
    const totalGamma = Number(s.greeks?.net_gamma || 0);
    const marginPct = Number(s.margin_pct || 0);
    const totalPnl = Number(s.total_pnl || 0);
    const totalTheta = Number(s.theta_day || 0);
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
  }, [s, positions]);

  const concentration = useMemo(() => {
    const rows = positions?.positions || [];
    const accountSize = Number(s?.account_size || 0);
    const groups: Record<string, { count: number; pnl: number; notional: number }> = {};
    rows.forEach((p) => {
      const exp = p.expiry as any;
      const key = exp ? (typeof exp === "string" ? exp.slice(0, 10) : String(exp)) : "Unknown";
      if (!groups[key]) groups[key] = { count: 0, pnl: 0, notional: 0 };
      groups[key].count += 1;
      groups[key].pnl += Number(p.pnl || 0);
      groups[key].notional += Math.abs(Number(p.quantity || 0)) * Number(p.strike || 0);
    });
    const rowsOut = Object.entries(groups).map(([exp, data]) => ({
      expiry: exp,
      positions: data.count,
      pnl: data.pnl,
      notional: data.notional,
      leverage: accountSize > 0 ? data.notional / accountSize : 0
    }));
    const largest = [...rows]
      .sort((a, b) => Math.abs(Number(b.quantity || 0) * Number(b.strike || 0)) - Math.abs(Number(a.quantity || 0) * Number(a.strike || 0)))
      .slice(0, 5)
      .map((p) => {
        const notional = Math.abs(Number(p.quantity || 0)) * Number(p.strike || 0);
        return {
          symbol: String(p.tradingsymbol || ""),
          notional,
          pct: accountSize > 0 ? (notional / accountSize) * 100 : 0
        };
      });
    return { rows: rowsOut, largest };
  }, [positions, s]);

  useEffect(() => {
    const currentSpot = Number(s?.current_spot || positions?.current_spot || 25000);
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Spot Override (NIFTY)</span>
          <input
            className="control-input"
            type="number"
            step="10"
            value={spotOverride ?? currentSpot}
            onChange={(e) => setSpotOverride(Number(e.target.value))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Positions Expiry Filter</span>
          <select className="control-input" value={expiryFilter} onChange={(e) => setExpiryFilter(e.target.value)}>
            {expiryOptions.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
          </select>
        </label>
        <button
          className="control-input"
          style={{ background: "linear-gradient(135deg, var(--gs-accent), var(--gs-accent2))", border: "none", color: "#fff" }}
          onClick={() => {
            const v = spotOverride ?? currentSpot;
            refreshWithSpot(v);
          }}
        >
          Recompute Greeks (Portfolio)
        </button>
      </div>
    );
    setControls({
      key: `portfolio:${spotOverride ?? currentSpot}:${expiryFilter}`,
      title: "Controls",
      summary: (
        <>
          <span>
            <span className="controls-summary-key">Spot</span> {spotOverride ?? currentSpot}
          </span>&nbsp;
          <span>
            <span className="controls-summary-key">Expiry</span> {expiryFilter}
          </span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, spotOverride, expiryFilter, expiryOptions, refreshWithSpot, s, positions]);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      {!loading && (!positions?.positions || !positions.positions.length) ? (
        <SectionCard>
          <div>No positions loaded. Fetch positions from the Positions tab first.</div>
        </SectionCard>
      ) : null}

      {/* Controls moved to sticky ControlsBar */}

      <SectionCard title="Capital & Performance">
        {loading || !s ? (
          <LoadingState />
        ) : (
          <MetricGrid>
            <MetricCard label="Account Size" value={formatInr(s.account_size)} />
            <MetricCard
              label="Margin Used"
              value={formatInr(s.margin_used)}
              delta={`${formatPct(s.margin_pct)} ${Number(s.margin_pct || 0) < 50 ? "🟢" : Number(s.margin_pct || 0) < 70 ? "🟡" : "🔴"}`}
            />
            <MetricCard label="Net P&L" value={formatInr(s.total_pnl)} delta={formatPct(s.roi_pct)} tone={s.total_pnl >= 0 ? "positive" : "negative"} />
            <MetricCard label="ROI (Annualized)" value={formatPct(s.roi_annualized)} tone={s.roi_annualized >= 0 ? "positive" : "negative"} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard
        title="Portfolio Risk Summary"
        tooltip="Forward simulation uses Monte Carlo paths over the configured horizon (default 10 days) with IV mode and shock settings from Risk Buckets Settings. KPIs: Mean/Median are expected outcomes; P5/P1 are tail quantiles (worse-case bands). Prob. Loss is chance of negative P&L; Prob. Breach is chance of crossing the configured loss limit. ES99 shows expected loss of the worst 1% tail."
      >
        {portfolioRisk.loading || !portfolioRisk.data ? (
          <LoadingState />
        ) : (
          <MetricGrid>
            <MetricCard label="Mean" value={formatInr(portfolioRisk.data.sim_summary?.mean)} />
            <MetricCard label="Median" value={formatInr(portfolioRisk.data.sim_summary?.median)} />
            <MetricCard label="P5" value={formatInr(portfolioRisk.data.sim_summary?.p5)} />
            <MetricCard label="P1" value={formatInr(portfolioRisk.data.sim_summary?.p1)} />
            <MetricCard label="Prob. Loss" value={formatPct((Number(portfolioRisk.data.sim_summary?.prob_loss) || 0) * 100)} />
            <MetricCard label="Prob. Breach" value={formatPct((Number(portfolioRisk.data.sim_summary?.prob_breach) || 0) * 100)} />
            <MetricCard label="ES99 (₹)" value={formatInr(portfolioRisk.data.portfolio_es99_inr)} />
            <MetricCard label="ES99 (%)" value={formatPct(portfolioRisk.data.portfolio_es99_pct)} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard
        title="Greeks & Risk"
        tooltip="Delta conversions assume lot size for ₹/pt calculation. If your positions already include lot-size in total delta, ₹/pt equals Net Delta."
      >
        {loading || !s ? (
          <LoadingState />
        ) : (
          <MetricGrid>
            <MetricCard
              label="Net Delta (units)"
              value={`${formatNumber(s.greeks?.net_delta, 2)} ${Math.abs(Number(s.greeks?.net_delta || 0)) < 40 ? "🟢" : Math.abs(Number(s.greeks?.net_delta || 0)) < 100 ? "🟡" : "🔴"}`}
            />
            <MetricCard label="Delta (₹/pt)" value={formatInr(s.greeks?.net_delta)} />
            <MetricCard label="Theta/Day" value={formatInr(s.theta_day)} delta={`${formatPct(s.theta_pct_capital)} of capital`} tone={s.theta_day >= 0 ? "positive" : "negative"} />
            <MetricCard label="Net Gamma" value={`${formatNumber(s.greeks?.net_gamma, 4)} ${Number(s.greeks?.net_gamma || 0) < -0.5 ? "⚠️" : ""}`} />
            <MetricCard label="Net Vega" value={formatInr(s.greeks?.net_vega)} />

            <MetricCard
              label="Days to Recover"
              value={s.days_to_recover !== null ? formatNumber(s.days_to_recover, 1) : "N/A"}
              delta={`Avg DTE: ${positions?.positions?.length ? (positions.positions.reduce((acc, p) => acc + Number(p.dte || 0), 0) / positions.positions.length).toFixed(0) : "30"}`}
            />
            <MetricCard
              label="Theta Efficiency"
              value={s.theta_efficiency !== null ? formatNumber(s.theta_efficiency, 0) : "N/A"}
              delta={`${Number(s.theta_efficiency || 0) < -200 ? "🔴" : Number(s.theta_efficiency || 0) < -100 ? "🟡" : "🟢"}`}
            />
            <MetricCard
              label="Notional Exposure"
              value={formatInr(s.notional_exposure)}
              delta={`${Number(s.leverage_ratio || 0).toFixed(0)} × capital ${Number(s.leverage_ratio || 0) < 50 ? "🟢" : Number(s.leverage_ratio || 0) < 100 ? "🟡" : "🔴"}`}
            />
            <MetricCard label="Delta Notional" value={formatInr(s.delta_notional)} delta={formatPct(s.delta_notional_pct)} />
            <MetricCard label="Vega % Capital" value={formatPct(s.vega_pct_capital)} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Portfolio–Market Alignment">
        {loading || !s ? (
          <LoadingState />
        ) : (
          <>
            <div className="alignment-grid">
              <div className="alignment-card">
                <div className="alignment-label">Health</div>
                <div className="alignment-value">{`${s.health.emoji} ${s.health.status}`}</div>
              </div>
              <div className="alignment-card">
                <div className="alignment-label">Market Signal</div>
                <div className="alignment-value">{`${s.market_signal.emoji} ${s.market_signal.text}`}</div>
              </div>
              <div className="alignment-card">
                <div className="alignment-label">Signal Note</div>
                <div className="alignment-value">{s.market_signal.recommendation}</div>
              </div>
            </div>
            <div className="alignment-grid-4">
              {(s.alignment || []).map((row: any, idx: number) => (
                <div key={`${row.Category || row.category || "row"}-${idx}`} className="alignment-panel">
                  <div className="alignment-panel-header">
                    <div className="alignment-panel-title">{row.Category || row.category}</div>
                  </div>
                  <div className="alignment-panel-body">
                    <div className="alignment-chip-row">
                      <span className="alignment-chip">{row.Market || row.market}</span>
                      <span className="alignment-chip">{row.Portfolio || row.portfolio}</span>
                    </div>
                    <div className="alignment-panel-status">{row.Status || row.status}</div>
                    <div className="alignment-panel-action">{row.Action || row.action}</div>
                  </div>
                </div>
              ))}
            </div>
            <div className="alignment-grid-4" style={{ marginTop: "16px" }}>
              {(s.recommendations || []).map((row: any, idx: number) => (
                <div key={`${row.type || "rec"}-${idx}`} className="alignment-panel">
                  <div className="alignment-panel-header">
                    <div className="alignment-panel-title">{row.type}</div>
                  </div>
                  <div className="alignment-panel-body">
                    <div className="alignment-panel-action">{row.recommendation}</div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </SectionCard>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">Positions Table</div>
        </summary>
        <div className="section-body">
          {loading || !positions ? (
            <LoadingState />
          ) : (
            <DataTable
              columns={[
                { key: "tradingsymbol", label: "Symbol" },
                { key: "quantity", label: "Qty" },
                { key: "strike", label: "Strike" },
                { key: "option_type", label: "Type" },
                { key: "dte", label: "DTE" },
                { key: "average_price", label: "Avg Price", format: (v) => formatInr(v) },
                { key: "last_price", label: "LTP", format: (v) => formatInr(v) },
                { key: "pnl", label: "P&L", format: (v) => formatInr(v) },
                { key: "implied_vol", label: "IV", format: (v) => formatPct((Number(v) || 0) * 100) },
                { key: "delta", label: "Delta", format: (v) => formatNumber(v, 3) },
                { key: "gamma", label: "Gamma", format: (v) => formatNumber(v, 4) },
                { key: "vega", label: "Vega", format: (v) => formatNumber(v, 2) },
                { key: "theta", label: "Theta", format: (v) => formatNumber(v, 2) },
                { key: "position_delta", label: "Pos Δ", format: (v) => formatNumber(v, 2) },
                { key: "position_gamma", label: "Pos Γ", format: (v) => formatNumber(v, 3) },
                { key: "position_vega", label: "Pos V", format: (v) => formatNumber(v, 1) },
                { key: "position_theta", label: "Pos Θ", format: (v) => formatNumber(v, 1) }
              ]}
              rows={filteredPositions}
            />
          )}
        </div>
      </details>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">Position Concentration</div>
        </summary>
        <div className="section-body">
          {loading || !positions ? (
            <LoadingState />
          ) : (
            <>
              <DataTable
                columns={[
                  { key: "expiry", label: "Expiry" },
                  { key: "positions", label: "Positions" },
                  { key: "pnl", label: "P&L", format: (v) => formatInr(v) },
                  { key: "notional", label: "Notional", format: (v) => formatInr(v) },
                  { key: "leverage", label: "Leverage (× capital)", format: (v) => formatNumber(Number(v) || 0, 0) }
                ]}
                rows={concentration.rows}
              />
              <div style={{ marginTop: "12px" }}>
                <div style={{ fontWeight: 600, marginBottom: "6px" }}>Largest Positions by Notional</div>
                {concentration.largest.map((p) => (
                  <div key={p.symbol}>
                    - {p.symbol}: {formatInr(p.notional)} ({p.pct.toFixed(1)}% of portfolio)
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </details>
    </>
  );
}
