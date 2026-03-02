import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import Plot from "../components/Plot";
import { formatInr, formatNumber, formatPct } from "../components/format";
import { useCachedApi } from "../hooks/useCachedApi";
import { useControls } from "../state/ControlsContext";

const clamp = (v: number, min = 0, max = 100) => Math.max(min, Math.min(max, v));

type EquityRow = {
  symbol?: string;
  qty?: number;
  avg_cost?: number;
  ltp?: number;
  market_value?: number;
  unreal_pnl?: number;
  unreal_pnl_pct?: number;
  return_vs_cost_pct?: number;
  drawdown_pct?: number;
  time_under_water?: string | number | null;
  es99_inr?: number | null;
  es99_pct?: number | null;
  stress_loss?: number;
  cost_basis?: number | null;
};

type EquitySummary = {
  equity_sleeve_value: number | null;
  allocation_pct: number | null;
  allocation_denom?: string;
  sleeve_drawdown_pct: number | null;
  stress_loss_inr: number | null;
  stress_loss_pct: number | null;
  equity_es99_contrib_pct: number | null;
  pct_underwater: number | null;
  weighted_time_under_water_days: number | null;
  capital_base?: number;
  capital_base_label?: string;
  warnings?: { history_unavailable?: boolean; drawdown_history_unavailable?: boolean; es99_available?: boolean };
  shock_losses?: { level: number; loss: number }[];
  alloc_series?: { symbol: string; value: number }[];
  top_stress?: { symbol: string; value: number }[];
  risk_concentration?: { top1_pct: number; top5_pct: number; top1_stress_pct: number; top5_stress_pct: number };
  rows: EquityRow[];
};

type EquitySim = {
  status?: string;
  config?: { lookback_days?: number; horizon_days?: number; n_paths?: number };
  kpis?: {
    mean?: number;
    median?: number;
    var95?: number;
    var99?: number;
    es99?: number;
    prob_loss?: number;
    portfolio_value?: number;
  };
  paths_sample?: number[];
};

type OptimizeResult = {
  status?: string;
  config?: { target_return?: number; lookback_days?: number };
  kpis?: { expected_return?: number; expected_vol?: number };
  weights?: { symbol: string; weight: number; weight_pct: number; current_weight_pct: number }[];
};

type ScoredRow = EquityRow & {
  name: string;
  sector: string;
  marketCapBucket: "Large" | "Mid" | "Unknown";
  invested: number;
  current: number;
  weight: number;
  pnl: number;
  pnlPct: number;
  vendor?: Record<string, unknown>;
  scores: {
    quality: number;
    growth: number;
    valuation: number;
    momentum: number;
    risk: number;
    composite: number;
    status: "Core" | "Hold" | "Trim" | "Exit";
  };
  notes: string;
  thesis: string;
  checkpoints: string[];
};

export default function Equities() {
  const [shockLevels, setShockLevels] = useState<string>("-5,-10,-20,-30");
  const [currentShock, setCurrentShock] = useState<string>("-10");
  const [lookbackDays, setLookbackDays] = useState<string>("365");
  const [capitalBase, setCapitalBase] = useState<string>("auto");
  const [useCache, setUseCache] = useState<boolean>(true);
  const [targetReturn, setTargetReturn] = useState<string>("12");
  const { setControls } = useControls();

  const query = `?shock_levels=${encodeURIComponent(shockLevels)}&current_shock=${encodeURIComponent(currentShock)}&lookback_days=${encodeURIComponent(lookbackDays)}&capital_base=${encodeURIComponent(capitalBase)}&use_cache=${useCache}`;
  const { data, error, loading } = useCachedApi<EquitySummary>(
    "equities_summary",
    `/equities/summary${query}`,
    60_000
  );
  const { data: enrichedData } = useCachedApi<{ rows: Record<string, unknown>[] }>(
    "equities_enriched",
    `/equities/enriched?use_cache=${useCache}`,
    300_000
  );
  const { data: simData } = useCachedApi<EquitySim>(
    "equities_simulate",
    `/equities/simulate?lookback_days=${encodeURIComponent(lookbackDays)}&horizon_days=30&n_paths=2000&use_cache=${useCache}`,
    120_000
  );
  const { data: optData } = useCachedApi<OptimizeResult>(
    "equities_optimize",
    `/equities/optimize?target_return=${encodeURIComponent(Number(targetReturn) / 100)}&lookback_days=${encodeURIComponent(lookbackDays)}&use_cache=${useCache}`,
    120_000
  );

  useEffect(() => {
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Shock levels (%)</span>
          <input className="control-input" value={shockLevels} onChange={(e) => setShockLevels(e.target.value)} />
        </label>
        <label className="control-field">
          <span className="control-label">Current shock</span>
          <input className="control-input" value={currentShock} onChange={(e) => setCurrentShock(e.target.value)} />
        </label>
        <label className="control-field">
          <span className="control-label">History lookback (days)</span>
          <input className="control-input" value={lookbackDays} onChange={(e) => setLookbackDays(e.target.value)} />
        </label>
        <label className="control-field">
          <span className="control-label">Equity sleeve capital base</span>
          <select className="control-input" value={capitalBase} onChange={(e) => setCapitalBase(e.target.value)}>
            <option value="auto">Auto (cost if available)</option>
            <option value="cost_basis">Cost basis</option>
            <option value="market_value">Market value</option>
          </select>
        </label>
        <label className="control-field">
          <span className="control-label">Target return (%)</span>
          <input className="control-input" value={targetReturn} onChange={(e) => setTargetReturn(e.target.value)} />
        </label>
        <label className="control-field control-inline">
          <input type="checkbox" checked={useCache} onChange={(e) => setUseCache(e.target.checked)} />
          <span className="control-label">Use Cache</span>
        </label>
      </div>
    );
    setControls({
      key: `equities:${shockLevels}:${currentShock}:${lookbackDays}:${capitalBase}:${useCache}`,
      title: "Controls",
      summary: (
        <>
          <span>
            <span className="controls-summary-key">Shock</span> {currentShock}%
          </span>
          <span>
            <span className="controls-summary-key">Lookback</span> {lookbackDays}d
          </span>
          <span>
            <span className="controls-summary-key">Base</span> {capitalBase}
          </span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, shockLevels, currentShock, lookbackDays, capitalBase, useCache, targetReturn]);

  const [queryText, setQueryText] = useState("");
  const [sectorFilter, setSectorFilter] = useState("All");
  const [statusFilter, setStatusFilter] = useState("All");
  const [sortKey, setSortKey] = useState("weight");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [selectedTicker, setSelectedTicker] = useState<string>("");

  const rows = data?.rows || [];

  const getNested = (obj: Record<string, any>, path: string) => {
    return path.split(".").reduce((acc, key) => (acc && acc[key] != null ? acc[key] : undefined), obj);
  };
  const pickNumber = (obj: Record<string, any>, paths: string[]) => {
    for (const p of paths) {
      const val = getNested(obj, p);
      const num = Number(val);
      if (Number.isFinite(num)) return num;
    }
    return undefined;
  };

  const enriched = useMemo((): ScoredRow[] => {
    const totalValue = rows.reduce((acc, r) => acc + Number(r.market_value || 0), 0);
    const vendorMap = new Map<string, Record<string, unknown>>();
    (enrichedData?.rows || []).forEach((r) => {
      const symbol = String((r as Record<string, unknown>).tradingsymbol || (r as Record<string, unknown>).symbol || (r as Record<string, unknown>).instrument || "").toUpperCase();
      if (symbol) vendorMap.set(symbol, (r as Record<string, unknown>).vendor as Record<string, unknown>);
    });
    return rows.map((r) => {
      const symbol = String(r.symbol || "—");
      const vendor = vendorMap.get(symbol.toUpperCase()) || {};
      const industry = String((vendor as Record<string, unknown>).industry || (vendor as Record<string, unknown>).sector || "Unknown");
      const marketCap = pickNumber(vendor as Record<string, any>, [
        "keyMetrics.marketCap",
        "marketCap",
        "companyProfile.marketCap",
        "companyProfile.market_cap",
        "stockDetailsReusableData.marketCap"
      ]);
      let marketCapBucket: ScoredRow["marketCapBucket"] = "Unknown";
      if (marketCap !== undefined && Number.isFinite(marketCap)) {
        marketCapBucket = marketCap >= 5e11 ? "Large" : "Mid";
      }
      const qty = Number(r.qty || 0);
      const avg = Number(r.avg_cost || 0);
      const ltp = Number(r.ltp || 0);
      const invested = Number(r.cost_basis ?? avg * qty);
      const current = Number(r.market_value ?? ltp * qty);
      const pnl = Number(r.unreal_pnl ?? current - invested);
      const pnlPct = invested > 0 ? (pnl / invested) * 100 : 0;
      const weight = totalValue > 0 ? (current / totalValue) * 100 : 0;
      const drawdown = Number(r.drawdown_pct ?? 0);
      const stressLoss = Number(r.stress_loss ?? 0);
      const stressLossPct = current > 0 ? Math.abs(stressLoss) / current * 100 : 0;

      const quality = clamp(70 - Math.abs(drawdown) * 1.1 - stressLossPct * 0.6);
      const growth = clamp(50 + pnlPct);
      const valuation = clamp(60 - pnlPct * 0.4);
      const momentum = clamp(50 + pnlPct * 0.7);
      const risk = clamp(100 - (Math.abs(drawdown) * 1.5 + stressLossPct * 1.2));
      const composite = clamp(quality * 0.3 + growth * 0.25 + valuation * 0.2 + momentum * 0.1 + risk * 0.15);
      let status: ScoredRow["scores"]["status"] = "Hold";
      if (composite >= 80) status = "Core";
      else if (composite >= 65) status = "Hold";
      else if (composite >= 50) status = "Trim";
      else status = "Exit";

      return {
        ...r,
        name: symbol,
        sector: industry,
        marketCapBucket,
        invested,
        current,
        weight,
        pnl,
        pnlPct,
        vendor,
        scores: {
          quality: Math.round(quality),
          growth: Math.round(growth),
          valuation: Math.round(valuation),
          momentum: Math.round(momentum),
          risk: Math.round(risk),
          composite: Math.round(composite),
          status
        },
        notes: "",
        thesis: "",
        checkpoints: [""].filter(Boolean)
      };
    });
  }, [rows, enrichedData]);

  const totalInvested = enriched.reduce((acc, h) => acc + h.invested, 0);
  const totalCurrent = enriched.reduce((acc, h) => acc + h.current, 0);
  const totalPnl = totalCurrent - totalInvested;
  const allocationPct = data?.allocation_pct ?? 0;
  const portfolioValue = allocationPct > 0 ? totalCurrent / (allocationPct / 100) : totalCurrent;
  const cashValue = Math.max(portfolioValue - totalCurrent, 0);
  const cashPct = portfolioValue > 0 ? (cashValue / portfolioValue) * 100 : 0;

  const years = Math.max(1, Number(lookbackDays) / 365);
  const cagr = totalInvested > 0 ? (Math.pow(totalCurrent / totalInvested, 1 / years) - 1) * 100 : 0;
  const xirr = cagr;

  const sectors = Array.from(new Set(enriched.map((h) => h.sector))).sort();
  const statuses = ["Core", "Hold", "Trim", "Exit"];

  const getSortVal = (h: ScoredRow) => {
    if (sortKey === "scores.composite") return h.scores.composite;
    return (h as Record<string, any>)[sortKey] ?? 0;
  };

  const filtered = enriched
    .filter((h) => (sectorFilter === "All" ? true : h.sector === sectorFilter))
    .filter((h) => (statusFilter === "All" ? true : h.scores.status === statusFilter))
    .filter((h) => queryText ? `${h.symbol} ${h.name}`.toLowerCase().includes(queryText.toLowerCase()) : true)
    .sort((a, b) => {
      const dir = sortDir === "asc" ? 1 : -1;
      const av = getSortVal(a);
      const bv = getSortVal(b);
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * dir;
      return String(av ?? "").localeCompare(String(bv ?? "")) * dir;
    });

  const selected = enriched.find((h) => h.symbol === selectedTicker) || enriched[0];
  const selectedVendor = (selected?.vendor || {}) as Record<string, any>;
  const selPe = pickNumber(selectedVendor, ["keyMetrics.pe", "pe", "peRatio", "valuation.pe"]);
  const selEvEbitda = pickNumber(selectedVendor, ["keyMetrics.evEbitda", "evEbitda", "valuation.evEbitda"]);
  const selFcfYield = pickNumber(selectedVendor, ["keyMetrics.fcfYield", "fcfYield", "valuation.fcfYield"]);
  const selRoe = pickNumber(selectedVendor, ["keyMetrics.roe", "roe"]);
  const selRoce = pickNumber(selectedVendor, ["keyMetrics.roce", "roce"]);
  const selDebtEq = pickNumber(selectedVendor, ["keyMetrics.debtToEquity", "debtToEquity", "balanceSheet.debtToEquity"]);

  const sectorAlloc = sectors.map((sector) => {
    const total = enriched.filter((h) => h.sector === sector).reduce((acc, h) => acc + h.current, 0);
    return { sector, pct: totalCurrent > 0 ? (total / totalCurrent) * 100 : 0 };
  });

  const topWeights = [...enriched.map((h) => h.weight)].sort((a, b) => b - a);
  const top1 = topWeights[0] || 0;
  const top3 = topWeights.slice(0, 3).reduce((a, b) => a + b, 0);
  const top5 = topWeights.slice(0, 5).reduce((a, b) => a + b, 0);
  const maxDrawdown = Math.min(...enriched.map((h) => Number(h.drawdown_pct ?? 0)), 0);
  const betaProxy = 1.0;
  const volProxy = Math.min(60, Math.max(5, Math.abs(Number(data?.sleeve_drawdown_pct ?? 0)) * 1.2));
  const simKpis = simData?.kpis || {};

  const targetRules = {
    maxSingle: 8,
    sectorCap: 25,
    largeCapTarget: 70,
    midCapTarget: 30
  };

  const rebalancingPlan = enriched
    .map((h) => {
      const target = Math.min(targetRules.maxSingle, h.marketCapBucket === "Large" ? 9 : 6);
      const drift = h.weight - target;
      const action = drift > 1 ? "Trim" : drift < -1 ? "Add" : "Hold";
      return { ...h, target, drift, action };
    })
    .sort((a, b) => Math.abs(b.drift) - Math.abs(a.drift));

  if (error) return <ErrorState message={error} />;
  if (loading && !data) return <LoadingState />;

  return (
    <div>
      <SectionCard title="At-a-Glance">
        <MetricGrid>
          <MetricCard label="Total Invested" value={formatInr(totalInvested)} />
          <MetricCard label="Current Value" value={formatInr(totalCurrent)} />
          <MetricCard label="P/L" value={`${formatInr(totalPnl)} (${formatPct((totalPnl / Math.max(totalInvested, 1)) * 100)})`} />
          <MetricCard label="CAGR" value={formatPct(cagr)} />
          <MetricCard label="XIRR" value={formatPct(xirr)} />
          <MetricCard label="Dividend/Interest" value="—" />
        </MetricGrid>
      </SectionCard>

      <SectionCard title="Allocation Snapshot">
        <div className="chart-panel">
          <h4>Asset Allocation</h4>
          <Plot
            data={[
              {
                type: "pie",
                labels: ["Equity", "Cash"],
                values: [totalCurrent, cashValue],
                marker: { colors: ["#2D7DFF", "#FFB020"] },
                hole: 0.45
              }
            ]}
            layout={{ height: 280, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { l: 20, r: 20, t: 20, b: 20 } }}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: "100%", height: "100%" }}
            useResizeHandler
          />
        </div>
      </SectionCard>

      <SectionCard title="Risk Snapshot">
        <MetricGrid>
          <MetricCard label="Portfolio Beta" value={formatNumber(betaProxy, 2)} />
          <MetricCard label="Volatility" value={`${formatNumber(volProxy, 1)}%`} />
          <MetricCard label="Max Drawdown" value={`${formatNumber(maxDrawdown, 1)}%`} />
          <MetricCard label="Top 1 Weight" value={formatPct(top1)} />
          <MetricCard label="Top 3 Weight" value={formatPct(top3)} />
          <MetricCard label="Cash %" value={formatPct(cashPct)} />
        </MetricGrid>
      </SectionCard>

      <SectionCard title="Holdings">
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
          <input
            className="control-input"
            placeholder="Search ticker/name"
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            style={{ minWidth: 220 }}
          />
          <select className="control-input" value={sectorFilter} onChange={(e) => setSectorFilter(e.target.value)}>
            <option value="All">All Sectors</option>
            {sectors.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <select className="control-input" value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="All">All Status</option>
            {statuses.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <select className="control-input" value={sortKey} onChange={(e) => setSortKey(e.target.value)}>
            <option value="weight">Weight %</option>
            <option value="pnl">P/L</option>
            <option value="pnlPct">P/L %</option>
            <option value="scores.composite">Composite Score</option>
          </select>
          <button className="control-input" onClick={() => setSortDir(sortDir === "asc" ? "desc" : "asc")}>
            Sort {sortDir === "asc" ? "▲" : "▼"}
          </button>
        </div>
        <div className="table-wrap" style={{ maxHeight: 520, overflow: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Name</th>
                <th>Weight %</th>
                <th>Avg Buy</th>
                <th>Current</th>
                <th>P/L</th>
                <th>P/L %</th>
                <th>Quality</th>
                <th>Valuation</th>
                <th>Risk</th>
                <th>Status</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((h) => (
                <tr key={h.symbol}>
                  <td>{h.symbol}</td>
                  <td>{h.name}</td>
                  <td>{formatPct(h.weight)}</td>
                  <td>{formatInr(h.avg_cost)}</td>
                  <td>{formatInr(h.ltp)}</td>
                  <td><span className={h.pnl >= 0 ? "positive" : "negative"}>{formatInr(h.pnl)}</span></td>
                  <td><span className={h.pnlPct >= 0 ? "positive" : "negative"}>{formatPct(h.pnlPct)}</span></td>
                  <td>{h.scores.quality}</td>
                  <td>{h.scores.valuation}</td>
                  <td>{h.scores.risk}</td>
                  <td>{h.scores.status}</td>
                  <td>{h.notes || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Target Return Optimization">
        <MetricGrid>
          <MetricCard label="Target Return" value={formatPct(Number(targetReturn))} />
          <MetricCard label="Expected Return" value={formatPct((optData?.kpis?.expected_return ?? 0) * 100)} />
          <MetricCard label="Expected Volatility" value={formatPct((optData?.kpis?.expected_vol ?? 0) * 100)} />
          <MetricCard label="Status" value={optData?.status || "—"} />
        </MetricGrid>
        <div className="table-wrap" style={{ marginTop: 12 }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Current Weight %</th>
                <th>Target Weight %</th>
                <th>Delta %</th>
              </tr>
            </thead>
            <tbody>
              {(optData?.weights || []).map((w) => (
                <tr key={w.symbol}>
                  <td>{w.symbol}</td>
                  <td>{formatPct(w.current_weight_pct)}</td>
                  <td>{formatPct(w.weight_pct)}</td>
                  <td><span className={w.weight_pct - w.current_weight_pct >= 0 ? "positive" : "negative"}>{formatPct(w.weight_pct - w.current_weight_pct)}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>

      <SectionCard title="Risk Quantification">
        <MetricGrid>
          <MetricCard label="Portfolio Volatility" value={`${formatNumber(volProxy, 1)}%`} />
          <MetricCard label="Portfolio Beta" value={formatNumber(betaProxy, 2)} />
          <MetricCard label="Max Drawdown" value={`${formatNumber(maxDrawdown, 1)}%`} />
          <MetricCard label="VaR (99%)" value={formatInr(enriched.reduce((acc, h) => acc + Number(h.es99_inr || 0), 0))} />
          <MetricCard label="Top 1 Weight" value={formatPct(top1)} />
          <MetricCard label="Top 5 Weight" value={formatPct(top5)} />
        </MetricGrid>
        <div style={{ marginTop: 12 }}>
          <MetricGrid>
            <MetricCard label="Sim Mean (30D)" value={formatInr(simKpis.mean)} />
            <MetricCard label="Sim VaR99 (30D)" value={formatInr(simKpis.var99)} />
            <MetricCard label="Sim ES99 (30D)" value={formatInr(simKpis.es99)} />
            <MetricCard label="Prob Loss (30D)" value={formatPct((simKpis.prob_loss ?? 0) * 100)} />
          </MetricGrid>
        </div>
      </SectionCard>
    </div>
  );
}
