import { useEffect, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import { useCachedApi } from "../hooks/useCachedApi";
import { apiFetch } from "../api/client";

export default function LongTerm() {
  const { data: ohlcvData } = useCachedApi<{ rows: Record<string, unknown>[]; count?: number }>(
    "long_term_ohlcv_latest",
    "/long-term/ohlcv-latest?days=30&limit=250&use_cache=true",
    0
  );
  const { data: momentumData } = useCachedApi<{ rows: Record<string, unknown>[]; count?: number }>(
    "long_term_momentum_scores",
    "/long-term/momentum-scores?lookback_days=320&limit=250&use_cache=true",
    0
  );
  const { data: earningsData } = useCachedApi<{ rows: Record<string, unknown>[]; count?: number }>(
    "long_term_earnings_scores",
    "/long-term/earnings-scores?limit=250",
    0
  );
  const { data: qualityData } = useCachedApi<{ rows: Record<string, unknown>[]; count?: number }>(
    "long_term_quality_scores",
    "/long-term/quality-scores?limit=250",
    0
  );
  const { data: finalCompositeData } = useCachedApi<{ rows: Record<string, unknown>[]; count?: number }>(
    "long_term_final_composite_scores",
    "/long-term/final-composite-scores?limit=250",
    0
  );
  const [fundamentalsData, setFundamentalsData] = useState<{ rows: Record<string, unknown>[]; count?: number } | null>(null);
  const [fundamentalsError, setFundamentalsError] = useState<string | null>(null);
  const [fundamentalsLoading, setFundamentalsLoading] = useState<boolean>(true);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");
  const [detailJson, setDetailJson] = useState<Record<string, unknown> | null>(null);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<string | null>(null);


  useEffect(() => {
    let mounted = true;
    setFundamentalsLoading(true);
    apiFetch("/long-term/fundamentals")
      .then((payload) => {
        if (!mounted) return;
        setFundamentalsData(payload as { rows: Record<string, unknown>[]; count?: number });
        setFundamentalsError(null);
      })
      .catch((err) => {
        if (!mounted) return;
        setFundamentalsError(String(err));
      })
      .finally(() => {
        if (!mounted) return;
        setFundamentalsLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <>
      <SectionCard title="Universe OHLCV (Latest, Sorted by Volume)">
        {!ohlcvData ? (
          <LoadingState />
        ) : (
          <>
            <div style={{ marginBottom: 12, color: "#b9c4d6" }}>
              Rows: {ohlcvData.count ?? ohlcvData.rows.length}
            </div>
            <DataTable
              columns={[
                { key: "symbol", label: "Symbol" },
                { key: "date", label: "Date" },
                { key: "open", label: "Open" },
                { key: "high", label: "High" },
                { key: "low", label: "Low" },
                { key: "close", label: "Close" },
                { key: "volume", label: "Volume" },
              ]}
              rows={ohlcvData.rows || []}
              maxHeight={640}
            />
          </>
        )}
      </SectionCard>
      <SectionCard title="Universe Fundamentals (Debug)">
        {fundamentalsLoading ? (
          <LoadingState />
        ) : fundamentalsError ? (
          <ErrorState message={fundamentalsError} />
        ) : (
          <>
            <div style={{ marginBottom: 12, color: "#b9c4d6" }}>
              Rows: {fundamentalsData?.count ?? fundamentalsData?.rows?.length ?? 0}
            </div>
            <DataTable
              columns={[
                { key: "symbol", label: "Symbol" },
                { key: "company", label: "Company" },
                { key: "industry", label: "Industry" },
                { key: "market_cap_fmt", label: "Market Cap (Cr)", format: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
                { key: "eps", label: "EPS" },
                { key: "pe", label: "PE" },
                { key: "roe", label: "ROE" },
                { key: "roce", label: "ROCE" },
                { key: "revenue_fmt", label: "Revenue (Cr)", format: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
                { key: "net_income_fmt", label: "Net Income (Cr)", format: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
                { key: "DilutedNormalizedEPS", label: "DilutedNormalizedEPS" },
                { key: "DilutedEPSExcludingExtraOrdItems", label: "DilutedEPSExclExtraOrd" },
                { key: "NetIncome", label: "NetIncome" },
                { key: "TotalRevenue", label: "TotalRevenue" },
                { key: "OperatingIncome", label: "OperatingIncome" },
                { key: "ePSChangePercentTTMOverTTM", label: "EPSChangeTTM" },
                { key: "revenueChangePercentTTMPOverTTM", label: "RevenueChangeTTM" },
                { key: "ePSGrowthRate5Year", label: "EPSGrowth5Y" },
                { key: "returnOnAverageEquityMostRecentFiscalYear", label: "ROE (FY)" },
                { key: "returnOnInvestmentMostRecentFiscalYear", label: "ROI (FY)" },
                { key: "operatingMarginTrailing12Month", label: "OpMargin TTM" },
                { key: "netProfitMarginPercentTrailing12Month", label: "NetMargin TTM" },
                { key: "TotalDebt", label: "TotalDebt" },
                { key: "TotalEquity", label: "TotalEquity" },
                { key: "ltDebtPerEquityMostRecentFiscalYear", label: "LT Debt/Equity" },
                { key: "netInterestCoverageMostRecentFiscalYear", label: "NetInterestCov" },
                { key: "CashfromOperatingActivities", label: "CFO" },
                { key: "freeCashFlowMostRecentFiscalYear", label: "FCF (FY)" },
                { key: "pPerEBasicExcludingExtraordinaryItemsTTM", label: "PE Basic TTM" },
                { key: "pegRatio", label: "PEG" },
                { key: "priceToSalesTrailing12Month", label: "Price/Sales TTM" },
                { key: "priceToBookMostRecentFiscalYear", label: "Price/Book FY" },
                {
                  key: "action",
                  label: "Action",
                  render: (row) => {
                    const sym = String(row.symbol || "");
                    return (
                      <div style={{ display: "flex", gap: 8 }}>
                        <button
                          className="control-input"
                          onClick={() => {
                            setSelectedSymbol(sym);
                            setDetailLoading(true);
                            setDetailError(null);
                            apiFetch(`/long-term/fundamentals-cache?symbol=${encodeURIComponent(sym)}`)
                              .then((payload) => {
                                const data = (payload as Record<string, unknown>).data as Record<string, unknown> | null;
                                setDetailJson(data || null);
                              })
                              .catch((err) => setDetailError(String(err)))
                              .finally(() => setDetailLoading(false));
                          }}
                        >
                          Show Details
                        </button>
                        <button
                          className="control-input"
                          onClick={() => {
                            setSelectedSymbol(sym);
                            setDetailLoading(true);
                            setDetailError(null);
                            apiFetch(`/long-term/fundamentals-refresh?symbol=${encodeURIComponent(sym)}`, { method: "POST" })
                              .then((payload) => {
                                const data = (payload as Record<string, unknown>).data as Record<string, unknown> | null;
                                setDetailJson(data || null);
                              })
                              .catch((err) => setDetailError(String(err)))
                              .finally(() => setDetailLoading(false));
                          }}
                        >
                          Fetch Latest
                        </button>
                      </div>
                    );
                  }
                }
              ]}
              rows={fundamentalsData?.rows || []}
              maxHeight={400}
            />
          </>
        )}
      </SectionCard>
      <SectionCard title="Momentum Strategy Scores">
        {!momentumData ? (
          <LoadingState />
        ) : (
          <>
            <div style={{ marginBottom: 12, color: "#b9c4d6" }}>
              Rows: {momentumData.count ?? momentumData.rows.length}
            </div>
            <DataTable
              columns={[
                { key: "symbol", label: "Symbol" },
                { key: "close", label: "Close", format: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
                { key: "momentum_1m_pct", label: "1M Mom %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "momentum_3m_pct", label: "3M Mom %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "momentum_6m_pct", label: "6M Mom %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "trend_strength_pct", label: "Trend % (50/200)", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "breakout_55d_pct", label: "55D Breakout %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "risk_adj_3m", label: "Risk-Adj 3M", format: (v) => (v == null ? "—" : Number(v).toFixed(3)) },
                { key: "score_1m", label: "Score 1M", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_3m", label: "Score 3M", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_6m", label: "Score 6M", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_trend", label: "Score Trend", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_breakout", label: "Score Breakout", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_risk_adj", label: "Score RiskAdj", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "composite_score", label: "Composite Score", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
              ]}
              rows={momentumData.rows || []}
              maxHeight={640}
            />
          </>
        )}
      </SectionCard>
      <SectionCard title="Earnings Acceleration Scores (*)">
        {!earningsData ? (
          <LoadingState />
        ) : (
          <>
            <div style={{ marginBottom: 12, color: "#b9c4d6" }}>
              Rows: {earningsData.count ?? earningsData.rows.length}
            </div>
            <DataTable
              columns={[
                { key: "symbol", label: "Symbol" },
                { key: "eps_qoq_yoy_pct", label: "EPS QoQ YoY %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "eps_ttm_growth_pct", label: "EPS TTM Growth %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "revenue_qoq_yoy_pct", label: "Revenue QoQ YoY %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "operating_margin_trend_pct", label: "Op Margin Trend %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "score_eps_qoq_yoy", label: "Score EPS QoQ (40%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_eps_ttm", label: "Score EPS TTM (30%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_revenue_qoq_yoy", label: "Score Rev QoQ (20%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_op_margin_trend", label: "Score OpMargin (10%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "coverage_weight", label: "Coverage %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(0)}%`) },
                { key: "earnings_acceleration_score", label: "Composite Score*", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
              ]}
              rows={earningsData.rows || []}
              maxHeight={640}
            />
            <div style={{ marginTop: 8, color: "#8fa1b8", fontSize: 12 }}>
              * Coverage-normalized score (weights rescaled by available metrics).
            </div>
          </>
        )}
      </SectionCard>
      <SectionCard title="Quality Scores (*)">
        {!qualityData ? (
          <LoadingState />
        ) : (
          <>
            <div style={{ marginBottom: 12, color: "#b9c4d6" }}>
              Rows: {qualityData.count ?? qualityData.rows.length}
            </div>
            <DataTable
              columns={[
                { key: "symbol", label: "Symbol" },
                { key: "roe_pct", label: "ROE %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "roce_pct", label: "ROCE %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "debt_to_equity", label: "Debt/Equity", format: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
                { key: "interest_coverage", label: "Interest Coverage", format: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
                { key: "operating_margin_pct", label: "Operating Margin %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "net_profit_margin_pct", label: "Net Profit Margin %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(2)}%`) },
                { key: "beta", label: "Beta", format: (v) => (v == null ? "—" : Number(v).toFixed(2)) },
                { key: "score_roe", label: "Score ROE (30%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_roce", label: "Score ROCE (20%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_debt_to_equity", label: "Score D/E Inv (20%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_interest_coverage", label: "Score IntCov (20%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "score_operating_margin", label: "Score OpMargin (10%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "coverage_weight", label: "Coverage %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(0)}%`) },
                { key: "quality_score", label: "Quality Score*", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
              ]}
              rows={qualityData.rows || []}
              maxHeight={640}
            />
            <div style={{ marginTop: 8, color: "#8fa1b8", fontSize: 12 }}>
              * Coverage-normalized score (weights rescaled by available metrics).
            </div>
          </>
        )}
      </SectionCard>
      <SectionCard title="Final Composite Scores (55/30/15) (*)">
        {!finalCompositeData ? (
          <LoadingState />
        ) : (
          <>
            <div style={{ marginBottom: 12, color: "#b9c4d6" }}>
              Rows: {finalCompositeData.count ?? finalCompositeData.rows.length}
            </div>
            <DataTable
              columns={[
                { key: "symbol", label: "Symbol" },
                { key: "momentum_composite_score", label: "Momentum (55%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "earnings_acceleration_score", label: "Earnings (30%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "quality_score", label: "Quality (15%)", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
                { key: "coverage_weight", label: "Coverage %", format: (v) => (v == null ? "—" : `${Number(v).toFixed(0)}%`) },
                { key: "final_composite_score", label: "Final Composite*", format: (v) => (v == null ? "—" : Number(v).toFixed(1)) },
              ]}
              rows={finalCompositeData.rows || []}
              maxHeight={640}
            />
            <div style={{ marginTop: 8, color: "#8fa1b8", fontSize: 12 }}>
              * Coverage-normalized score (missing model weights are renormalized).
            </div>
          </>
        )}
      </SectionCard>
    </>
  );
}
