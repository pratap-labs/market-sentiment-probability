import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import { useCachedApi } from "../hooks/useCachedApi";
import { apiFetch } from "../api/client";

export default function LongTerm() {
  const { error } = useCachedApi<{ rows: Record<string, unknown>[]; count?: number; shown?: number }>(
    "long_term_instruments",
    "/long-term/instruments?limit=250&instrument_type=EQ&exchange=NSE",
    300_000
  );
  const { data: universeData } = useCachedApi<{
    nifty100?: { matched?: Record<string, unknown>[]; missing?: string[]; count?: number };
    midcap150?: { matched?: Record<string, unknown>[]; missing?: string[]; count?: number };
  }>(
    "long_term_universe_match",
    "/long-term/universe-match",
    300_000
  );
  const { data: ohlcvData } = useCachedApi<{ rows: Record<string, unknown>[]; count?: number }>(
    "long_term_ohlcv_latest",
    "/long-term/ohlcv-latest?days=30&limit=250",
    300_000
  );
  const [fundamentalsData, setFundamentalsData] = useState<{ rows: Record<string, unknown>[]; count?: number } | null>(null);
  const [fundamentalsError, setFundamentalsError] = useState<string | null>(null);
  const [fundamentalsLoading, setFundamentalsLoading] = useState<boolean>(true);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("");
  const [detailJson, setDetailJson] = useState<Record<string, unknown> | null>(null);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const matchedRows = useMemo(() => {
    const rows: Record<string, unknown>[] = [];
    (universeData?.nifty100?.matched || []).forEach((m) => {
      rows.push({ symbol: (m as Record<string, unknown>).tradingsymbol, group: "NIFTY100" });
    });
    (universeData?.midcap150?.matched || []).forEach((m) => {
      rows.push({ symbol: (m as Record<string, unknown>).tradingsymbol, group: "MIDCAP150" });
    });
    return rows;
  }, [universeData]);


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
      {error ? <ErrorState message={error} /> : null}
      <SectionCard title="Universe Match (NIFTY100 + MIDCAP150)">
        {!universeData ? (
          <LoadingState />
        ) : (
          <>
            <div style={{ marginBottom: 12, color: "#b9c4d6" }}>
              NIFTY100 missing: {universeData.nifty100?.missing?.length ?? 0}
              {" "} | MIDCAP150 missing: {universeData.midcap150?.missing?.length ?? 0}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16, marginBottom: 16 }}>
              <div className="chart-panel">
                <h4>NIFTY100 Matched</h4>
                <div style={{ maxHeight: 220, overflow: "auto", color: "#b9c4d6" }}>
                  {(universeData.nifty100?.matched || []).map((m) => (
                    <div key={String((m as Record<string, unknown>).tradingsymbol || (m as Record<string, unknown>).symbol || (m as Record<string, unknown>).name || Math.random())}>
                      {String((m as Record<string, unknown>).tradingsymbol || (m as Record<string, unknown>).symbol || (m as Record<string, unknown>).name || "—")}
                    </div>
                  ))}
                </div>
              </div>
              <div className="chart-panel">
                <h4>MIDCAP150 Matched</h4>
                <div style={{ maxHeight: 220, overflow: "auto", color: "#b9c4d6" }}>
                  {(universeData.midcap150?.matched || []).map((m) => (
                    <div key={String((m as Record<string, unknown>).tradingsymbol || (m as Record<string, unknown>).symbol || (m as Record<string, unknown>).name || Math.random())}>
                      {String((m as Record<string, unknown>).tradingsymbol || (m as Record<string, unknown>).symbol || (m as Record<string, unknown>).name || "—")}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
              <div className="chart-panel">
                <h4>NIFTY100 Missing</h4>
                <div style={{ maxHeight: 220, overflow: "auto", color: "#b9c4d6" }}>
                  {(universeData.nifty100?.missing || []).map((m) => (
                    <div key={m}>{m}</div>
                  ))}
                </div>
              </div>
              <div className="chart-panel">
                <h4>MIDCAP150 Missing</h4>
                <div style={{ maxHeight: 220, overflow: "auto", color: "#b9c4d6" }}>
                  {(universeData.midcap150?.missing || []).map((m) => (
                    <div key={m}>{m}</div>
                  ))}
                </div>
              </div>
            </div>
          </>
        )}
      </SectionCard>
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
              maxHeight={640}
            />
          </>
        )}
      </SectionCard>
    </>
  );
}
