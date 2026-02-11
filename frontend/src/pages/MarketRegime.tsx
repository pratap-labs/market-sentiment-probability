import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import Plot from "../components/Plot";
import { formatPct, formatInr } from "../components/format";
import { useCachedApi } from "../hooks/useCachedApi";
import { useControls } from "../state/ControlsContext";

type MarketRegimeResponse = {
  regime: Record<string, unknown>;
  iv_rv_series?: { date: string; iv: number; rv: number }[];
  pcr_series?: { date: string; pcr_oi: number }[];
  expiry_list?: string[];
  expiry_filter?: string;
};

type Ohlcv = { rows: Record<string, unknown>[] };

export default function MarketRegime() {
  const [expiryFilter, setExpiryFilter] = useState<string>("all");
  const { setControls } = useControls();
  const expiryParam = encodeURIComponent(expiryFilter || "all");
  const regime = useCachedApi<MarketRegimeResponse>(
    `market_regime:${expiryParam}`,
    `/market-regime?expiry=${expiryParam}`,
    60_000
  );
  const ohlcv = useCachedApi<Ohlcv>("nifty_ohlcv", "/derivatives/nifty-ohlcv?limit=520", 60_000);

  const r = regime.data?.regime || {};
  const expiryList = regime.data?.expiry_list || [];

  useEffect(() => {
    if (expiryFilter !== "all" && expiryList.length && !expiryList.includes(expiryFilter)) {
      setExpiryFilter("all");
    }
  }, [expiryFilter, expiryList]);

  useEffect(() => {
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">PCR Expiry</span>
          <select className="control-input" value={expiryFilter} onChange={(e) => setExpiryFilter(e.target.value)}>
            <option value="all">All Expiries</option>
            {expiryList.map((exp) => (
              <option key={exp} value={exp}>{exp}</option>
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
            <span className="controls-summary-key">Expiry</span> {expiryFilter}
          </span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, expiryFilter, expiryList]);

  const volSeries = useMemo(() => {
    const rows = ohlcv.data?.rows || [];
    const closes = rows.map((r) => Number(r.close || 0));
    const dates = rows.map((r) => String(r.date || ""));
    const returns: number[] = [];
    for (let i = 1; i < closes.length; i++) {
      const prev = closes[i - 1] || 0;
      const curr = closes[i] || 0;
      returns.push(prev ? (curr - prev) / prev : 0);
    }
    const window = 30;
    const vol: number[] = [];
    const x: string[] = [];
    for (let i = window; i < returns.length; i++) {
      const slice = returns.slice(i - window, i);
      const mean = slice.reduce((a, b) => a + b, 0) / window;
      const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / window;
      const sigma = Math.sqrt(variance) * Math.sqrt(252) * 100;
      vol.push(sigma);
      x.push(dates[i + 1] || "");
    }
    const meanVol = vol.reduce((a, b) => a + b, 0) / (vol.length || 1);
    const stdVol = Math.sqrt(vol.reduce((a, b) => a + (b - meanVol) ** 2, 0) / (vol.length || 1));
    const current = vol[vol.length - 1] || 0;
    return { x, vol, mean: meanVol, upper: meanVol + stdVol, lower: Math.max(0, meanVol - stdVol), current };
  }, [ohlcv]);

  const guidance = useMemo(() => {
    const current = volSeries.current || 0;
    if (!current) return { regime: "—", strategy: "—", note: "" };
    if (current < volSeries.lower) {
      return {
        regime: "LOW VOLATILITY",
        strategy: "Gamma Scalping",
        note: "Volatility compressed; consider long gamma with tight risk."
      };
    }
    if (current > volSeries.upper) {
      return {
        regime: "HIGH VOLATILITY",
        strategy: "Short Vol (Premium Selling)",
        note: "Vol elevated; favor defined-risk premium selling with hedges."
      };
    }
    return {
      regime: "MEDIUM VOLATILITY",
      strategy: "Selective Short Vol",
      note: "Vol in normal range; prefer balanced premium selling."
    };
  }, [volSeries]);

  return (
    <>
      {regime.error || ohlcv.error ? <ErrorState message={String(regime.error || ohlcv.error)} /> : null}

      <SectionCard title="Regime Summary">
        {regime.loading || !regime.data ? <LoadingState /> : (
          <MetricGrid>
            <MetricCard
              label="Regime"
              value={String(r["market_regime"] || "—")}
              tooltip="Overall market volatility regime derived from realized volatility."
            />
            <MetricCard
              label="Suggested Strategy"
              value={guidance.strategy}
            />
            <MetricCard
              label="IV Rank (90d)"
              value={`${Number(r["iv_rank"] || 0).toFixed(0)}%`}
              tooltip={`**IV Rank** = Where current IV sits in 90-day range (0-100%)\n\nFormula: (Current IV - Min IV) / (Max IV - Min IV) × 100\n\n**HIGH IV RANK** 🔴 (>70):\n• Example: Current IV = 18%, Min = 12%, Max = 20%\n• IV Rank = (18-12)/(20-12) × 100 = 75%\n• Options are EXPENSIVE (near 90-day highs)\n• ✅ SELL premium: Iron condors, credit spreads, strangles\n\n**LOW IV RANK** 🟢 (<30):\n• Example: Current IV = 13%, Min = 12%, Max = 20%\n• IV Rank = (13-12)/(20-12) × 100 = 12.5%\n• Options are CHEAP (near 90-day lows)\n• ✅ BUY premium: Long straddles, debit spreads, calendars\n\n**MID RANGE** 🟡 (30-70): Fair value, neutral strategies`}
            />
            <MetricCard
              label="PCR (OI)"
              value={Number(r["pcr_oi"] || 0).toFixed(2)}
              tooltip={`**Put-Call Ratio** = Total Put OI / Total Call OI\n\n**BULLISH** 🟢 (PCR > 1.2):\n• Example: Put OI = 60L, Call OI = 45L → PCR = 1.33\n• More puts than calls = hedging/protection buying\n• Traders expect up move, buying puts to protect profits\n• ✅ Good for: Selling puts, Bull spreads\n\n**BEARISH** 🔴 (PCR < 0.8):\n• Example: Put OI = 35L, Call OI = 50L → PCR = 0.70\n• More calls than puts = excessive optimism\n• Everyone chasing upside, no protection\n• ⚠️ Warning: Complacent market, consider bear spreads\n\n**NEUTRAL** 🟡 (PCR 0.8-1.2): Balanced sentiment, no extreme positioning`}
            />
            <MetricCard
              label="Skew"
              value={formatPct((Number(r["skew"]) || 0) * 100)}
              tooltip={`**Volatility Skew = OTM Put IV - OTM Call IV (both ~3% away from spot)\n**Example: NIFTY @ 25,000**\n\n**HIGH SKEW (Fear)** 🔴:\n• 24,250 Put (3% OTM): IV = 20%\n• 25,750 Call (3% OTM): IV = 15%\n• Skew = +5% → FEAR mode! Puts are expensive\n• ❌ Don't buy OTM puts (overpriced protection)\n• ✅ Sell put spreads (collect rich premium)\n\n**LOW/NEGATIVE SKEW (Complacent)** 🟢:\n• 24,250 Put: IV = 14%\n• 25,750 Call: IV = 18%\n• Skew = -4% → Excessive optimism! Calls expensive\n• ⚠️ Warning sign - market too complacent\n• ✅ Buy put protection (cheap insurance)\n\n**BALANCED** 🟡:\n• Put IV ≈ Call IV (within ±1%)\n• Fair pricing, no extreme sentiment`}
            />
            <MetricCard
              label="Term Structure"
              value={formatPct((Number(r["term_structure"]) || 0) * 100)}
              tooltip={`**Term Structure** = How IV changes across expiries (Far IV - Near IV).\n\n**CONTANGO** (Positive, Normal):\n• Nov expiry: IV = 15%, Dec expiry: IV = 18% → Term Structure = +3%\n• Far options MORE expensive (more time = more uncertainty)\n• ✅ Good for: Calendar spreads (sell Nov, buy Dec)\n• Normal healthy market\n\n**BACKWARDATION** (Negative, Stress):\n• Nov expiry: IV = 22%, Dec expiry: IV = 16% → Term Structure = -6%\n• Near options MORE expensive (immediate fear/event)\n• ⚠️ Warning: Avoid selling near-term, market expects short-term turbulence\n• Happens before: RBI policy, Budget, crashes`}
            />
            <MetricCard
              label="Spot"
              value={formatInr(r["current_spot"])}
              tooltip="Latest NIFTY spot from cache."
            />
            <MetricCard
              label="Mean Volatility"
              value={formatPct(volSeries.mean)}
              tooltip="Mean realized volatility over the selected lookback."
            />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Volatility Over Time">
        {ohlcv.loading || !ohlcv.data ? <LoadingState /> : (
          <Plot
            data={[
              { type: "scatter", mode: "lines", x: volSeries.x, y: volSeries.vol, name: "Vol" },
              { type: "scatter", mode: "lines", x: volSeries.x, y: volSeries.x.map(() => volSeries.mean), name: "Mean", line: { dash: "dash", color: "#6CC0FF" } },
              { type: "scatter", mode: "lines", x: volSeries.x, y: volSeries.x.map(() => volSeries.upper), name: "+1σ", line: { dash: "dot" } },
              { type: "scatter", mode: "lines", x: volSeries.x, y: volSeries.x.map(() => volSeries.lower), name: "-1σ", line: { dash: "dot" } }
            ]}
            layout={{
              height: 380,
              paper_bgcolor: "rgba(0,0,0,0)",
              plot_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 30, r: 12, t: 10, b: 36 },
              xaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
              yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } }
            }}
            config={{ displayModeBar: false, responsive: true }}
            useResizeHandler
            style={{ width: "100%" }}
          />
        )}
      </SectionCard>

      <div className="chart-grid-1x2">
        <SectionCard title="IV vs RV">
          {regime.loading || !regime.data ? <LoadingState /> : (
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "lines",
                  x: (regime.data.iv_rv_series || []).map((d) => d.date),
                  y: (regime.data.iv_rv_series || []).map((d) => d.iv),
                  name: "Implied Vol",
                  line: { color: "#2D7DFF", width: 2 }
                },
                {
                  type: "scatter",
                  mode: "lines",
                  x: (regime.data.iv_rv_series || []).map((d) => d.date),
                  y: (regime.data.iv_rv_series || []).map((d) => d.rv),
                  name: "Realized Vol",
                  line: { color: "#FFB020", width: 2 }
                }
              ]}
              layout={{
                height: 260,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 30, r: 12, t: 10, b: 36 },
                xaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } },
                yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } }
              }}
              config={{ displayModeBar: false, responsive: true }}
              useResizeHandler
              style={{ width: "100%" }}
            />
          )}
        </SectionCard>

        <SectionCard title="PCR Over Time">
          {regime.loading || !regime.data ? <LoadingState /> : (
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "lines",
                  x: (regime.data.pcr_series || []).map((d) => d.date),
                  y: (regime.data.pcr_series || []).map((d) => d.pcr_oi),
                  name: "PCR (OI)",
                  line: { color: "#2ecc71", width: 2 }
                }
              ]}
              layout={{
                height: 260,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { l: 30, r: 12, t: 10, b: 36 },
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

      {/* Strategy guidance now merged into Regime Summary */}
    </>
  );
}
