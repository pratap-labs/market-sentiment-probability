import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import MetricGrid from "../components/MetricGrid";
import MetricCard from "../components/MetricCard";
import DataTable from "../components/DataTable";
import Plot from "../components/Plot";
import { useCachedApi } from "../hooks/useCachedApi";
import { formatInr, formatPct, formatNumber } from "../components/format";
import { useControls } from "../state/ControlsContext";
import { usePortfolio } from "../state/PortfolioContext";
import { apiFetch } from "../api/client";
import RiskBucketsPortfolio, { type PortfolioRB } from "./RiskBucketsPortfolio";

type PreTrade = {
  spot: number;
  capital: number;
  greeks: Record<string, number>;
  normalized: Record<string, number>;
  iv_regime: string;
  zone: { num: number; name: string; color: string; message: string };
  comparison_table: Record<string, unknown>[];
  bucket_probs: Record<string, number>;
  bucket_rows: Record<string, unknown>[];
  threshold_rows: Record<string, unknown>[];
  repriced_rows: Record<string, unknown>[];
  scenario_table: Record<string, unknown>[];
};

type OptionsChainRow = {
  strike: number;
  call_ltp: number | null;
  call_oi: number | null;
  call_symbol: string;
  call_token?: number | null;
  put_ltp: number | null;
  put_oi: number | null;
  put_symbol: string;
  put_token?: number | null;
};

type OptionsChain = {
  expiry: string | null;
  expiry_list: string[];
  spot: number | null;
  date: string | null;
  rows: OptionsChainRow[];
};

type VirtualLeg = {
  id: string;
  tradingsymbol: string;
  expiry: string;
  strike: number | string;
  type: string;
  qty: number;
  price: number;
  source: "position" | "chain";
  instrument_token?: number | null;
  enabled: boolean;
};

type VirtualGreeksResponse = {
  greeks: Record<string, number>;
  current_spot: number;
};

export default function PreTradeStress() {
  const { summary, positions } = usePortfolio();
  const { setControls } = useControls();
  const [draft, setDraft] = useState<Record<string, number>>({});
  const [applied, setApplied] = useState<Record<string, number>>({});
  const [simData, setSimData] = useState<PortfolioRB | null>(null);
  const [simLoading, setSimLoading] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [virtualLegs, setVirtualLegs] = useState<VirtualLeg[]>([]);
  const [virtualInit, setVirtualInit] = useState(false);
  const [selectedExpiry, setSelectedExpiry] = useState<string>("");
  const [virtualGreeks, setVirtualGreeks] = useState<Record<string, number> | null>(null);
  const [virtualGreeksLoading, setVirtualGreeksLoading] = useState(false);
  const chainDefaultQty = 65;

  const defaults = useMemo(() => {
    const s = summary as any;
    const defaultSpot = positions?.current_spot ?? s?.current_spot ?? null;
    const defaultTheta = s?.greeks?.net_theta ?? null;
    const defaultGamma = s?.greeks?.net_gamma ?? null;
    const defaultVega = s?.greeks?.net_vega ?? null;
    const defaultCapital = s?.margin_used ?? s?.account_size ?? null;
    const out: Record<string, number> = {};
    if (defaultSpot != null) out.spot = Math.round(Number(defaultSpot));
    if (defaultTheta != null) out.theta = Math.round(Number(defaultTheta));
    if (defaultGamma != null) out.gamma = Number(Number(defaultGamma).toFixed(3));
    if (defaultVega != null) out.vega = Math.round(Number(defaultVega));
    if (defaultCapital != null) out.capital = Number((Number(defaultCapital) / 100000).toFixed(1));
    return out;
  }, [summary, positions]);

  useEffect(() => {
    if (!Object.keys(defaults).length) return;
    if (!Object.keys(draft).length) setDraft(defaults);
    if (!Object.keys(applied).length) setApplied(defaults);
  }, [defaults, draft, applied]);

  const normalizeCapital = (value?: number) => {
    if (value == null) return value;
    const v = Number(value);
    if (!Number.isFinite(v)) return value;
    return v > 1000 ? v / 100000 : v;
  };

  const positionId = (p: Record<string, unknown>) => {
    const token = p.instrument_token ?? p.token;
    if (token != null) return String(token);
    const sym = p.tradingsymbol ?? p.trading_symbol ?? p.symbol ?? "";
    const expiry = p.expiry ?? p.expiry_date ?? "";
    const strike = p.strike ?? p.strike_price ?? "";
    const opt = p.option_type ?? p.instrument_type ?? "";
    return `${sym}|${expiry}|${strike}|${opt}`;
  };

  useEffect(() => {
    if (virtualInit) return;
    const rows = positions?.positions || [];
    if (!rows.length) return;
    const initial = rows
      .filter((p) => Number(p.quantity ?? p.qty ?? 0) !== 0)
      .map((p) => {
        const qty = Number(p.quantity ?? p.qty ?? 0);
        const price = Number(
          p.last_price ??
          p.price ??
          p.average_price ??
          p.avg_price ??
          p.buy_price ??
          p.sell_price ??
          0
        );
        return {
          id: positionId(p),
          tradingsymbol: String(p.tradingsymbol ?? p.trading_symbol ?? p.symbol ?? "—"),
          expiry: p.expiry ? String(p.expiry).slice(0, 10) : "—",
          strike: p.strike ?? p.strike_price ?? "—",
          type: String(p.option_type ?? p.instrument_type ?? "—"),
          qty,
          price,
          source: "position" as const,
          instrument_token: (p.instrument_token ?? p.token) as number | null,
          enabled: true
        };
      });
    setVirtualLegs(initial);
    setVirtualInit(true);
  }, [positions, virtualInit]);

  const optionsChain = useCachedApi<OptionsChain>(
    `options_chain_v2_${selectedExpiry || "default"}`,
    `/derivatives/options-chain${selectedExpiry ? `?expiry=${encodeURIComponent(selectedExpiry)}&v=2` : "?v=2"}`,
    60_000
  );

  const maxCallOi = useMemo(() => {
    const rows = optionsChain.data?.rows || [];
    return rows.reduce((acc, r) => Math.max(acc, Number(r.call_oi || 0)), 0) || 1;
  }, [optionsChain.data]);

  const maxPutOi = useMemo(() => {
    const rows = optionsChain.data?.rows || [];
    return rows.reduce((acc, r) => Math.max(acc, Number(r.put_oi || 0)), 0) || 1;
  }, [optionsChain.data]);

  useEffect(() => {
    if (!selectedExpiry && optionsChain.data?.expiry) {
      setSelectedExpiry(optionsChain.data.expiry);
    }
  }, [optionsChain.data, selectedExpiry]);

  const addVirtualLeg = (row: OptionsChainRow, side: "CE" | "PE", action: "BUY" | "SELL") => {
    const isCall = side === "CE";
    const tradingsymbol = isCall ? row.call_symbol : row.put_symbol;
    if (!tradingsymbol) return;
    const price = Number(isCall ? row.call_ltp : row.put_ltp) || 0;
    const qty = action === "SELL" ? -Math.abs(chainDefaultQty) : Math.abs(chainDefaultQty);
    const legId = `${tradingsymbol}:${side}`;
    setVirtualLegs((prev) => {
      const existing = prev.find((l) => l.tradingsymbol === tradingsymbol);
      if (!existing) {
        return [
          ...prev,
          {
            id: legId,
            tradingsymbol,
            expiry: selectedExpiry || "",
            strike: row.strike,
            type: side,
            qty,
            price,
            source: "chain",
            instrument_token: isCall ? row.call_token ?? null : row.put_token ?? null,
            enabled: true
          }
        ];
      }
      const nextQty = existing.qty + qty;
      if (nextQty === 0) {
        return prev.filter((l) => l.tradingsymbol !== tradingsymbol);
      }
      return prev.map((l) =>
        l.tradingsymbol === tradingsymbol ? { ...l, qty: nextQty, price, enabled: true } : l
      );
    });
  };

  const updateVirtualQty = (id: string, qty: number) => {
    setVirtualLegs((prev) => prev.map((l) => (l.id === id ? { ...l, qty } : l)));
  };

  const toggleVirtualLeg = (id: string, enabled: boolean) => {
    setVirtualLegs((prev) => prev.map((l) => (l.id === id ? { ...l, enabled } : l)));
  };

  const selectedMap = useMemo(() => {
    const map = new Map<string, VirtualLeg>();
    virtualLegs.filter((l) => l.enabled).forEach((l) => {
      map.set(`${l.tradingsymbol}:${l.type}`, l);
    });
    return map;
  }, [virtualLegs]);

  const resetVirtual = () => {
    setVirtualInit(false);
  };

  const virtualPayload = useMemo(
    () => ({
      virtual_positions: virtualLegs.filter((l) => l.enabled).map((l) => ({
        tradingsymbol: l.tradingsymbol,
        quantity: l.qty,
        last_price: l.price,
        instrument_token: l.instrument_token ?? undefined
      }))
    }),
    [virtualLegs]
  );

  const sortedVirtualLegs = useMemo(() => {
    return [...virtualLegs].sort((a, b) => {
      const expCmp = String(a.expiry || "").localeCompare(String(b.expiry || ""));
      if (expCmp !== 0) return expCmp;
      const strikeA = Number(a.strike || 0);
      const strikeB = Number(b.strike || 0);
      if (strikeA !== strikeB) return strikeA - strikeB;
      const typeCmp = String(a.type || "").localeCompare(String(b.type || ""));
      if (typeCmp !== 0) return typeCmp;
      return String(a.tradingsymbol || "").localeCompare(String(b.tradingsymbol || ""));
    });
  }, [virtualLegs]);

  const currentVirtualLegs = useMemo(
    () => sortedVirtualLegs.filter((l) => l.source === "position"),
    [sortedVirtualLegs]
  );

  const newVirtualLegs = useMemo(
    () => sortedVirtualLegs.filter((l) => l.source === "chain"),
    [sortedVirtualLegs]
  );


  useEffect(() => {
    if (!virtualLegs.some((l) => l.enabled)) {
      setVirtualGreeks(null);
      return;
    }
    let active = true;
    setVirtualGreeksLoading(true);
    const timer = window.setTimeout(() => {
      apiFetch<VirtualGreeksResponse>("/portfolio/virtual-greeks", {
        method: "POST",
        body: virtualPayload
      })
        .then((res) => {
          if (!active) return;
          setVirtualGreeks(res.greeks || null);
        })
        .catch(() => {
          if (!active) return;
          setVirtualGreeks(null);
        })
        .finally(() => {
          if (!active) return;
          setVirtualGreeksLoading(false);
        });
    }, 250);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [virtualPayload, virtualLegs.length]);

  const runPortfolioSimulation = async () => {
    if (!virtualLegs.some((l) => l.enabled)) {
      setSimError("Add at least one leg to the virtual portfolio.");
      return;
    }
    setSimLoading(true);
    setSimError(null);
    try {
      const response = await apiFetch<PortfolioRB>("/risk-buckets/portfolio/simulate", {
        method: "POST",
        body: virtualPayload
      });
      setSimData(response);
    } catch (err) {
      setSimError(String(err));
    } finally {
      setSimLoading(false);
    }
  };

  const query = useMemo(() => {
    const params = new URLSearchParams();
    if (applied.spot != null) params.set("spot", String(applied.spot));
    if (applied.theta != null) params.set("theta", String(applied.theta));
    if (applied.gamma != null) params.set("gamma", String(applied.gamma));
    if (applied.vega != null) params.set("vega", String(applied.vega));
    if (applied.capital != null) params.set("capital", String(applied.capital * 100000));
    const qs = params.toString();
    return qs ? `?${qs}` : "";
  }, [applied]);

  const { data, error, loading } = useCachedApi<PreTrade>(
    `pre_trade${query}`,
    `/pre-trade/analysis${query}`,
    60_000
  );

  const comparisonMap = useMemo(() => {
    const rows = data?.comparison_table || [];
    const map: Record<string, string> = {};
    rows.forEach((r) => {
      const key = String((r as any)["Metric"] || "").toLowerCase();
      const value = (r as any)["Your Position"];
      if (!key) return;
      map[key] = value != null ? String(value) : "";
    });
    return map;
  }, [data]);

  const scenarioStats = useMemo(() => {
    const probMap = new Map<string, number>();
    (data?.scenario_table || []).forEach((r: any) => {
      const name = String(r["Scenario"] || "");
      const p = Number(String(r["Probability"] || "0").replace("%", ""));
      if (name) probMap.set(name, Number.isFinite(p) ? p / 100 : 0);
    });
    const rows = (data?.repriced_rows || []).map((r) => {
      const scenario = String(r["Scenario"]);
      const pnl = Number(String(r["Repriced P&L (₹)"] || "0").replace(/[₹,]/g, ""));
      const prob = probMap.get(scenario) ?? 0;
      return { scenario, pnl, prob };
    }).filter((r) => Number.isFinite(r.pnl) && Number.isFinite(r.prob));

    const totalProb = rows.reduce((acc, r) => acc + r.prob, 0);
    if (!rows.length || totalProb <= 0) return null;

    const mean = rows.reduce((acc, r) => acc + r.pnl * r.prob, 0) / totalProb;
    const sorted = [...rows].sort((a, b) => a.pnl - b.pnl);

    const esAt = (tail: number) => {
      let cum = 0;
      let sum = 0;
      let taken = 0;
      for (const r of sorted) {
        if (cum >= tail) break;
        const take = Math.min(r.prob, tail - cum);
        sum += r.pnl * take;
        taken += take;
        cum += take;
      }
      return taken > 0 ? sum / taken : 0;
    };

    return {
      mean,
      es95: esAt(0.05),
      es99: esAt(0.01)
    };
  }, [data]);

  const pickComparison = (needle: string) => {
    const key = Object.keys(comparisonMap).find((k) => k.includes(needle));
    return key ? comparisonMap[key] : "";
  };

  const toneFromComparison = (value: string, fallback: "neutral" | "info" = "neutral") => {
    if (!value) return fallback;
    if (value.includes("🟢")) return "positive";
    if (value.includes("🟡")) return "warning";
    if (value.includes("🔴")) return "negative";
    if (value.includes("⚠️")) return "warning";
    return fallback;
  };

  const emojiOnly = (value: string) => {
    if (!value) return "";
    const match = value.match(/(🟢|🟡|🔴|⚠️)/);
    return match ? match[1] : "";
  };

  useEffect(() => {
    const content = (
      <div className="control-grid">
        <label className="control-field">
          <span className="control-label">Spot</span>
          <input
            className="control-input"
            type="number"
            step="1"
            value={draft.spot ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, spot: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Capital (₹ Lacs)</span>
          <input
            className="control-input"
            type="number"
            step="0.1"
            value={draft.capital ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, capital: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Theta</span>
          <input
            className="control-input"
            type="number"
            step="1"
            value={draft.theta ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, theta: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Gamma</span>
          <input
            className="control-input"
            type="number"
            step="0.001"
            value={draft.gamma ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, gamma: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field">
          <span className="control-label">Vega</span>
          <input
            className="control-input"
            type="number"
            step="1"
            value={draft.vega ?? ""}
            onChange={(e) => setDraft((prev) => ({ ...prev, vega: Number(e.target.value) }))}
          />
        </label>
        <label className="control-field control-inline" style={{ alignSelf: "end" }}>
          <button
            className="control-input"
            onClick={() => {
              setApplied({
                spot: draft.spot != null ? Math.round(draft.spot) : draft.spot,
                theta: draft.theta != null ? Math.round(draft.theta) : draft.theta,
                gamma: draft.gamma != null ? Number(Number(draft.gamma).toFixed(3)) : draft.gamma,
                vega: draft.vega != null ? Math.round(draft.vega) : draft.vega,
                capital: draft.capital != null ? Number(normalizeCapital(draft.capital)?.toFixed(1)) : draft.capital
              });
            }}
          >
            Apply
          </button>
        </label>
      </div>
    );
    setControls({
      title: "Controls",
      summary: (
        <>
          <span><span className="controls-summary-key">Spot</span> {draft.spot != null ? Math.round(draft.spot) : "—"}</span>
          <span><span className="controls-summary-key">Theta</span> {draft.theta != null ? Math.round(draft.theta) : "—"}</span>
          <span><span className="controls-summary-key">Gamma</span> {draft.gamma != null ? Number(draft.gamma).toFixed(3) : "—"}</span>
          <span><span className="controls-summary-key">Vega</span> {draft.vega != null ? Math.round(draft.vega) : "—"}</span>
          <span><span className="controls-summary-key">Capital</span> {draft.capital != null ? `${Number(draft.capital).toFixed(1)} lacs` : "—"}</span>
        </>
      ),
      content
    });
    return () => setControls(null);
  }, [setControls, draft]);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      <SectionCard title="Virtual Portfolio Builder">
        <div className="pretrade-builder">
          <div className="pretrade-chain">
            <div className="chain-toolbar">
              <div className="chain-spot">
                NIFTY {optionsChain.data?.spot ? Number(optionsChain.data.spot).toFixed(2) : "—"}
                {optionsChain.data?.date ? <span className="chain-date">{optionsChain.data.date}</span> : null}
              </div>
              <div className="chain-controls">
                <label className="chain-field">
                  <span>Expiry</span>
                  <select value={selectedExpiry} onChange={(e) => setSelectedExpiry(e.target.value)}>
                    {(optionsChain.data?.expiry_list || []).map((opt) => (
                      <option key={opt} value={opt}>{opt}</option>
                    ))}
                  </select>
                </label>
              </div>
            </div>
            {optionsChain.loading ? (
              <LoadingState />
            ) : optionsChain.error ? (
              <ErrorState message={optionsChain.error} />
            ) : (
              <div className="chain-table-wrap">
                <table className="options-chain">
                  <thead>
                    <tr>
                      <th className="call-col">Call LTP</th>
                      <th className="call-col">Call OI</th>
                      <th className="chain-strike-head">Strike</th>
                      <th className="put-col">Put OI</th>
                      <th className="put-col">Put LTP</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(optionsChain.data?.rows || []).map((row) => {
                      const callSelected = selectedMap.get(`${row.call_symbol}:CE`);
                      const putSelected = selectedMap.get(`${row.put_symbol}:PE`);
                      return (
                      <tr key={row.strike}>
                        <td className={`chain-ltp-cell call-col ${callSelected ? "is-selected" : ""}`}>
                          <div className="chain-ltp-wrap">
                            <span className="chain-ltp">{row.call_ltp != null ? row.call_ltp.toFixed(2) : "—"}</span>
                            <div className="chain-actions fixed">
                              <button className="chain-action buy" onClick={() => addVirtualLeg(row, "CE", "BUY")}>B</button>
                              <button className="chain-action sell" onClick={() => addVirtualLeg(row, "CE", "SELL")}>S</button>
                            </div>
                          </div>
                        </td>
                        <td className="chain-oi call-col">
                          <div className="chain-oi-wrap">
                            <span className="chain-oi-value">{row.call_oi != null ? row.call_oi.toFixed(0) : "—"}</span>
                            <span className="chain-oi-bar call-bar" style={{ width: `${Math.min(100, ((row.call_oi || 0) / maxCallOi) * 100)}%` }} />
                            {callSelected ? (
                              <span className={`chain-pill ${callSelected.qty < 0 ? "sell" : "buy"}`}>
                                {callSelected.qty < 0 ? "S" : "B"} {Math.abs(callSelected.qty)}
                              </span>
                            ) : null}
                          </div>
                        </td>
                        <td className="chain-strike">
                          <div className="chain-strike-text">{row.strike.toFixed(0)}</div>
                          <div className="chain-strike-symbol">{row.call_symbol || row.put_symbol || "—"}</div>
                        </td>
                        <td className="chain-oi put-col">
                          <div className="chain-oi-wrap">
                            <span className="chain-oi-value">{row.put_oi != null ? row.put_oi.toFixed(0) : "—"}</span>
                            <span className="chain-oi-bar put-bar" style={{ width: `${Math.min(100, ((row.put_oi || 0) / maxPutOi) * 100)}%` }} />
                            {putSelected ? (
                              <span className={`chain-pill ${putSelected.qty < 0 ? "sell" : "buy"}`}>
                                {putSelected.qty < 0 ? "S" : "B"} {Math.abs(putSelected.qty)}
                              </span>
                            ) : null}
                          </div>
                        </td>
                        <td className={`chain-ltp-cell put-col ${putSelected ? "is-selected" : ""}`}>
                          <div className="chain-ltp-wrap">
                            <span className="chain-ltp">{row.put_ltp != null ? row.put_ltp.toFixed(2) : "—"}</span>
                            <div className="chain-actions fixed">
                              <button className="chain-action buy" onClick={() => addVirtualLeg(row, "PE", "BUY")}>B</button>
                              <button className="chain-action sell" onClick={() => addVirtualLeg(row, "PE", "SELL")}>S</button>
                            </div>
                          </div>
                        </td>
                      </tr>
                    );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
          <div className="pretrade-portfolio">
            <div className="portfolio-header">
              <div>Virtual Portfolio ({virtualLegs.filter((l) => l.enabled).length})</div>
              <button className="control-input" onClick={resetVirtual}>Reset to Current</button>
            </div>
            <div className="virtual-greeks">
              {virtualGreeksLoading ? (
                <span>Net Greeks: Loading...</span>
              ) : virtualGreeks ? (
                <span>
                  Net Greeks: Δ {formatNumber(virtualGreeks.net_delta, 2)} | Γ {formatNumber(virtualGreeks.net_gamma, 4)} | Θ {formatNumber(virtualGreeks.net_theta, 0)} | Vega {formatNumber(virtualGreeks.net_vega, 0)}
                </span>
              ) : (
                <span>Net Greeks: —</span>
              )}
            </div>
            {virtualLegs.length ? (
              <>
                <div className="virtual-section-title">Current Positions</div>
                <DataTable
                  columns={[
                    {
                      key: "enabled",
                      label: "",
                      render: (row) => (
                        <input
                          className="table-check"
                          type="checkbox"
                          checked={Boolean(row.enabled)}
                          onChange={(e) => toggleVirtualLeg(String(row.id || ""), e.target.checked)}
                        />
                      )
                    },
                    { key: "tradingsymbol", label: "Symbol" },
                    { key: "expiry", label: "Expiry" },
                    { key: "strike", label: "Strike" },
                    { key: "type", label: "Type" },
                    {
                      key: "qty",
                      label: "Qty",
                      render: (row) => (
                        <input
                          className="table-input"
                          type="number"
                          step="1"
                          value={Number(row.qty || 0)}
                          onChange={(e) => updateVirtualQty(String(row.id || ""), Number(e.target.value))}
                        />
                      )
                    },
                    {
                      key: "price",
                      label: "Price",
                      format: (value) => formatInr(value as number)
                    }
                  ]}
                  rows={currentVirtualLegs as unknown as Record<string, unknown>[]}
                  maxHeight={220}
                />
                <div className="virtual-section-title">New Trades</div>
                <DataTable
                  columns={[
                    {
                      key: "enabled",
                      label: "",
                      render: (row) => (
                        <input
                          className="table-check"
                          type="checkbox"
                          checked={Boolean(row.enabled)}
                          onChange={(e) => toggleVirtualLeg(String(row.id || ""), e.target.checked)}
                        />
                      )
                    },
                    { key: "tradingsymbol", label: "Symbol" },
                    { key: "expiry", label: "Expiry" },
                    { key: "strike", label: "Strike" },
                    { key: "type", label: "Type" },
                    {
                      key: "qty",
                      label: "Qty",
                      render: (row) => (
                        <input
                          className="table-input"
                          type="number"
                          step="1"
                          value={Number(row.qty || 0)}
                          onChange={(e) => updateVirtualQty(String(row.id || ""), Number(e.target.value))}
                        />
                      )
                    },
                    {
                      key: "price",
                      label: "Price",
                      format: (value) => formatInr(value as number)
                    }
                  ]}
                  rows={newVirtualLegs as unknown as Record<string, unknown>[]}
                  maxHeight={220}
                />
              </>
            ) : (
              <div>No legs selected.</div>
            )}
            <div className="controls-row" style={{ marginTop: 12, display: "flex", gap: 12, alignItems: "center" }}>
              <button
                className="control-input"
                style={{ background: "linear-gradient(135deg, var(--gs-accent), var(--gs-accent2))", border: "none", color: "#fff" }}
                onClick={runPortfolioSimulation}
              >
                {simData ? "Simulate Again" : "Run Portfolio Simulation"}
              </button>
              {simError ? <span className="negative">{simError}</span> : null}
            </div>
          </div>
        </div>
      </SectionCard>

      {!simData && !simLoading && !simError ? (
        <SectionCard title="Risk Buckets Portfolio Simulation">
          <div>Select positions and run a simulation to view portfolio risk buckets.</div>
        </SectionCard>
      ) : (
        <RiskBucketsPortfolio
          dataOverride={simData ?? undefined}
          loadingOverride={simLoading}
          errorOverride={simError}
          showControlsBar={false}
        />
      )}

      <SectionCard title="Stress Testing">
        {loading || !data ? <LoadingState /> : (
          <MetricGrid>
            <MetricCard label="Spot" value={formatInr(data.spot)} />
            <MetricCard label="Capital" value={formatInr(data.capital)} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Scenario KPIs">
        {loading ? <LoadingState /> : (
          <MetricGrid>
            <MetricCard label="Mean P&L" value={scenarioStats ? formatInr(scenarioStats.mean) : "—"} />
            <MetricCard label="ES95" value={scenarioStats ? formatInr(scenarioStats.es95) : "—"} />
            <MetricCard label="ES99" value={scenarioStats ? formatInr(scenarioStats.es99) : "—"} />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Normalized Greeks (per ₹1L)">
        {loading || !data ? <LoadingState /> : (
          <MetricGrid>
            <MetricCard
              label="Delta"
              value={formatNumber(data.normalized.delta, 2)}
              delta={pickComparison("delta")}
              tone="info"
            />
            <MetricCard
              label="Theta"
              value={formatNumber(data.normalized.theta, 0)}
              delta={pickComparison("theta")}
              tone={toneFromComparison(pickComparison("theta"))}
            />
            <MetricCard
              label="Gamma"
              value={formatNumber(data.normalized.gamma, 4)}
              delta={emojiOnly(pickComparison("gamma"))}
              tone={toneFromComparison(pickComparison("gamma"))}
            />
            <MetricCard
              label="Vega"
              value={formatNumber(data.normalized.vega, 0)}
              delta={emojiOnly(pickComparison("vega"))}
              tone={toneFromComparison(pickComparison("vega"))}
            />
          </MetricGrid>
        )}
      </SectionCard>

      <SectionCard title="Scenario Repricing">
        {loading || !data ? <LoadingState /> : (
          <>
            {(() => {
              const probMap = new Map<string, string>();
              (data.scenario_table || []).forEach((r: any) => {
                const name = String(r["Scenario"] || "");
                const p = String(r["Probability"] || "");
                if (name) probMap.set(name, p);
              });
              const rows = (data.repriced_rows || []).map((r) => {
                const status = String(r["Breach"] || r["Status"] || r["status"] || "");
                const lossPct = Number(String(r["Loss % Capital"] || "0").replace("%", ""));
                const thresholdPct = Number(String(r["Threshold % NAV"] || "0").replace("%", ""));
                const isBad =
                  (Number.isFinite(lossPct) && Number.isFinite(thresholdPct) && lossPct > thresholdPct) ||
                  /fail|breach|red/i.test(status);
                const bubble = isBad ? "🔴" : "🟢";
                return { ...r, Probability: probMap.get(String(r["Scenario"])) || "—", Breach: `${bubble} ${status}`.trim() };
              });
              return (
            <DataTable
              columns={[
                { key: "Scenario", label: "Scenario" },
                { key: "dS% / dIV", label: "dS% / dIV" },
                { key: "Repriced P&L (₹)", label: "Repriced P&L" },
                { key: "Loss % Capital", label: "Loss % NAV" },
                { key: "Threshold % NAV", label: "Threshold % NAV" },
                { key: "Probability", label: "Prob %" },
                { key: "Breach", label: "Status" }
              ]}
              rows={rows}
            />
              );
            })()}
          </>
        )}
      </SectionCard>

      <SectionCard title="Scenario Loss Distribution">
        {loading || !data ? <LoadingState /> : (
          (() => {
            const probMap = new Map<string, number>();
            (data.scenario_table || []).forEach((r: any) => {
              const name = String(r["Scenario"] || "");
              const p = Number(String(r["Probability"] || "0").replace("%", ""));
              if (name) probMap.set(name, Number.isFinite(p) ? p : 0);
            });
            const rows = (data.repriced_rows || []).map((r) => {
              const lossPct = Number(String(r["Loss % Capital"] || "0").replace("%", ""));
              const pnl = Number(String(r["Repriced P&L (₹)"] || "0").replace(/[₹,]/g, ""));
              return {
                scenario: String(r["Scenario"]),
                lossPct,
                pnl,
                prob: probMap.get(String(r["Scenario"])) ?? 0
              };
            }).sort((a, b) => a.pnl - b.pnl);
            let cum = 0;
            const totalProb = rows.reduce((acc, r) => acc + (Number.isFinite(r.prob) ? r.prob : 0), 0);
            const cumProb = rows.map((r) => {
              const p = Number.isFinite(r.prob) ? r.prob : 0;
              cum += p;
              return cum;
            });
            return (
              <Plot
                data={[
                  {
                    type: "bar",
                    x: rows.map((r) => r.scenario),
                    y: rows.map((r) => r.pnl),
                    marker: {
                      color: rows.map((r) => (r.pnl < 0 ? "#ff4d4d" : "#2ecc71"))
                    },
                    name: "P&L (₹)"
                  },
                  {
                    type: "scatter",
                    mode: "lines+markers",
                    x: rows.map((r) => r.scenario),
                    y: cumProb,
                    yaxis: "y2",
                    line: { color: "#F5B041", width: 2 },
                    marker: { color: "#F5B041", size: 6 },
                    name: "Cumulative %"
                  }
                ]}
                layout={{
                  height: 360,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 40, r: 40, t: 20, b: 70 },
                  xaxis: { tickangle: -30, tickfont: { color: "#b9c4d6", size: 10 }, automargin: true },
                  yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, title: "P&L (₹)" },
                  yaxis2: {
                    overlaying: "y",
                    side: "right",
                    range: [0, 100],
                    tickfont: { color: "#b9c4d6", size: 10 },
                    title: "Cum %"
                  },
                  legend: { orientation: "h", x: 0, y: -0.2, font: { color: "#b9c4d6", size: 10 } }
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            );
          })()
        )}
      </SectionCard>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">Zone Comparison Table (Debug)</div>
        </summary>
        <div className="section-body">
          {loading || !data ? <LoadingState /> : (
            <DataTable
              columns={[
                { key: "Metric", label: "Metric" },
                { key: "Zone 1 Range", label: "Zone 1" },
                { key: "Zone 2 Range", label: "Zone 2" },
                { key: "Zone 3 Range", label: "Zone 3" },
                { key: "Your Position", label: "Your Position" }
              ]}
              rows={data.comparison_table}
            />
          )}
        </div>
      </details>
    </>
  );
}
