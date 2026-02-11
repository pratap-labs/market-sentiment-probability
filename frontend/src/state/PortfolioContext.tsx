import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { apiFetch } from "../api/client";
import { getCache, setCache } from "../api/cache";

export type PortfolioSummary = Record<string, unknown> | null;
export type PositionsResponse = { positions: Record<string, unknown>[]; current_spot: number } | null;

type PortfolioState = {
  summary: PortfolioSummary;
  positions: PositionsResponse;
  loading: boolean;
  error: string | null;
  refresh: () => void;
  refreshWithSpot: (spot: number) => void;
};

const PortfolioContext = createContext<PortfolioState | null>(null);

const CACHE_TTL = 60_000;

export function PortfolioProvider({ children }: { children: React.ReactNode }) {
  const [summary, setSummary] = useState<PortfolioSummary>(null);
  const [positions, setPositions] = useState<PositionsResponse>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      apiFetch<PortfolioSummary>("/portfolio/summary"),
      apiFetch<PositionsResponse>("/portfolio/positions")
    ])
      .then(([s, p]) => {
        setSummary(s);
        setPositions(p);
        setCache("portfolio_summary", s, CACHE_TTL);
        setCache("portfolio_positions", p, CACHE_TTL);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  const refreshWithSpot = useCallback((spot: number) => {
    setLoading(true);
    setError(null);
    const q = `?spot=${encodeURIComponent(spot)}`;
    Promise.all([
      apiFetch<PortfolioSummary>(`/portfolio/summary${q}`),
      apiFetch<PositionsResponse>(`/portfolio/positions${q}`)
    ])
      .then(([s, p]) => {
        setSummary(s);
        setPositions(p);
        setCache("portfolio_summary", s, CACHE_TTL);
        setCache("portfolio_positions", p, CACHE_TTL);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    const cachedSummary = getCache<PortfolioSummary>("portfolio_summary");
    const cachedPositions = getCache<PositionsResponse>("portfolio_positions");
    if (cachedSummary) setSummary(cachedSummary);
    if (cachedPositions) setPositions(cachedPositions);
    load();
  }, [load]);

  const value = useMemo(
    () => ({ summary, positions, loading, error, refresh: load, refreshWithSpot }),
    [summary, positions, loading, error, load, refreshWithSpot]
  );

  return <PortfolioContext.Provider value={value}>{children}</PortfolioContext.Provider>;
}

export function usePortfolio() {
  const ctx = useContext(PortfolioContext);
  if (!ctx) throw new Error("usePortfolio must be used within PortfolioProvider");
  return ctx;
}
