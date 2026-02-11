import { useEffect, useMemo, useState } from "react";
import { API_BASE_URL } from "../api/client";

type AuthStatus = { has_token: boolean; token_expired: boolean; saved_at: string | null };
type CacheStatusItem = { updated_at: string | null };
type CacheStatus = Record<string, CacheStatusItem>;

function formatAge(iso?: string | null) {
  if (!iso) return "N/A";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return "N/A";
  const diff = Date.now() - t;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

export default function LiveTicker() {
  const [auth, setAuth] = useState<AuthStatus | null>(null);
  const [cache, setCache] = useState<CacheStatus | null>(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const [a, c] = await Promise.all([
          fetch(`${API_BASE_URL}/auth/status`).then((r) => r.json()),
          fetch(`${API_BASE_URL}/data-source/cache-status`).then((r) => r.json())
        ]);
        if (!active) return;
        setAuth(a);
        setCache(c);
      } catch {
        if (!active) return;
      }
    };
    load();
    const id = setInterval(load, 30000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const items = useMemo(() => {
    const res: { text: string; tone?: "good" | "bad" | "neutral" }[] = [];
    if (auth) {
      const ok = auth.has_token && !auth.token_expired;
      res.push({ text: `Kite: ${ok ? "Authenticated" : "Not Authenticated"}`, tone: ok ? "good" : "bad" });
      if (auth.saved_at) res.push({ text: `Token saved ${formatAge(auth.saved_at)}`, tone: "neutral" });
    } else {
      res.push({ text: "Kite: —", tone: "neutral" });
    }
    if (cache) {
      res.push({ text: `NIFTY OHLCV was last updated ${formatAge(cache.nifty_ohlcv?.updated_at)}`, tone: "neutral" });
      res.push({ text: `Futures data was last updated ${formatAge(cache.nifty_futures?.updated_at)}`, tone: "neutral" });
      const optionsUpdated = cache.nifty_options_ce?.updated_at || cache.nifty_options_pe?.updated_at || null;
      res.push({ text: `Options data was last updated ${formatAge(optionsUpdated)}`, tone: "neutral" });
      res.push({ text: `Pre trade selection uses options pricing from last updated date ${formatAge(optionsUpdated)}`, tone: "neutral" });
    }
    return res;
  }, [auth, cache]);

  return (
    <div className="ticker">
      <div className="ticker-track">
        {items.map((item, idx) => (
          <span key={`${item.text}-${idx}`} className="ticker-item">
            {item.text}
            {idx < items.length - 1 ? <span className="ticker-sep">•</span> : null}
          </span>
        ))}
      </div>
    </div>
  );
}
