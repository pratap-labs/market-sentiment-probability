import { useEffect, useState } from "react";
import { apiFetch } from "../api/client";
import { getCacheEntry, setCache } from "../api/cache";

type State<T> = { data: T | null; error: string | null; loading: boolean };
const inflight = new Map<string, Promise<unknown>>();

export function useCachedApi<T>(key: string, path: string, ttlMs = 60_000): State<T> {
  const [state, setState] = useState<State<T>>({ data: null, error: null, loading: true });

  useEffect(() => {
    const cacheKey = `${key}::${path}`;
    const cached = getCacheEntry<T>(cacheKey);
    if (cached) {
      setState({ data: cached.value, error: null, loading: false });
      return;
    }

    const existing = inflight.get(cacheKey);
    if (existing) {
      existing
        .then((data) => {
          setState({ data: data as T, error: null, loading: false });
        })
        .catch((err) => {
          setState((prev) => ({ ...prev, error: String(err), loading: false }));
        });
      return;
    }

    const promise = apiFetch<T>(path);
    inflight.set(cacheKey, promise);
    promise
      .then((data) => {
        setCache(cacheKey, data, ttlMs);
        setState({ data, error: null, loading: false });
      })
      .catch((err) => {
        setState((prev) => ({ ...prev, error: String(err), loading: false }));
      })
      .finally(() => {
        inflight.delete(cacheKey);
      });
  }, [key, path, ttlMs]);

  return state;
}
