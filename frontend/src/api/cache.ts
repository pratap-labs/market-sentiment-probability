export type CacheEntry<T> = {
  value: T;
  expiresAt: number;
};

const PREFIX = "gs_cache:";

export function getCache<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CacheEntry<T>;
    if (Date.now() > parsed.expiresAt) {
      localStorage.removeItem(PREFIX + key);
      return null;
    }
    return parsed.value;
  } catch {
    return null;
  }
}

export function getCacheEntry<T>(key: string): CacheEntry<T> | null {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CacheEntry<T>;
    if (Date.now() > parsed.expiresAt) {
      localStorage.removeItem(PREFIX + key);
      return null;
    }
    return parsed;
  } catch {
    return null;
  }
}

export function setCache<T>(key: string, value: T, ttlMs: number): void {
  const entry: CacheEntry<T> = { value, expiresAt: Date.now() + ttlMs };
  localStorage.setItem(PREFIX + key, JSON.stringify(entry));
}
