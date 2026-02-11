export function formatInr(value: unknown): string {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return `₹${num.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
}

export function formatPct(value: unknown): string {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return `${num.toFixed(2)}%`;
}

export function formatNumber(value: unknown, decimals = 2): string {
  const num = Number(value);
  if (!Number.isFinite(num)) return "—";
  return num.toFixed(decimals);
}
