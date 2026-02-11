type MetricCardProps = {
  label: string;
  value: string | number | null | undefined;
  delta?: string | number | null;
  tone?: "positive" | "negative" | "neutral" | "warning" | "info";
  tooltip?: string;
};

export default function MetricCard({ label, value, delta, tone = "neutral", tooltip }: MetricCardProps) {
  return (
    <div className="metric-card">
      <div className="metric-label">
        <span>{label}</span>
        {tooltip ? (
          <span className="metric-help">
            ?
            <span className="metric-popover">
              {tooltip}
            </span>
          </span>
        ) : null}
      </div>
      <div className={`metric-value ${tone}`}>{value ?? "—"}</div>
      <div className={`metric-delta ${tone}`}>{delta ?? "\u00A0"}</div>
    </div>
  );
}
