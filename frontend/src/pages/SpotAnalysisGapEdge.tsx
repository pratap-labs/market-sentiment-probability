import { useMemo } from "react";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import MetricCard from "../components/MetricCard";
import MetricGrid from "../components/MetricGrid";
import SectionCard from "../components/SectionCard";
import { useCachedApi } from "../hooks/useCachedApi";

type GapBucket = {
  events: number;
  hit_count: number;
  hit_rate_pct: number;
  hit_rate_ci_low_pct: number;
  hit_rate_ci_high_pct: number;
  extend_beyond_open_gap_count: number;
  extend_beyond_open_gap_pct: number;
  reversal_count: number;
  reversal_pct: number;
  mean_close_pct: number | null;
  median_gap_pct: number | null;
  median_close_pct: number | null;
  mean_close_minus_gap_pct: number | null;
};

type GapWindow = {
  years: number;
  start_date: string;
  end_date: string;
  sample_days: number;
  threshold_pct: number;
  gap_up: GapBucket;
  gap_down: GapBucket;
  combined: {
    events: number;
    hit_count: number;
    hit_rate_pct: number;
    hit_rate_ci_low_pct: number;
    hit_rate_ci_high_pct: number;
  };
};

type GapEdgeResponse = {
  source: string;
  as_of_date: string;
  threshold_pct: number;
  windows: GapWindow[];
};

function pct(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${Number(v).toFixed(digits)}%`;
}

function num(v: number | null | undefined, digits = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return Number(v).toFixed(digits);
}

export default function SpotAnalysisGapEdge() {
  const edge = useCachedApi<GapEdgeResponse>(
    "spot_gap_edge",
    "/spot-analysis/gap-edge?threshold_pct=0.5",
    60_000
  );

  const windows = useMemo(
    () => (edge.data?.windows || []).slice().sort((a, b) => a.years - b.years),
    [edge.data]
  );
  const thresholdLabel = useMemo(() => num(edge.data?.threshold_pct, 2), [edge.data]);
  const window3y = useMemo(
    () => windows.find((w) => Number(w.years) === 3) || windows[windows.length - 1] || null,
    [windows]
  );

  return (
    <>
      <SectionCard title="Gap Persistence Edge (NIFTY)">
        {edge.loading ? (
          <LoadingState />
        ) : edge.error || !edge.data ? (
          <ErrorState message={String(edge.error || "Gap edge data unavailable")} />
        ) : (
          <>
            <div style={{ marginBottom: 12, color: "#9fb0c9", fontSize: 12 }}>
              Rule: if open gaps by more than ±{num(edge.data.threshold_pct, 2)}% vs previous close, check if close
              stays beyond the same threshold in the same direction.
              <br />
              As of: {edge.data.as_of_date}
            </div>

            {window3y ? (
              <MetricGrid>
                <MetricCard label="3Y Events" value={window3y.combined.events} />
                <MetricCard
                  label="3Y Hit Rate"
                  value={pct(window3y.combined.hit_rate_pct)}
                  delta={`${window3y.combined.hit_count}/${window3y.combined.events}`}
                />
                <MetricCard
                  label="3Y 95% CI"
                  value={`${pct(window3y.combined.hit_rate_ci_low_pct)} to ${pct(window3y.combined.hit_rate_ci_high_pct)}`}
                />
                <MetricCard
                  label="3Y Gap-Up Hit"
                  value={pct(window3y.gap_up.hit_rate_pct)}
                  delta={`${window3y.gap_up.hit_count}/${window3y.gap_up.events}`}
                />
                <MetricCard
                  label="3Y Gap-Down Hit"
                  value={pct(window3y.gap_down.hit_rate_pct)}
                  delta={`${window3y.gap_down.hit_count}/${window3y.gap_down.events}`}
                />
              </MetricGrid>
            ) : null}

            <div className="table-wrap" style={{ marginTop: 14 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Window</th>
                    <th>Date Range</th>
                    <th>Sample Days</th>
                    <th>Total Events</th>
                    <th>Hit Count</th>
                    <th>Hit Rate</th>
                    <th>95% CI</th>
                    <th>Gap-Up Mean Close %</th>
                    <th>Gap-Up Median Close %</th>
                    <th>Gap-Down Mean Close %</th>
                    <th>Gap-Down Median Close %</th>
                  </tr>
                </thead>
                <tbody>
                  {windows.map((w) => (
                    <tr key={`win-${w.years}`}>
                      <td>{w.years}Y</td>
                      <td>{w.start_date} to {w.end_date}</td>
                      <td>{w.sample_days}</td>
                      <td>{w.combined.events}</td>
                      <td>{w.combined.hit_count}</td>
                      <td>{pct(w.combined.hit_rate_pct)}</td>
                      <td>{pct(w.combined.hit_rate_ci_low_pct)} to {pct(w.combined.hit_rate_ci_high_pct)}</td>
                      <td>{pct(w.gap_up.mean_close_pct, 3)}</td>
                      <td>{pct(w.gap_up.median_close_pct, 3)}</td>
                      <td>{pct(w.gap_down.mean_close_pct, 3)}</td>
                      <td>{pct(w.gap_down.median_close_pct, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </SectionCard>

      {window3y ? (
        <SectionCard title="3Y Directional Breakdown">
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Side</th>
                  <th>Events</th>
                  <th>Hit Count</th>
                  <th>Hit Rate</th>
                  <th>95% CI</th>
                  <th>Extend Beyond Open Gap</th>
                  <th>Reversal Count</th>
                  <th>Mean Close %</th>
                  <th>Median Close %</th>
                  <th>Median Gap %</th>
                  <th>Mean (Close - Gap) %</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { side: `Gap Up > +${thresholdLabel}%`, bucket: window3y.gap_up },
                  { side: `Gap Down < -${thresholdLabel}%`, bucket: window3y.gap_down },
                ].map((row) => (
                  <tr key={row.side}>
                    <td>{row.side}</td>
                    <td>{row.bucket.events}</td>
                    <td>{row.bucket.hit_count}</td>
                    <td>{pct(row.bucket.hit_rate_pct)}</td>
                    <td>{pct(row.bucket.hit_rate_ci_low_pct)} to {pct(row.bucket.hit_rate_ci_high_pct)}</td>
                    <td>
                      {row.bucket.extend_beyond_open_gap_count}/{row.bucket.events} ({pct(row.bucket.extend_beyond_open_gap_pct)})
                    </td>
                    <td>
                      {row.bucket.reversal_count}/{row.bucket.events} ({pct(row.bucket.reversal_pct)})
                    </td>
                    <td>{pct(row.bucket.mean_close_pct, 3)}</td>
                    <td>{pct(row.bucket.median_close_pct, 3)}</td>
                    <td>{pct(row.bucket.median_gap_pct, 3)}</td>
                    <td>{pct(row.bucket.mean_close_minus_gap_pct, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </SectionCard>
      ) : null}
    </>
  );
}
