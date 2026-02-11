import { useMemo } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import Plot from "../components/Plot";
import { formatInr, formatPct } from "../components/format";
import { useCachedApi } from "../hooks/useCachedApi";

type BucketResponse = {
  bucket_rows: Record<string, unknown>[];
  bucket_sims: Record<string, Record<string, number> | null>;
};

export default function RiskBucketsBucket() {
  const { data, error, loading } = useCachedApi<BucketResponse>(
    "risk_buckets_bucket",
    "/risk-buckets/buckets",
    60_000
  );

  const cards = useMemo(() => {
    const rows = data?.bucket_rows || [];
    const sims = data?.bucket_sims || {};
    return rows.map((r) => {
      const bucket = String(r.bucket || "");
      const sim = sims[bucket] || null;
      return {
        bucket,
        es99: formatInr(r.bucket_es99_inr),
        es99Pct: formatPct(r.bucket_es99_pct_of_bucket_capital),
        mean: sim ? formatInr(sim.mean) : "N/A",
        median: sim ? formatInr(sim.median) : "N/A",
        pop: sim ? formatPct((Number(sim.prob_loss) || 0) * -100) : "N/A",
        tail: sim ? formatInr(sim.p5) : "N/A"
      };
    });
  }, [data]);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      {loading || !data ? (
        <LoadingState />
      ) : (
        <>
          <div className="rb-card-grid">
            {cards.map((c) => (
              <div key={c.bucket} className="rb-card">
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <h4>{c.bucket} Risk Bucket</h4>
                  <span className="badge">Live</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "30px", paddingTop: '1rem' }}>
                  <div>
                    <div className="metric-label">ES99</div>
                    <div className="metric-value">{c.es99}</div>
                  </div>
                  <div>
                    <div className="metric-label">ES99 %</div>
                    <div className="metric-value">{c.es99Pct}</div>
                  </div>
                  <div>
                    <div className="metric-label">Mean PnL</div>
                    <div className="metric-value">{c.mean}</div>
                  </div>
                  <div>
                    <div className="metric-label">Median PnL</div>
                    <div className="metric-value">{c.median}</div>
                  </div>
                  <div>
                    <div className="metric-label">POP %</div>
                    <div className="metric-value">{c.pop}</div>
                  </div>
                  <div>
                    <div className="metric-label">Tail (P5)</div>
                    <div className="metric-value negative">{c.tail}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="rb-two-col" style={{ marginTop: "20px" }}>
            <SectionCard title="Bucket ES99 Comparison">
              <Plot
                data={[{
                  type: "bar",
                  x: data.bucket_rows.map((r) => r.bucket as string),
                  y: data.bucket_rows.map((r) => Number(r.bucket_es99_inr || 0)),
                  marker: { color: "#2D7DFF" }
                }]}
                layout={{
                  height: 320,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 44, r: 20, t: 20, b: 48 },
                  xaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 }, automargin: true },
                  yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } }
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            </SectionCard>

            <SectionCard title="Bucket Expected PnL">
              <Plot
                data={[{
                  type: "bar",
                  x: data.bucket_rows.map((r) => r.bucket as string),
                  y: data.bucket_rows.map((r) => Number(r.bucket_expected_pnl_inr || 0)),
                  marker: { color: "#FFB020" }
                }]}
                layout={{
                  height: 320,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { l: 44, r: 20, t: 20, b: 48 },
                  xaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 }, automargin: true },
                  yaxis: { tickfont: { color: "#b9c4d6", size: 10 }, titlefont: { color: "#b9c4d6", size: 11 } }
                }}
                config={{ displayModeBar: false, responsive: true }}
                useResizeHandler
                style={{ width: "100%" }}
              />
            </SectionCard>
          </div>
        </>
      )}
    </>
  );
}
