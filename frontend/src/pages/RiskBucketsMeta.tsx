import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import { useCachedApi } from "../hooks/useCachedApi";

type Meta = { zone_rules: Record<string, unknown>[]; bucket_meta: Record<string, unknown>[]; iv_regime: string };

export default function RiskBucketsMeta() {
  const { data, error, loading } = useCachedApi<Meta>("risk_buckets_meta", "/risk-buckets/meta", 60_000);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      <SectionCard title="Zone Rules">
        {loading || !data ? <LoadingState /> : (
          <DataTable
            columns={[
              { key: "Zone", label: "Zone" },
              { key: "Theta (per 1L)", label: "Theta/1L" },
              { key: "Gamma", label: "Gamma" },
              { key: "Vega (per 1L)", label: "Vega" }
            ]}
            rows={data.zone_rules}
          />
        )}
      </SectionCard>
      <SectionCard title="Historical Buckets">
        {loading || !data ? <LoadingState /> : (
          <DataTable
            columns={[
              { key: "Bucket", label: "Bucket" },
              { key: "Label", label: "Label" },
              { key: "Definition", label: "Definition" },
              { key: "Notes", label: "Notes" }
            ]}
            rows={data.bucket_meta}
          />
        )}
      </SectionCard>
    </>
  );
}
