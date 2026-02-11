import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import { useCachedApi } from "../hooks/useCachedApi";

type Product = { framework: string; daily_workflow: string[] };

export default function ProductOverview() {
  const { data, error, loading } = useCachedApi<Product>("product_overview", "/product/overview", 60_000);

  return (
    <>
      {error ? <ErrorState message={error} /> : null}
      <SectionCard title="Framework Summary">
        {loading || !data ? <LoadingState /> : <div>{data.framework}</div>}
      </SectionCard>
      <SectionCard title="Daily Workflow">
        {loading || !data ? <LoadingState /> : (
          <ul>
            {data.daily_workflow.map((step) => (
              <li key={step}>{step}</li>
            ))}
          </ul>
        )}
      </SectionCard>
    </>
  );
}
