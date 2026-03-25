import { useMemo } from "react";
import DataTable from "../components/DataTable";
import ErrorState from "../components/ErrorState";
import LoadingState from "../components/LoadingState";
import MetricCard from "../components/MetricCard";
import MetricGrid from "../components/MetricGrid";
import SectionCard from "../components/SectionCard";
import { useCachedApi } from "../hooks/useCachedApi";

type NeighborhoodRow = {
  code?: string;
  neighborhood?: string;
  household_total?: number;
  residents_total?: number;
  family_share_pct?: number;
  single_share_pct?: number;
  recent_mover_pct?: number;
  long_term_pct?: number;
  netherlands_nationality_pct?: number;
  netherlands_nationality?: number;
  uk_nationality?: number;
  italy_nationality?: number;
  turkey_nationality?: number;
  germany_nationality?: number;
  france_nationality?: number;
  migration_total_2021?: number;
  dutch_background_2021?: number;
  western_background_2021?: number;
  non_western_background_2021?: number;
  surinamese_background_2021?: number;
  antillean_background_2021?: number;
  turkish_background_2021?: number;
  moroccan_background_2021?: number;
  dutch_background_pct?: number;
  western_background_pct?: number;
  non_western_background_pct?: number;
};

type NeighborhoodResponse = {
  rows: NeighborhoodRow[];
  summary?: {
    neighborhood_count?: number;
    avg_family_share_pct?: number | null;
    avg_long_term_pct?: number | null;
    source_files?: string[];
  };
};

export default function Neighborhoods() {
  const { data, error, loading } = useCachedApi<NeighborhoodResponse>(
    "neighborhood_reports_summary",
    "/neighborhood-reports/summary",
    300_000
  );

  const rows = useMemo(() => data?.rows || [], [data]);

  const mostFamily = useMemo(
    () => [...rows]
      .filter((r) => Number.isFinite(Number(r.family_share_pct)))
      .sort((a, b) => Number(b.family_share_pct || 0) - Number(a.family_share_pct || 0))
      .slice(0, 10),
    [rows]
  );

  const mostStable = useMemo(
    () => [...rows]
      .filter((r) => Number.isFinite(Number(r.long_term_pct)))
      .sort((a, b) => Number(b.long_term_pct || 0) - Number(a.long_term_pct || 0))
      .slice(0, 10),
    [rows]
  );

  const highestTurnover = useMemo(
    () => [...rows]
      .filter((r) => Number.isFinite(Number(r.recent_mover_pct)))
      .sort((a, b) => Number(b.recent_mover_pct || 0) - Number(a.recent_mover_pct || 0))
      .slice(0, 10),
    [rows]
  );

  if (loading) return <LoadingState />;
  if (error) return <ErrorState message={String(error)} />;

  return (
    <>
      <SectionCard title="Neighborhood Demographics (Amsterdam)">
        <MetricGrid>
          <MetricCard
            label="Neighborhoods"
            value={data?.summary?.neighborhood_count ?? rows.length}
          />
          <MetricCard
            label="Avg Family Share"
            value={data?.summary?.avg_family_share_pct != null ? `${data.summary.avg_family_share_pct}%` : "—"}
          />
          <MetricCard
            label="Avg Long-term Residents"
            value={data?.summary?.avg_long_term_pct != null ? `${data.summary.avg_long_term_pct}%` : "—"}
          />
          <MetricCard
            label="Source Files"
            value={data?.summary?.source_files?.length ?? 0}
          />
        </MetricGrid>
      </SectionCard>

      <SectionCard title="Top Family-Oriented Neighborhoods">
        <DataTable
          columns={[
            { key: "neighborhood", label: "Neighborhood" },
            { key: "family_share_pct", label: "Family Share %" },
            { key: "single_share_pct", label: "Single Share %" },
            { key: "household_total", label: "Total Households" },
          ]}
          rows={mostFamily as Record<string, unknown>[]}
        />
      </SectionCard>

      <SectionCard title="Top Stable Neighborhoods (20+ years residents)">
        <DataTable
          columns={[
            { key: "neighborhood", label: "Neighborhood" },
            { key: "long_term_pct", label: "Long-term %" },
            { key: "recent_mover_pct", label: "Recent Movers %" },
            { key: "residents_total", label: "Residents" },
          ]}
          rows={mostStable as Record<string, unknown>[]}
        />
      </SectionCard>

      <SectionCard title="Highest Turnover Neighborhoods">
        <DataTable
          columns={[
            { key: "neighborhood", label: "Neighborhood" },
            { key: "recent_mover_pct", label: "Recent Movers %" },
            { key: "long_term_pct", label: "Long-term %" },
            { key: "residents_total", label: "Residents" },
          ]}
          rows={highestTurnover as Record<string, unknown>[]}
        />
      </SectionCard>

      <SectionCard title="All Neighborhoods Table">
        <DataTable
          maxHeight={520}
          columns={[
            { key: "code", label: "Code" },
            { key: "neighborhood", label: "Neighborhood" },
            { key: "family_share_pct", label: "Family %" },
            { key: "single_share_pct", label: "Single %" },
            { key: "recent_mover_pct", label: "Recent Movers %" },
            { key: "long_term_pct", label: "Long-term %" },
            { key: "netherlands_nationality_pct", label: "Dutch Nationality %" },
            { key: "residents_total", label: "Residents" },
          ]}
          rows={rows as Record<string, unknown>[]}
        />
      </SectionCard>

      <SectionCard title="Neighborhood and Ethnicity (Migration Background)">
        <DataTable
          maxHeight={520}
          columns={[
            { key: "neighborhood", label: "Neighborhood" },
            { key: "migration_total_2021", label: "Total (2021)" },
            { key: "dutch_background_2021", label: "Dutch BG" },
            { key: "western_background_2021", label: "Western BG" },
            { key: "non_western_background_2021", label: "Non-Western BG" },
            { key: "surinamese_background_2021", label: "Surinamese" },
            { key: "antillean_background_2021", label: "Antillean" },
            { key: "turkish_background_2021", label: "Turkish" },
            { key: "moroccan_background_2021", label: "Moroccan" },
            { key: "dutch_background_pct", label: "Dutch BG %" },
            { key: "non_western_background_pct", label: "Non-Western BG %" },
          ]}
          rows={rows as Record<string, unknown>[]}
        />
      </SectionCard>
    </>
  );
}
