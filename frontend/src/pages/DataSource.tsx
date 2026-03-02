import { useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import { useCachedApi } from "../hooks/useCachedApi";
import { API_BASE_URL } from "../api/client";
import { useNotifications } from "../state/NotificationContext";

type CacheStatus = Record<string, {
  name?: string;
  exists: boolean;
  updated_at: string | null;
  size_bytes: number;
  latest_data_date?: string | null;
}>;
type RowsResponse = { rows: Record<string, unknown>[] };
type OptionsResponse = { ce: Record<string, unknown>[]; pe: Record<string, unknown>[] };

export default function DataSource() {
  const [refreshKey, setRefreshKey] = useState(0);
  const [refreshing, setRefreshing] = useState<string | null>(null);
  const { notify } = useNotifications();
  const { data, error, loading } = useCachedApi<CacheStatus>(
    `cache_status_${refreshKey}`,
    `/data-source/cache-status${refreshKey ? `?t=${refreshKey}` : ""}`,
    60_000
  );
  const ohlcv = useCachedApi<RowsResponse>("nifty_ohlcv_preview", "/data-source/preview/nifty_ohlcv?limit=20", 60_000);
  const futures = useCachedApi<RowsResponse>("nifty_futures_preview", "/data-source/preview/nifty_futures?limit=20", 60_000);
  const optionsCe = useCachedApi<RowsResponse>("nifty_options_ce_preview", "/data-source/preview/nifty_options_ce?limit=20", 60_000);
  const optionsPe = useCachedApi<RowsResponse>("nifty_options_pe_preview", "/data-source/preview/nifty_options_pe?limit=20", 60_000);

  const rows = useMemo(() => (data
    ? Object.entries(data).map(([key, val]) => ({
        name: key,
        exists: val.exists ? "Yes" : "No",
        updated_at: val.updated_at || "N/A",
        size_bytes: val.size_bytes,
        latest_data_date: val.latest_data_date || "N/A"
      }))
    : []), [data]);

  const buildColumns = (sample: Record<string, unknown> | undefined) => {
    if (!sample) return [];
    return Object.keys(sample).slice(0, 8).map((key) => ({ key, label: key }));
  };

  const ohlcvColumns = useMemo(
    () => buildColumns(ohlcv.data?.rows?.[0]),
    [ohlcv.data]
  );
  const futuresColumns = useMemo(
    () => buildColumns(futures.data?.rows?.[0]),
    [futures.data]
  );
  const ceColumns = useMemo(
    () => buildColumns(optionsCe.data?.rows?.[0]),
    [optionsCe.data]
  );
  const peColumns = useMemo(
    () => buildColumns(optionsPe.data?.rows?.[0]),
    [optionsPe.data]
  );

  return (
    <>
      {error || ohlcv.error || futures.error || optionsCe.error || optionsPe.error ? (
        <ErrorState message={String(error || ohlcv.error || futures.error || optionsCe.error || optionsPe.error)} />
      ) : null}
      <SectionCard title="Cache Status">
        {loading || !data ? <LoadingState /> : (
          <DataTable
            columns={[
              { key: "name", label: "Dataset" },
              { key: "exists", label: "Exists" },
              { key: "updated_at", label: "Updated" },
              { key: "latest_data_date", label: "Latest Data Date" },
              { key: "size_bytes", label: "Size (bytes)" },
              {
                key: "action",
                label: "Action",
                render: (row) => (
                  <button
                    className="control-input"
                    disabled={refreshing === String(row.name)}
                    onClick={async () => {
                      const name = String(row.name);
                      setRefreshing(name);
                      try {
                        const res = await fetch(`${API_BASE_URL}/data-source/refresh/${name}`, {
                          method: "POST"
                        });
                        if (!res.ok) {
                          const txt = await res.text();
                          throw new Error(txt || "Refresh failed");
                        }
                        notify({ type: "success", title: "Updated", message: name });
                        setRefreshKey(Date.now());
                      } catch (err) {
                        notify({ type: "error", title: "Refresh failed", message: String(err) });
                      } finally {
                        setRefreshing(null);
                      }
                    }}
                  >
                    {refreshing === String(row.name)
                      ? "Updating..."
                      : String(row.name) === "participants"
                        ? "Update to Today"
                        : "Fetch Latest"}
                  </button>
                )
              }
            ]}
            rows={rows}
          />
        )}
      </SectionCard>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">NIFTY OHLCV (Debug)</div>
        </summary>
        <div className="section-body">
          {ohlcv.loading ? <LoadingState /> : (
            <DataTable
              maxHeight={400}
              columns={ohlcvColumns}
              rows={ohlcv.data?.rows || []}
            />
          )}
        </div>
      </details>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">NIFTY Futures (Debug)</div>
        </summary>
        <div className="section-body">
          {futures.loading ? <LoadingState /> : (
            <DataTable
              maxHeight={400}
              columns={futuresColumns}
              rows={futures.data?.rows || []}
            />
          )}
        </div>
      </details>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">NIFTY Options CE (Debug)</div>
        </summary>
        <div className="section-body">
          {optionsCe.loading ? <LoadingState /> : (
            <DataTable
              maxHeight={400}
              columns={ceColumns}
              rows={optionsCe.data?.rows || []}
            />
          )}
        </div>
      </details>

      <details className="card collapsible-card">
        <summary className="section-title">
          <div className="section-title-text">NIFTY Options PE (Debug)</div>
        </summary>
        <div className="section-body">
          {optionsPe.loading ? <LoadingState /> : (
            <DataTable
              maxHeight={400}
              columns={peColumns}
              rows={optionsPe.data?.rows || []}
            />
          )}
        </div>
      </details>
    </>
  );
}
