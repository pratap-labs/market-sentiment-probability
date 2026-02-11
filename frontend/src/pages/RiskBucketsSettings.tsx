import { useEffect, useMemo, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import DataTable from "../components/DataTable";
import { useCachedApi } from "../hooks/useCachedApi";
import { apiFetch } from "../api/client";
import { useNotifications } from "../state/NotificationContext";

type SettingsConfig = {
  alloc_low: number;
  alloc_med: number;
  alloc_high: number;
  portfolio_es_limit: number;
  bucket_es_limit_low: number;
  bucket_es_limit_med: number;
  bucket_es_limit_high: number;
  trade_low_max: number;
  trade_med_max: number;
  sim_days: number;
  sim_paths: number;
  iv_mode: string;
  iv_shock: number;
  gate_tail_ratio_watch: number;
  gate_tail_ratio_fail: number;
  gate_prob_loss_watch: number;
  gate_prob_loss_fail: number;
  gate_portfolio_breach_prob: number;
  gate_bucket_breach_prob_low: number;
  gate_bucket_breach_prob_med: number;
  gate_bucket_breach_prob_high: number;
  gate_p1_breach_fail_days: number;
  gate_portfolio_p10_fail_days: number;
  gate_bucket_p10_fail_days_med: number;
  gate_bucket_p10_fail_days_high: number;
};

type SettingsResponse = {
  grouped_trades: { trade_id: string; trade_es99_inr: number; expected_pnl_inr: number; bucket: string; manual_bucket: string }[];
  positions: { id: string; label: string; is_grouped: boolean }[];
  groups: { index: number; name: string; legs: string[] }[];
  overrides: Record<string, string>;
};

export default function RiskBucketsSettings() {
  const { notify } = useNotifications();
  const config = useCachedApi<{ settings: SettingsConfig }>("risk_buckets_settings_config", "/risk-buckets/settings/config", 60_000);
  const settings = useCachedApi<SettingsResponse>("risk_buckets_settings", "/risk-buckets/settings", 60_000);
  const [form, setForm] = useState<SettingsConfig | null>(null);
  const [selectedLegs, setSelectedLegs] = useState<Record<string, boolean>>({});
  const [groupName, setGroupName] = useState<string>("");

  useEffect(() => {
    if (config.data?.settings) {
      setForm(config.data.settings);
    }
  }, [config.data]);

  const groupedRows = settings.data?.grouped_trades || [];
  const positions = settings.data?.positions || [];
  const groups = settings.data?.groups || [];

  const saveConfig = async () => {
    if (!form) return;
    try {
      await apiFetch("/risk-buckets/settings/config", {
        method: "POST",
        body: { settings: form }
      });
      notify({ type: "success", message: "Settings updated." });
    } catch (err) {
      notify({ type: "error", message: String(err) });
    }
  };

  const updateOverride = async (tradeId: string, bucket: string) => {
    try {
      await apiFetch("/risk-buckets/settings/overrides", {
        method: "POST",
        body: { trade_id: tradeId, bucket }
      });
      notify({ type: "success", message: "Bucket override updated." });
    } catch (err) {
      notify({ type: "error", message: String(err) });
    }
  };

  const saveGroup = async () => {
    const legs = Object.entries(selectedLegs)
      .filter(([, v]) => v)
      .map(([k]) => k);
    if (!legs.length) {
      notify({ type: "warning", message: "Select at least one leg to group." });
      return;
    }
    try {
      await apiFetch("/risk-buckets/settings/groups", {
        method: "POST",
        body: { name: groupName || "Manual Trade", legs }
      });
      notify({ type: "success", message: "Trade group saved." });
      setGroupName("");
      setSelectedLegs({});
    } catch (err) {
      notify({ type: "error", message: String(err) });
    }
  };

  const removeGroup = async (index: number) => {
    try {
      await apiFetch(`/risk-buckets/settings/groups/${index}`, { method: "DELETE" });
      notify({ type: "info", message: "Trade group removed." });
    } catch (err) {
      notify({ type: "error", message: String(err) });
    }
  };

  const clearGroups = async () => {
    try {
      await apiFetch("/risk-buckets/settings/groups", { method: "DELETE" });
      notify({ type: "info", message: "All trade groups deleted." });
    } catch (err) {
      notify({ type: "error", message: String(err) });
    }
  };

  const configLoading = config.loading || !form;

  return (
    <>
      {config.error || settings.error ? <ErrorState message={String(config.error || settings.error)} /> : null}

      <SectionCard title="Thresholds & Configuration">
        {configLoading ? <LoadingState /> : (
          <>
            <details className="control-panel" open>
              <summary>Allocations & Limits</summary>
              <div className="control-grid">
                <label className="control-field">
                  <span className="control-label">Alloc Low %</span>
                  <input className="control-input" type="number" value={form.alloc_low} onChange={(e) => setForm({ ...form, alloc_low: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Alloc Med %</span>
                  <input className="control-input" type="number" value={form.alloc_med} onChange={(e) => setForm({ ...form, alloc_med: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Alloc High %</span>
                  <input className="control-input" type="number" value={form.alloc_high} onChange={(e) => setForm({ ...form, alloc_high: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Portfolio ES99 Limit %</span>
                  <input className="control-input" type="number" value={form.portfolio_es_limit} onChange={(e) => setForm({ ...form, portfolio_es_limit: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Bucket ES99 Low %</span>
                  <input className="control-input" type="number" value={form.bucket_es_limit_low} onChange={(e) => setForm({ ...form, bucket_es_limit_low: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Bucket ES99 Med %</span>
                  <input className="control-input" type="number" value={form.bucket_es_limit_med} onChange={(e) => setForm({ ...form, bucket_es_limit_med: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Bucket ES99 High %</span>
                  <input className="control-input" type="number" value={form.bucket_es_limit_high} onChange={(e) => setForm({ ...form, bucket_es_limit_high: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Trade Low Max %</span>
                  <input className="control-input" type="number" value={form.trade_low_max} onChange={(e) => setForm({ ...form, trade_low_max: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Trade Med Max %</span>
                  <input className="control-input" type="number" value={form.trade_med_max} onChange={(e) => setForm({ ...form, trade_med_max: Number(e.target.value) })} />
                </label>
              </div>
            </details>

            <details className="control-panel">
              <summary>Forward Simulation</summary>
              <div className="control-grid">
                <label className="control-field">
                  <span className="control-label">Sim Days</span>
                  <input className="control-input" type="number" value={form.sim_days} onChange={(e) => setForm({ ...form, sim_days: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Sim Paths</span>
                  <input className="control-input" type="number" value={form.sim_paths} onChange={(e) => setForm({ ...form, sim_paths: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">IV Mode</span>
                  <select className="control-input" value={form.iv_mode} onChange={(e) => setForm({ ...form, iv_mode: e.target.value })}>
                    <option value="IV Flat">IV Flat</option>
                    <option value="IV Up Shock">IV Up Shock</option>
                    <option value="IV Down Shock">IV Down Shock</option>
                  </select>
                </label>
                <label className="control-field">
                  <span className="control-label">IV Shock (pts)</span>
                  <input className="control-input" type="number" value={form.iv_shock} onChange={(e) => setForm({ ...form, iv_shock: Number(e.target.value) })} />
                </label>
              </div>
            </details>

            <details className="control-panel">
              <summary>Gate Thresholds</summary>
              <div className="control-grid">
                <label className="control-field">
                  <span className="control-label">Gate Tail Ratio Watch</span>
                  <input className="control-input" type="number" value={form.gate_tail_ratio_watch} onChange={(e) => setForm({ ...form, gate_tail_ratio_watch: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Tail Ratio Fail</span>
                  <input className="control-input" type="number" value={form.gate_tail_ratio_fail} onChange={(e) => setForm({ ...form, gate_tail_ratio_fail: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Prob Loss Watch %</span>
                  <input className="control-input" type="number" value={form.gate_prob_loss_watch} onChange={(e) => setForm({ ...form, gate_prob_loss_watch: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Prob Loss Fail %</span>
                  <input className="control-input" type="number" value={form.gate_prob_loss_fail} onChange={(e) => setForm({ ...form, gate_prob_loss_fail: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Portfolio Breach %</span>
                  <input className="control-input" type="number" value={form.gate_portfolio_breach_prob} onChange={(e) => setForm({ ...form, gate_portfolio_breach_prob: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Bucket Breach Low %</span>
                  <input className="control-input" type="number" value={form.gate_bucket_breach_prob_low} onChange={(e) => setForm({ ...form, gate_bucket_breach_prob_low: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Bucket Breach Med %</span>
                  <input className="control-input" type="number" value={form.gate_bucket_breach_prob_med} onChange={(e) => setForm({ ...form, gate_bucket_breach_prob_med: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Bucket Breach High %</span>
                  <input className="control-input" type="number" value={form.gate_bucket_breach_prob_high} onChange={(e) => setForm({ ...form, gate_bucket_breach_prob_high: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate P1 Breach Fail Days</span>
                  <input className="control-input" type="number" value={form.gate_p1_breach_fail_days} onChange={(e) => setForm({ ...form, gate_p1_breach_fail_days: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Portfolio P10 Fail Days</span>
                  <input className="control-input" type="number" value={form.gate_portfolio_p10_fail_days} onChange={(e) => setForm({ ...form, gate_portfolio_p10_fail_days: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Bucket P10 Fail Med Days</span>
                  <input className="control-input" type="number" value={form.gate_bucket_p10_fail_days_med} onChange={(e) => setForm({ ...form, gate_bucket_p10_fail_days_med: Number(e.target.value) })} />
                </label>
                <label className="control-field">
                  <span className="control-label">Gate Bucket P10 Fail High Days</span>
                  <input className="control-input" type="number" value={form.gate_bucket_p10_fail_days_high} onChange={(e) => setForm({ ...form, gate_bucket_p10_fail_days_high: Number(e.target.value) })} />
                </label>
              </div>
            </details>

            <div style={{ marginTop: "12px" }}>
              <button className="control-input" style={{ width: "160px" }} onClick={saveConfig}>Save Settings</button>
            </div>
          </>
        )}
      </SectionCard>

      <SectionCard title="Manual Bucket Assignment (Grouped Trades)">
        {settings.loading ? <LoadingState /> : (
          <details className="control-panel" open>
            <summary>Overrides</summary>
            <DataTable
              columns={[
                { key: "trade_id", label: "Trade" },
                { key: "trade_es99_inr", label: "ES99 (₹)" },
                { key: "expected_pnl_inr", label: "Expected PnL (₹)" },
                { key: "bucket", label: "Bucket" },
                { key: "manual_bucket", label: "Manual Override" }
              ]}
              rows={groupedRows.map((row) => ({
                ...row,
                manual_bucket: (
                  <select
                    className="control-input"
                    value={row.manual_bucket || "Auto"}
                    onChange={(e) => updateOverride(row.trade_id, e.target.value)}
                  >
                    <option value="Auto">Auto</option>
                    <option value="Low">Low</option>
                    <option value="Med">Med</option>
                    <option value="High">High</option>
                  </select>
                )
              }))}
            />
          </details>
        )}
      </SectionCard>

      <SectionCard title="Trade Grouping (Manual)">
        {settings.loading ? <LoadingState /> : (
          <>
            <div style={{ marginBottom: "10px", color: "var(--gs-muted)" }}>
              Select legs to group (grouped legs are locked).
            </div>
            <details className="control-panel" open>
              <summary>Build Groups</summary>
              <div className="control-grid">
                {positions.map((p) => (
                  <label key={p.id} className="control-field control-inline">
                    <input
                      type="checkbox"
                      disabled={p.is_grouped}
                      checked={!!selectedLegs[p.id]}
                      onChange={(e) => setSelectedLegs((prev) => ({ ...prev, [p.id]: e.target.checked }))}
                    />
                    <span className="control-label">{p.label}</span>
                  </label>
                ))}
              </div>
              <div style={{ display: "flex", gap: "10px", marginTop: "12px", alignItems: "center" }}>
                <input
                  className="control-input"
                  placeholder="Trade name"
                  value={groupName}
                  onChange={(e) => setGroupName(e.target.value)}
                />
                <button className="control-input" style={{ width: "160px" }} onClick={saveGroup}>Save Trade Group</button>
                {groups.length ? <button className="control-input" style={{ width: "160px" }} onClick={clearGroups}>Delete All Groups</button> : null}
              </div>
            </details>
            {groups.length ? (
              <details className="control-panel">
                <summary>Saved Groups</summary>
                <DataTable
                  columns={[
                    { key: "name", label: "Group" },
                    { key: "legs", label: "Legs" },
                    { key: "actions", label: "Actions" }
                  ]}
                  rows={groups.map((g) => ({
                    name: g.name,
                    legs: (g.legs || []).join(", "),
                    actions: (
                      <button className="control-input" onClick={() => removeGroup(g.index)}>Remove</button>
                    )
                  }))}
                />
              </details>
            ) : null}
          </>
        )}
      </SectionCard>
    </>
  );
}
