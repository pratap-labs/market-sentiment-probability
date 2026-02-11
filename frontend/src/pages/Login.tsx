import { useEffect, useState } from "react";
import SectionCard from "../components/SectionCard";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import { apiFetch, API_BASE_URL } from "../api/client";

type Health = { status: string; timestamp: string };

type KiteDiag = {
  env_api_key: boolean;
  env_access_token: boolean;
  cache_file_exists: boolean;
  cached_valid: boolean;
  cached_saved_at: string | null;
  token_expired: boolean;
};
type AuthStatus = { has_token: boolean; token_expired: boolean; saved_at: string | null };

export default function Login() {
  const [health, setHealth] = useState<Health | null>(null);
  const [kite, setKite] = useState<KiteDiag | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null);

  useEffect(() => {
    let mounted = true;
    Promise.all([
      apiFetch<Health>("/health"),
      apiFetch<KiteDiag>("/diagnostics/kite"),
      apiFetch<AuthStatus>("/auth/status")
    ])
      .then(([h, k, a]) => {
        if (!mounted) return;
        setHealth(h);
        setKite(k);
        setAuthStatus(a);
      })
      .catch((err) => setError(String(err)));
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <>
      <SectionCard title="API Health">
        {!health && !error ? <LoadingState /> : null}
        {error ? <ErrorState message={error} /> : null}
        {health ? (
          <div>
            Status: {health.status} | {health.timestamp}
          </div>
        ) : null}
      </SectionCard>
      <SectionCard title="Kite Diagnostics">
        {!kite && !error ? <LoadingState /> : null}
        {kite ? (
          <ul style={{ margin: 0, paddingLeft: "16px" }}>
            <li>Env API Key: {String(kite.env_api_key)}</li>
            <li>Env Access Token: {String(kite.env_access_token)}</li>
            <li>Cache File Exists: {String(kite.cache_file_exists)}</li>
            <li>Cached Valid: {String(kite.cached_valid)}</li>
            <li>Cached Saved At: {kite.cached_saved_at || "N/A"}</li>
            <li>Token Expired: {String(kite.token_expired)}</li>
          </ul>
        ) : null}
      </SectionCard>
      <SectionCard title="OAuth Login">
        <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
          <a
            href={`${API_BASE_URL}/auth/login`}
            style={{
              background: "linear-gradient(135deg, var(--gs-accent), var(--gs-accent2))",
              color: "#fff",
              padding: "8px 14px",
              borderRadius: "10px",
              fontWeight: 600
            }}
          >
            Connect Kite
          </a>
          {authStatus ? (
            <div style={{ color: authStatus.has_token && !authStatus.token_expired ? "var(--gs-success)" : "var(--gs-warning)" }}>
              {authStatus.has_token && !authStatus.token_expired ? "Authenticated" : "Not authenticated"}
            </div>
          ) : null}
        </div>
      </SectionCard>
    </>
  );
}
