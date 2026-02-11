import { useState } from "react";

export default function AuthBar() {
  const [apiKey, setApiKey] = useState(localStorage.getItem("kite_api_key") || "");
  const [accessToken, setAccessToken] = useState(localStorage.getItem("kite_access_token") || "");

  function save() {
    localStorage.setItem("kite_api_key", apiKey);
    localStorage.setItem("kite_access_token", accessToken);
    window.location.reload();
  }

  return (
    <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
      <input
        value={apiKey}
        onChange={(e) => setApiKey(e.target.value)}
        placeholder="Kite API Key"
        style={{ background: "var(--gs-surface)", border: "1px solid var(--gs-border)", color: "var(--gs-text)", padding: "6px 10px", borderRadius: "8px" }}
      />
      <input
        value={accessToken}
        onChange={(e) => setAccessToken(e.target.value)}
        placeholder="Kite Access Token"
        style={{ background: "var(--gs-surface)", border: "1px solid var(--gs-border)", color: "var(--gs-text)", padding: "6px 10px", borderRadius: "8px" }}
      />
      <button
        onClick={save}
        style={{ background: "linear-gradient(135deg, var(--gs-accent), var(--gs-accent2))", color: "#fff", border: "none", padding: "6px 12px", borderRadius: "8px", cursor: "pointer" }}
      >
        Save
      </button>
    </div>
  );
}
