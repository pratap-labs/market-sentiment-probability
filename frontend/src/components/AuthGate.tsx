import { useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { API_BASE_URL } from "../api/client";
import { useNotifications } from "../state/NotificationContext";

export default function AuthGate({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { notify } = useNotifications();

  useEffect(() => {
    const check = async () => {
      if (location.pathname === "/login") return;
      try {
        const res = await fetch(`${API_BASE_URL}/auth/status`);
        if (!res.ok) throw new Error("auth status failed");
        const data = await res.json();
        if (!data.has_token || data.token_expired) {
          notify({ type: "warning", title: "Auth Required", message: "Session expired. Please login again." });
          navigate("/login");
        }
      } catch {
        notify({ type: "warning", title: "Auth Required", message: "Unable to verify session. Please login." });
        navigate("/login");
      }
    };
    check();
  }, [location.pathname, navigate, notify]);

  useEffect(() => {
    const handler = () => {
      notify({ type: "warning", title: "Unauthorized", message: "Kite session invalid. Redirecting to login." });
      navigate("/login");
    };
    window.addEventListener("auth:unauthorized", handler as EventListener);
    return () => window.removeEventListener("auth:unauthorized", handler as EventListener);
  }, [navigate, notify]);

  return <>{children}</>;
}
