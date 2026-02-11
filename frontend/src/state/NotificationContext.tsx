import { createContext, useContext, useMemo, useState } from "react";

export type NotificationType = "info" | "success" | "warning" | "error";

export type Notification = {
  id: string;
  type: NotificationType;
  title?: string;
  message: string;
  timeout?: number;
};

type NotificationContextValue = {
  notify: (n: Omit<Notification, "id">) => void;
  dismiss: (id: string) => void;
};

const NotificationContext = createContext<NotificationContextValue | null>(null);

export function NotificationProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<Notification[]>([]);

  const dismiss = (id: string) => {
    setItems((prev) => prev.filter((n) => n.id !== id));
  };

  const notify = (n: Omit<Notification, "id">) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const item: Notification = { id, timeout: 5000, ...n };
    setItems((prev) => [item, ...prev]);
    if (item.timeout && item.timeout > 0) {
      setTimeout(() => dismiss(id), item.timeout);
    }
  };

  const value = useMemo(() => ({ notify, dismiss }), []);

  return (
    <NotificationContext.Provider value={value}>
      {children}
      <div className="toast-stack">
        {items.map((n) => (
          <div key={n.id} className={`toast ${n.type}`}>
            <div className="toast-header">
              <div className="toast-title">{n.title || n.type.toUpperCase()}</div>
              <button className="toast-close" onClick={() => dismiss(n.id)}>×</button>
            </div>
            <div className="toast-body">{n.message}</div>
          </div>
        ))}
      </div>
    </NotificationContext.Provider>
  );
}

export function useNotifications() {
  const ctx = useContext(NotificationContext);
  if (!ctx) throw new Error("useNotifications must be used within NotificationProvider");
  return ctx;
}
