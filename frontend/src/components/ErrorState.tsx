import { useEffect, useRef } from "react";
import { useNotifications } from "../state/NotificationContext";

export default function ErrorState({ message }: { message: string }) {
  const { notify } = useNotifications();
  const last = useRef<string>("");

  useEffect(() => {
    if (!message) return;
    if (last.current === message) return;
    last.current = message;
    notify({ type: "error", title: "Error", message, timeout: 10000 });
  }, [message, notify]);

  return null;
}
