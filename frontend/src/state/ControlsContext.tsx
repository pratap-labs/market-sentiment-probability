import { createContext, useCallback, useContext, useMemo, useState } from "react";

export type ControlsConfig = {
  key?: string;
  title?: string;
  summary?: React.ReactNode;
  content?: React.ReactNode;
};

type ControlsContextValue = {
  setControls: (cfg: ControlsConfig | null) => void;
  controls: ControlsConfig | null;
};

const ControlsContext = createContext<ControlsContextValue | null>(null);

export function ControlsProvider({ children }: { children: React.ReactNode }) {
  const [controls, setControlsState] = useState<ControlsConfig | null>(null);

  const setControls = useCallback((cfg: ControlsConfig | null) => {
    setControlsState((prev) => {
      if (cfg && prev?.key && cfg.key === prev.key) {
        return prev;
      }
      return cfg;
    });
  }, []);

  const value = useMemo(() => ({ controls, setControls }), [controls, setControls]);

  return <ControlsContext.Provider value={value}>{children}</ControlsContext.Provider>;
}

export function useControls() {
  const ctx = useContext(ControlsContext);
  if (!ctx) throw new Error("useControls must be used within ControlsProvider");
  return ctx;
}
