import { useState } from "react";
import { useControls } from "../state/ControlsContext";

export default function ControlsBar() {
  const { controls } = useControls();
  const [collapsed, setCollapsed] = useState(true);

  if (!controls || !controls.content) {
    return (
      <div className="controls-bar">
        <div className="controls-header">
          <div className="controls-title">Controls</div>
          <div className="controls-summary">No controls</div>
          <div className="controls-toggle">▾</div>
        </div>
      </div>
    );
  }

  return (
    <div className="controls-bar">
      <div className="controls-header" onClick={() => setCollapsed((v) => !v)}>
        <div className="controls-title">{controls.title || "Controls"}</div>
        <div className="controls-summary">
          {collapsed ? controls.summary : null}
        </div>
        <div className="controls-toggle">{collapsed ? "▾" : "▴"}</div>
      </div>
      {!collapsed ? <div className="controls-content">{controls.content}</div> : null}
    </div>
  );
}
