import { NavLink, Outlet } from "react-router-dom";
import { spotAnalysisTabs } from "./spotAnalysisTabs";

export default function SpotAnalysis() {
  return (
    <>
      <div className="subtab-bar">
        <div className="subtab-row">
          {spotAnalysisTabs.map((tab) => (
            <NavLink
              key={tab.path}
              to={tab.path}
              end
              className={({ isActive }) => (isActive ? "subtab-link active" : "subtab-link")}
            >
              {tab.label}
            </NavLink>
          ))}
        </div>
      </div>
      <Outlet />
    </>
  );
}
