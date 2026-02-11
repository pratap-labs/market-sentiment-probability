import { NavLink, Outlet } from "react-router-dom";
import { riskBucketTabs } from "./riskBucketsTabs";

export default function RiskBuckets() {
  return (
    <>
      <div className="subtab-bar">
        <div className="subtab-row">
          {riskBucketTabs.map((tab) => (
            <NavLink
              key={tab.path}
              to={tab.path}
              end
              className={({ isActive }) =>
                isActive ? "subtab-link active" : "subtab-link"
              }
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
