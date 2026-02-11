import { PropsWithChildren } from "react";

export default function MetricGrid({ children }: PropsWithChildren) {
  return <div className="metric-grid">{children}</div>;
}
