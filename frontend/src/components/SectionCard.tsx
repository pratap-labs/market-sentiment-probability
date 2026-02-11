import { PropsWithChildren } from "react";

type SectionCardProps = PropsWithChildren<{ title?: string; tooltip?: string }>

export default function SectionCard({ title, tooltip, children }: SectionCardProps) {
  return (
    <section className="card">
      {title ? (
        <div className="section-title">
          <div className="section-title-text">
            <span>{title}</span>
            {tooltip ? (
              <span className="metric-help">
                ?
                <span className="metric-popover">{tooltip}</span>
              </span>
            ) : null}
          </div>
        </div>
      ) : null}
      <div className="section-body">
        {children}
      </div>
    </section>
  );
}
