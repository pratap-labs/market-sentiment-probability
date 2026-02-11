type Column = {
  key: string;
  label: string;
  format?: (value: unknown) => string;
  tone?: "positive" | "negative" | "neutral";
  render?: (row: Record<string, unknown>) => React.ReactNode;
};

type DataTableProps = {
  columns: Column[];
  rows: Record<string, unknown>[];
  maxHeight?: number;
};

export default function DataTable({ columns, rows, maxHeight }: DataTableProps) {
  if (!rows.length) return <div>No data.</div>;
  const style = maxHeight ? { maxHeight, overflow: "auto" } : undefined;
  return (
    <div className="table-wrap" style={style}>
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.key}>{col.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {columns.map((col) => (
                <td key={col.key}>
                  {(() => {
                    if (col.render) return col.render(row);
                    const raw = row[col.key];
                    const formatted = col.format ? col.format(raw) : String(raw ?? "—");
                    const key = col.key.toLowerCase();
                    const isPnl = key.includes("pnl") || key.includes("loss");
                    if (!isPnl && !col.tone) return formatted;
                    if (col.tone) {
                      return <span className={col.tone}>{formatted}</span>;
                    }
                    const num = Number(raw);
                    if (!Number.isFinite(num)) return formatted;
                    const tone = num >= 0 ? "positive" : "negative";
                    return <span className={tone}>{formatted}</span>;
                  })()}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
