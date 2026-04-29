export const card: React.CSSProperties = {
  border: "1px solid #27272a",
  borderRadius: "12px",
  backgroundColor: "#111113",
};

export const muted: React.CSSProperties = {
  color: "#71717a",
  fontSize: "12px",
};

export function StatTile({ label, value, detail }: { label: string; value: string; detail?: string }) {
  return (
    <div style={{ ...card, padding: "14px" }}>
      <p style={{ ...muted, margin: 0 }}>{label}</p>
      <p style={{ margin: "6px 0 0 0", fontSize: "22px", fontWeight: 700, color: "#ededed" }}>
        {value}
      </p>
      {detail && <p style={{ ...muted, margin: "6px 0 0 0" }}>{detail}</p>}
    </div>
  );
}

export function FilterSelect({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        fontSize: "12px",
        borderRadius: "8px",
        padding: "8px 10px",
        backgroundColor: "#18181b",
        color: "#ededed",
        border: "1px solid #27272a",
        outline: "none",
      }}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
}

export function TableCard({
  title,
  badge,
  columns,
  rows,
}: {
  title: string;
  badge?: string;
  columns: string[];
  rows: React.ReactNode;
}) {
  return (
    <div style={card}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "8px",
          padding: "12px 14px",
          borderBottom: "1px solid #27272a",
        }}
      >
        <p style={{ margin: 0, fontSize: "13px", fontWeight: 700, color: "#ededed" }}>{title}</p>
        {badge && <span style={muted}>{badge}</span>}
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr>
              {columns.map((column) => (
                <th
                  key={column}
                  style={{
                    textAlign: "left",
                    fontSize: "11px",
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                    color: "#71717a",
                    padding: "10px 14px",
                    borderBottom: "1px solid #1f1f22",
                    whiteSpace: "nowrap",
                  }}
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>
  );
}

export function tdStyle(emphasis = false): React.CSSProperties {
  return {
    padding: "10px 14px",
    borderBottom: "1px solid #1a1a1d",
    fontSize: "12px",
    color: emphasis ? "#ededed" : "#d4d4d8",
    fontWeight: emphasis ? 600 : 400,
    whiteSpace: "nowrap",
  };
}
