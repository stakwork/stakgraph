// DataTable — minimal sortable table. Used by Dashboard rankings,
// Agents list, Run detail call log.
//
// Why hand-rolled (and not @tanstack/react-table or similar): four
// pages, fixed feature set (column-click sort + a stable page-size
// cap), and a bundle budget. The whole component fits in <80 lines
// and stays readable; replace if/when feature creep makes the
// table itself the bottleneck.

import { useMemo, useState } from "preact/hooks";

import { EmptyState } from "../EmptyState";

export interface Column<Row> {
  key: string;
  header: string;
  /** Return a renderable cell for this row+column. */
  cell: (r: Row) => any;
  /** Return a comparable for sort. Omit ⇒ column is not sortable. */
  sort?: (r: Row) => string | number;
  /** Style hook: "num" right-aligns and switches to mono. */
  align?: "num";
}

interface Props<Row> {
  columns: Column<Row>[];
  rows: Row[];
  emptyMessage: string;
  /** Optional click handler for whole-row navigation. */
  onRowClick?: (r: Row) => void;
  /** Initial sort column key (default: first sortable). */
  defaultSortKey?: string;
  /** Initial sort order. */
  defaultSortOrder?: "asc" | "desc";
}

export function DataTable<Row>({
  columns,
  rows,
  emptyMessage,
  onRowClick,
  defaultSortKey,
  defaultSortOrder = "desc",
}: Props<Row>) {
  const [sortKey, setSortKey] = useState<string | undefined>(
    defaultSortKey ?? columns.find((c) => c.sort)?.key
  );
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">(defaultSortOrder);

  const sorted = useMemo(() => {
    const col = columns.find((c) => c.key === sortKey);
    if (!col || !col.sort) return rows;
    const sign = sortOrder === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = col.sort!(a);
      const bv = col.sort!(b);
      if (av < bv) return -1 * sign;
      if (av > bv) return 1 * sign;
      return 0;
    });
  }, [rows, columns, sortKey, sortOrder]);

  if (rows.length === 0) {
    return <EmptyState>{emptyMessage}</EmptyState>;
  }

  function onHeaderClick(c: Column<Row>) {
    if (!c.sort) return;
    if (c.key === sortKey) {
      setSortOrder((o) => (o === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(c.key);
      setSortOrder("desc");
    }
  }

  return (
    <div class="table-wrap">
      <table class="table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                class={c.align === "num" ? "num" : ""}
                style={c.sort ? "cursor:pointer" : ""}
                onClick={() => onHeaderClick(c)}
              >
                {c.header}
                {sortKey === c.key ? (sortOrder === "asc" ? " ▲" : " ▼") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((r, i) => (
            <tr
              key={i}
              class={onRowClick ? "row-link" : ""}
              onClick={onRowClick ? () => onRowClick(r) : undefined}
            >
              {columns.map((c) => (
                <td key={c.key} class={c.align === "num" ? "num" : ""}>
                  {c.cell(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
