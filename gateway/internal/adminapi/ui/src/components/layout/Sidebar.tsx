import { Link, useLocation } from "wouter-preact";

import { ChevronLeftIcon } from "../icons";

// Sidebar nav. Canvas is the landing page; Dashboard, People and
// Agents are the phase-8/9 views. Phase 9's Sessions and Config
// pages are still unbuilt — when they land, each is one more row.
//
// People above Agents reflects the organising principle of the
// governance plan: "every LLM call traces to a specific human."
// The humans-first axis is the primary one; agents are tools.
const NAV: { to: string; label: string }[] = [
  { to: "/", label: "Canvas" },
  { to: "/dashboard", label: "Dashboard" },
  { to: "/people", label: "People" },
  { to: "/agents", label: "Agents" },
];

export function Sidebar({ onCollapse }: { onCollapse?: () => void }) {
  const [loc] = useLocation();
  return (
    <aside class="shell-sidebar">
      <div class="brand">
        <span class="brand-dot" />
        <span class="brand-label">Agent Mothership</span>
        {onCollapse ? (
          <button
            type="button"
            class="sidebar-toggle"
            onClick={onCollapse}
            aria-label="Collapse sidebar"
            title="Collapse sidebar"
          >
            <ChevronLeftIcon />
          </button>
        ) : null}
      </div>
      <nav class="nav">
        {NAV.map((n) => {
          // Active match: exact for "/" so AgentDetail doesn't
          // light up Dashboard; prefix-match for everything else.
          const active = n.to === "/" ? loc === "/" : loc.startsWith(n.to);
          return (
            <Link
              key={n.to}
              href={n.to}
              class={"nav-item" + (active ? " is-active" : "")}
            >
              {n.label}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
