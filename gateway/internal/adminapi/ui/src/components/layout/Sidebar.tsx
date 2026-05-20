import { Link, useLocation } from "wouter-preact";

// Sidebar nav. Phase 8 has Dashboard + People + Agents; phase 9
// grows Sessions and Config — each is just another row here.
//
// People above Agents reflects the organising principle of the
// governance plan: "every LLM call traces to a specific human."
// The humans-first axis is the primary one; agents are tools.
const NAV: { to: string; label: string }[] = [
  { to: "/", label: "Dashboard" },
  { to: "/people", label: "People" },
  { to: "/agents", label: "Agents" },
];

export function Sidebar() {
  const [loc] = useLocation();
  return (
    <aside class="shell-sidebar">
      <div class="brand">
        <span class="brand-dot" />
        Agent Gateway
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
