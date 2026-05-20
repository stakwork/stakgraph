// Shell = sidebar + topbar + main outlet. Every authed page wraps
// its content in <Shell>{...}</Shell>; login + standalone error
// pages bypass this.
//
// Sidebar-collapse state lives here so both <Sidebar> (renders the
// "collapse" chevron next to the brand) and <Topbar> (renders the
// "expand" chevron when the sidebar is hidden) can drive the same
// toggle. Persisted to localStorage so a refresh keeps the operator's
// chosen layout — the Canvas page in particular wants the full width
// for the flowchart-to-come.

import type { ComponentChildren } from "preact";
import { useCallback, useEffect, useState } from "preact/hooks";

import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

const STORAGE_KEY = "ui.sidebar.collapsed";

function readInitialCollapsed(): boolean {
  // Guard SSR / non-browser contexts even though we never run there
  // — the cost is one typeof check and it keeps the import safe to
  // pull in from tests that stub window away.
  if (typeof window === "undefined") return false;
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function Shell({ children }: { children: ComponentChildren }) {
  const [collapsed, setCollapsed] = useState<boolean>(readInitialCollapsed);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, collapsed ? "1" : "0");
    } catch {
      // localStorage unavailable (privacy mode, etc.) — non-fatal,
      // the toggle still works in-memory.
    }
  }, [collapsed]);

  const toggle = useCallback(() => setCollapsed((v) => !v), []);

  return (
    <div class={"shell" + (collapsed ? " is-sidebar-collapsed" : "")}>
      {collapsed ? null : <Sidebar onCollapse={toggle} />}
      <Topbar sidebarCollapsed={collapsed} onExpandSidebar={toggle} />
      <main class="shell-main">{children}</main>
    </div>
  );
}
