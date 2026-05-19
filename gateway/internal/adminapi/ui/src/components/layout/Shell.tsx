// Shell = sidebar + topbar + main outlet. Every authed page wraps
// its content in <Shell>{...}</Shell>; login + standalone error
// pages bypass this.

import type { ComponentChildren } from "preact";

import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";

export function Shell({ children }: { children: ComponentChildren }) {
  return (
    <div class="shell">
      <Sidebar />
      <Topbar />
      <main class="shell-main">{children}</main>
    </div>
  );
}
