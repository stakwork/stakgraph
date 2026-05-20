// SPA entry. Mounts <App/> into #root. Everything else lives in app.tsx.
//
// Before mounting, we check for a `?ticket=<value>` query param. If
// present, Hive has handed us a one-shot bootstrap ticket via an
// iframe src; we redeem it for a session cookie and strip the query
// so a refresh doesn't try to replay an already-dead ticket. On any
// failure we fall through to <App/>, which will hit /me, get 401,
// and redirect to the login page — preserving the direct-access
// fallback for ops/debugging.

import { render } from "preact";

import "./styles/base.css";
import "./styles/components.css";
import "uplot/dist/uPlot.min.css";

import { App } from "./app";
import { apiPost } from "./api/client";

async function redeemTicketIfPresent() {
  const url = new URL(window.location.href);
  const ticket = url.searchParams.get("ticket");
  if (!ticket) return;

  // Strip the ticket from the visible URL first so a refresh or a
  // bookmark can't accidentally try to re-redeem it. We do this
  // BEFORE the network call: even if the POST fails, the URL is
  // clean for the inevitable retry / login-fallback.
  url.searchParams.delete("ticket");
  window.history.replaceState({}, "", url.pathname + url.search + url.hash);

  try {
    await apiPost<{ user: string }>("/auth/redeem", { ticket });
  } catch {
    // Swallow — the SPA will boot, hit /me, get 401, and bounce to
    // /login. That's the right UX for a stale / re-used ticket.
  }
}

void (async () => {
  await redeemTicketIfPresent();
  const root = document.getElementById("root");
  if (!root) throw new Error("missing #root");
  render(<App />, root);
})();
