// App = QueryClient + global error handler + route table.
//
// Everything cross-cutting (the 401 redirect, the polling defaults,
// the route base path) lives here in one screenful. Phase 8's whole
// app is six pages; splitting this across three files would hide
// the wiring behind imports.

import { useMemo } from "preact/hooks";
import {
  QueryCache,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import { Route, Router, Switch, useLocation } from "wouter-preact";

import { UnauthorizedError } from "./api/client";
import { Shell } from "./components/layout/Shell";
import { AgentDetail } from "./pages/AgentDetail";
import { Agents } from "./pages/Agents";
import { Dashboard } from "./pages/Dashboard";
import { Login } from "./pages/Login";
import { NotFound } from "./pages/NotFound";
import { People } from "./pages/People";
import { RunDetail } from "./pages/RunDetail";
import { UserDetail } from "./pages/UserDetail";

// Wouter mounts client-side routes relative to <Router base="…"/>.
// All SPA paths live under /_plugin/ui — matches the Vite `base`
// and the Go-side //go:embed handler.
const BASE = "/_plugin/ui";

// AppShell is the inner component that has access to wouter's
// `useLocation` (which only works inside a Router). The QueryClient
// is created lazily via useMemo so the global 401 handler captures
// the live setLocation closure.
function AppShell() {
  const [location, setLocation] = useLocation();

  const qc = useMemo(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // Phase 8 "Data-fetching contract":
            //   staleTime: 0   – always refetch on focus
            //   gcTime: 5 min  – cache after unmount
            //   retry: 1       – modest retry on GETs
            staleTime: 0,
            gcTime: 5 * 60_000,
            retry: 1,
            refetchOnWindowFocus: true,
          },
        },
        queryCache: new QueryCache({
          onError: (err) => {
            // 401 short-circuits to a redirect, captured once
            // here rather than in every page's error branch.
            if (err instanceof UnauthorizedError) {
              if (location.startsWith("/login")) return;
              setLocation(`/login?next=${encodeURIComponent(location || "/")}`);
            }
          },
        }),
      }),
    // Recreating the client on every location change would tear
    // down in-flight queries; we want it stable for the SPA's
    // lifetime. The redirect closure reads `location` via the
    // QueryCache's `onError`, which fires per-error and re-reads
    // the live closure each time.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  return (
    <QueryClientProvider client={qc}>
      <Switch>
        <Route path="/login" component={Login} />
        <Route path="/runs/:id">
          {(params) => (
            <Shell>
              <RunDetail runID={params.id} />
            </Shell>
          )}
        </Route>
        <Route path="/agents/:name">
          {(params) => (
            <Shell>
              <AgentDetail name={params.name} />
            </Shell>
          )}
        </Route>
        <Route path="/agents">
          <Shell>
            <Agents />
          </Shell>
        </Route>
        <Route path="/people/:id">
          {(params) => (
            <Shell>
              <UserDetail userID={params.id} />
            </Shell>
          )}
        </Route>
        <Route path="/people">
          <Shell>
            <People />
          </Shell>
        </Route>
        <Route path="/">
          <Shell>
            <Dashboard />
          </Shell>
        </Route>
        <Route>
          <Shell>
            <NotFound />
          </Shell>
        </Route>
      </Switch>
    </QueryClientProvider>
  );
}

export function App() {
  return (
    <Router base={BASE}>
      <AppShell />
    </Router>
  );
}
