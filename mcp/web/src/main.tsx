import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

// Strip ?token=<jwt> from the URL after load. The token was used to
// authenticate the initial HTML request (e.g. iframe embed). The SPA
// itself uses window.__AUTH_TOKEN__ (injected server-side) for API calls,
// so the URL token is no longer needed and shouldn't leak via Referer,
// history, or shared links.
if (typeof window !== "undefined") {
  const url = new URL(window.location.href);
  if (url.searchParams.has("token")) {
    url.searchParams.delete("token");
    window.history.replaceState({}, "", url.toString());
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
