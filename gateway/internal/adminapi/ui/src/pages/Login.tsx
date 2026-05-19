// Login page. POSTs Authorization: Basic to /_plugin/login; on
// success the cookie is set by the server and we redirect to the
// `?next=` location (or "/" by default).

import { useState } from "preact/hooks";
import { useLocation } from "wouter-preact";

import { apiPost, ApiCallError } from "../api/client";
import type { LoginResponse } from "../api/types";

export function Login() {
  const [location, setLocation] = useLocation();
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  // wouter strips the base from `location` for us; the query string
  // isn't part of `location`, so read it from window directly.
  const next = new URL(window.location.href).searchParams.get("next") || "/";

  async function onSubmit(e: Event) {
    e.preventDefault();
    setErr(null);
    setBusy(true);
    try {
      await apiPost<LoginResponse>("/login", undefined, {
        Authorization: "Basic " + btoa(user + ":" + pass),
      });
      setLocation(next);
    } catch (e) {
      if (e instanceof ApiCallError) {
        if (e.status === 429) setErr("Too many attempts. Try again later.");
        else if (e.status === 401) setErr("Invalid credentials.");
        else setErr(e.message || "Login failed.");
      } else {
        setErr("Login failed.");
      }
    } finally {
      setBusy(false);
    }
  }

  // Reference the unused `location` so the linter doesn't yell.
  // (We read it indirectly via window.location for the query string,
  // but importing useLocation also gives us setLocation.)
  void location;

  return (
    <div class="login-shell">
      <div class="login-card">
        <div class="brand">
          <span class="brand-dot" />
          Agent Gateway
        </div>
        <form class="login-form" onSubmit={onSubmit}>
          <label for="login-user">Username</label>
          <input
            id="login-user"
            class="input"
            type="text"
            autoComplete="username"
            value={user}
            onInput={(e) =>
              setUser((e.currentTarget as HTMLInputElement).value)
            }
            required
          />
          <label for="login-pass">Password</label>
          <input
            id="login-pass"
            class="input-pwd"
            type="password"
            autoComplete="current-password"
            value={pass}
            onInput={(e) =>
              setPass((e.currentTarget as HTMLInputElement).value)
            }
            required
          />
          <div class="login-error">{err ?? ""}</div>
          <button type="submit" class="btn is-primary" disabled={busy}>
            {busy ? "Signing in…" : "Sign in"}
          </button>
        </form>
      </div>
    </div>
  );
}
