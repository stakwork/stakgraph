import { useLocation } from "wouter-preact";

import { apiPost, UnauthorizedError } from "../../api/client";
import { useMe } from "../../api/queries";

export function Topbar() {
  const me = useMe();
  const [, setLocation] = useLocation();

  async function onLogout(e: Event) {
    e.preventDefault();
    try {
      await apiPost<void>("/logout");
    } catch (err) {
      // 401 here just means the cookie was already stale — fine.
      if (!(err instanceof UnauthorizedError)) {
        // Silent: logout shouldn't block the redirect with a banner.
        // eslint-disable-next-line no-console
        console.warn("logout error", err);
      }
    }
    setLocation("/login");
  }

  const user = me.data?.user;
  return (
    <header class="shell-topbar">
      <div class="text-muted mono" style="font-size: 12px">
        {/* Reserved for future breadcrumbs / time-window picker. */}
      </div>
      <div style="display:flex;align-items:center;gap:16px">
        {user ? (
          <span class="mono text-muted" style="font-size: 12px">
            {user}
          </span>
        ) : null}
        <button type="button" class="btn" onClick={onLogout}>
          Log out
        </button>
      </div>
    </header>
  );
}
