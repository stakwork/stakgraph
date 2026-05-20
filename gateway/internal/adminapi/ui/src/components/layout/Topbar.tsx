import { useLocation } from "wouter-preact";

import { apiPost, UnauthorizedError } from "../../api/client";
import { useMe } from "../../api/queries";
import { ChevronRightIcon } from "../icons";

interface TopbarProps {
  /** True ⇒ render the expand button on the left so the user has a
   *  way to bring the sidebar back. Owned by Shell. */
  sidebarCollapsed?: boolean;
  onExpandSidebar?: () => void;
}

export function Topbar({ sidebarCollapsed, onExpandSidebar }: TopbarProps) {
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
      <div style="display:flex;align-items:center;gap:12px">
        {sidebarCollapsed && onExpandSidebar ? (
          <>
            {/* When collapsed, the topbar absorbs the brand from the
             *  (now-hidden) sidebar so the product name stays visible.
             *  Expand button sits to its right so the affordance reads
             *  "this is the gateway; click chevron to bring nav back." */}
            <div class="brand brand-inline">
              <span class="brand-dot" />
              <span class="brand-label">Agent Mothership</span>
            </div>
            <button
              type="button"
              class="sidebar-toggle"
              onClick={onExpandSidebar}
              aria-label="Expand sidebar"
              title="Expand sidebar"
            >
              <ChevronRightIcon />
            </button>
          </>
        ) : (
          <div class="text-muted mono" style="font-size: 12px">
            {/* Reserved for future breadcrumbs / time-window picker. */}
          </div>
        )}
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
