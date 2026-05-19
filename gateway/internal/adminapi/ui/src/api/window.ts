// Shared helper: Window label -> seconds. Kept in /api/ (not /pages)
// because the backend's parseWindow is the authority on this set; if
// it grows a new option, this mapping needs to grow with it.

import type { Window } from "./types";

export function windowToSeconds(w: Window): number {
  switch (w) {
    case "1h":
      return 3600;
    case "6h":
      return 6 * 3600;
    case "24h":
      return 24 * 3600;
    case "7d":
      return 7 * 24 * 3600;
    case "30d":
      return 30 * 24 * 3600;
  }
}
