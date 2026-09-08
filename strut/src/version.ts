import { createHash } from "node:crypto";

/**
 * Content hash for a workflow or step version, used purely as an internal
 * **dedup key**: identical content yields the same hash, so re-seeding
 * unchanged templates is a no-op and edited templates are detected. The hash
 * is stored in version metadata — it is NOT the user-facing version id.
 */
export function contentHash(content: string): string {
  return createHash("sha256").update(content, "utf-8").digest("hex").slice(0, 12);
}

/**
 * Allocate the next sequential, user-facing version label (`v1`, `v2`, …)
 * given the set of existing version ids. Non-`vN` ids are ignored for
 * numbering (but still counted as existing, so labels never collide).
 */
export function nextVersionLabel(existing: string[]): string {
  let max = 0;
  for (const v of existing) {
    const m = /^v(\d+)$/.exec(v);
    if (m) {
      const n = parseInt(m[1]!, 10);
      if (n > max) max = n;
    }
  }
  return `v${max + 1}`;
}
