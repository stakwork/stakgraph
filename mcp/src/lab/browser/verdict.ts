/**
 * `browser/verify`'s verdict, pure: what the page did → the check contract
 * strut's verify pass reads (`{ supports, content }`, strut
 * plans/job-claims.md §4), with every failure named. The service gathers
 * what happened (service.ts `verify`); this decides what it means, so the
 * wording is tested without a browser.
 *
 * The rule that makes it a check and not a capture: the page failing is a
 * RESULT. A document that answered HTTP ≥ 400, a step whose target matched
 * nothing, an expectation that does not hold — each is a line in `failed`
 * and `supports: false`. The instrument failing (no browser, a URL that
 * does not open) never reaches here: the service throws.
 */

/** A step of the declared list that failed: which, and why. */
export interface StoppedAt {
  /** 1-based. */
  step: number;
  verb: string;
  args: Record<string, unknown>;
  message: string;
}

/** One expectation, checked. */
export interface Expectation {
  /** What was expected, as a person reads it: `text "Saved"`. */
  label: string;
  holds: boolean;
  /** Why not, when it does not hold. */
  detail?: string;
}

export interface VerifyObservation {
  /** The document's HTTP status on open; 0 when nothing was fetched (a data: URL). */
  status: number;
  stopped?: StoppedAt;
  /** Where the page ended up. */
  pageUrl: string;
  expectations: Expectation[];
}

export interface Verdict {
  supports: boolean;
  /** One line a person reads: where the page got to, and what failed. */
  content: string;
  failed: string[];
}

const CONTENT_CAP = 600;

export function judgeVerify(o: VerifyObservation): Verdict {
  const failed: string[] = [];
  if (o.status >= 400) failed.push(`the page answered HTTP ${o.status}`);
  if (o.stopped) failed.push(`step ${o.stopped.step} (${stepLabel(o.stopped)}) failed: ${firstLine(o.stopped.message)}`);
  const missed = o.expectations.filter((e) => !e.holds);
  for (const e of missed) failed.push(e.detail ? `${e.label}: ${e.detail}` : e.label);

  const parts = [o.stopped ? `stopped at step ${o.stopped.step} (${stepLabel(o.stopped)}): ${firstLine(o.stopped.message)}` : `${where(o.pageUrl)} reached`];
  if (o.status >= 400) parts.push(`HTTP ${o.status}`);
  const n = o.expectations.length;
  if (n && !missed.length) parts.push(n === 1 ? "the expectation holds" : `all ${n} expectations hold`);
  if (missed.length) parts.push(`${missed.length} of ${n} expectation${n === 1 ? "" : "s"} failed: ${missed.map((e) => (e.detail ? `${e.label} ${e.detail}` : e.label)).join("; ")}`);
  return { supports: failed.length === 0, content: cap(parts.join("; "), CONTENT_CAP), failed };
}

/**
 * Where a page is, against what an author wrote: a glob when the pattern has
 * a `*` (`**` crosses `/`, `*` stays inside a path segment; the whole URL must
 * match), else a substring — the rule `browser/wait`'s `url` already follows.
 */
export function urlMatches(url: string, pattern: string): boolean {
  if (!pattern.includes("*")) return url.includes(pattern);
  let re = "";
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i]!;
    if (ch === "*") {
      if (pattern[i + 1] === "*") {
        re += ".*";
        i++;
      } else re += "[^/]*";
    } else re += ch.replace(/[.+?^${}()|[\]\\]/g, "\\$&");
  }
  return new RegExp(`^${re}$`).test(url);
}

/** `click text=Sign in` — the verb and its target, for a person. */
function stepLabel(s: StoppedAt): string {
  const a = s.args;
  const target = ["selector", "ref", "text", "url", "key", "value", "expression", "ms", "y"].map((k) => a[k]).find((v) => v !== undefined && v !== "");
  return target === undefined ? s.verb : `${s.verb} ${typeof target === "string" ? target : JSON.stringify(target)}`;
}

/** The page's path, or its host when it is the root. */
function where(pageUrl: string): string {
  try {
    const u = new URL(pageUrl);
    if (u.protocol === "data:") return "the page";
    const path = `${u.pathname}${u.search}`;
    return path && path !== "/" ? path : u.host || pageUrl;
  } catch {
    return pageUrl || "the page";
  }
}

/** Playwright's messages carry a call log after the first line. */
function firstLine(m: string): string {
  return cap(m.split("\n")[0]!.trim(), 200);
}

function cap(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}
