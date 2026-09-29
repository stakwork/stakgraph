/**
 * Settle: after a navigation and after each step — network
 * idle capped at a few seconds (HMR sockets and polling pages never go
 * idle), then `document.fonts.ready`, then two `requestAnimationFrame`
 * turns, then a short fixed settle plus the caller's `wait`. Fonts-ready
 * keeps FOUT out of the frame; the rAF turns keep canvas and animation
 * pages from being shot mid-load. Every phase is best-effort: a page that
 * never reaches one is still shot.
 */
import type { Page } from "playwright-core";

export const NETWORK_IDLE_MS = 3000;
export const FIXED_SETTLE_MS = 150;
const IN_PAGE_MS = 2000;

export async function settle(page: Page, extraMs = 0): Promise<void> {
  await page.waitForLoadState("networkidle", { timeout: NETWORK_IDLE_MS }).catch(() => {});
  await bounded(
    page.evaluate(() => (document as unknown as { fonts?: { ready: Promise<unknown> } }).fonts?.ready.then(() => null) ?? null),
    IN_PAGE_MS,
  );
  await bounded(
    page.evaluate(() => new Promise<null>((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(null))))),
    IN_PAGE_MS,
  );
  await sleep(FIXED_SETTLE_MS + Math.max(0, extraMs));
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

/** Await `p` for at most `ms`; a rejection or a timeout is ignored (a step's
 *  own timeout + the run budget are the real caps — see service.ts). */
async function bounded(p: Promise<unknown>, ms: number): Promise<void> {
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<void>((r) => (timer = setTimeout(r, ms)));
  await Promise.race([p.then(() => undefined, () => undefined), timeout]);
  clearTimeout(timer);
}
