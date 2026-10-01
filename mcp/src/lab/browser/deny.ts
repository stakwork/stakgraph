/**
 * The URL denylist: what `browser/goto` (and `open`,
 * `capture`) refuse, so a model-driven browser cannot read the host's own
 * filesystem or a cloud metadata endpoint. `BROWSER_URL_DENY` is a
 * comma-separated list of URL schemes (`file:`), IPv4/IPv6 CIDRs
 * (`169.254.0.0/16`, `fd00::/8`) and exact hostnames
 * (`metadata.google.internal`).
 *
 * Checked before a navigation and again on the landed URL (a redirect into
 * the list throws and the page is discarded). Only IP LITERALS are matched
 * against a CIDR — the host does no DNS; a hostname that resolves into a
 * denied range is the browser container's network exposure: what that
 * container can reach, a page can reach.
 */

export const DEFAULT_DENY = "file:,169.254.0.0/16,fd00::/8";

export type DenyRule =
  | { kind: "scheme"; scheme: string }
  | { kind: "host"; host: string }
  | { kind: "cidr"; family: 4 | 6; net: bigint; bits: number };

export function parseDenyList(spec: string = DEFAULT_DENY): DenyRule[] {
  const rules: DenyRule[] = [];
  for (const raw of spec.split(",")) {
    const item = raw.trim().toLowerCase();
    if (!item) continue;
    if (item.endsWith(":")) {
      rules.push({ kind: "scheme", scheme: item });
      continue;
    }
    const slash = item.indexOf("/");
    const addr = slash === -1 ? item : item.slice(0, slash);
    const ip = parseIp(addr);
    if (ip) {
      const max = ip.family === 4 ? 32 : 128;
      const bits = slash === -1 ? max : Number(item.slice(slash + 1));
      if (!Number.isInteger(bits) || bits < 0 || bits > max) throw new Error(`BROWSER_URL_DENY: bad prefix length in "${raw.trim()}"`);
      rules.push({ kind: "cidr", family: ip.family, net: mask(ip.value, bits, max), bits });
      continue;
    }
    rules.push({ kind: "host", host: item.replace(/\.$/, "") });
  }
  return rules;
}

/** The reason `url` is denied, or null when it is allowed. A URL that does
 *  not parse is denied too — the browser must never see it. */
export function denyReason(url: string, rules: DenyRule[]): string | null {
  let u: URL;
  try {
    u = new URL(url);
  } catch {
    return `not a URL: ${url}`;
  }
  const scheme = u.protocol.toLowerCase();
  const host = u.hostname.toLowerCase().replace(/^\[|\]$/g, "").replace(/\.$/, "");
  const ip = parseIp(host);
  // An IPv4-mapped IPv6 literal (::ffff:a.b.c.d) parses as a family-6 value but
  // reaches the embedded IPv4 on a dual-stack host; also test that v4 address
  // against the family-4 CIDR rules, or [::ffff:169.254.169.254] escapes a
  // 169.254.0.0/16 deny.
  const mapped4: { family: 4; value: bigint } | null =
    ip && ip.family === 6 && (ip.value >> 32n) === 0xffffn ? { family: 4, value: ip.value & 0xffffffffn } : null;
  for (const r of rules) {
    if (r.kind === "scheme" && r.scheme === scheme) return `scheme ${scheme} is denied`;
    if (r.kind === "host" && r.host === host) return `host ${host} is denied`;
    if (r.kind === "cidr") {
      for (const cand of [ip, mapped4]) {
        if (cand && cand.family === r.family && mask(cand.value, r.bits, r.family === 4 ? 32 : 128) === r.net)
          return `address ${host} is in a denied range`;
      }
    }
  }
  return null;
}

function mask(value: bigint, bits: number, max: number): bigint {
  return bits === 0 ? 0n : (value >> BigInt(max - bits)) << BigInt(max - bits);
}

function parseIp(s: string): { family: 4 | 6; value: bigint } | null {
  const v4 = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(s);
  if (v4) {
    const parts = v4.slice(1).map(Number);
    if (parts.some((p) => p > 255)) return null;
    return { family: 4, value: parts.reduce((acc, p) => (acc << 8n) | BigInt(p), 0n) };
  }
  if (!s.includes(":")) return null;
  // IPv6: expand `::` once, 8 groups of 16 bits (a trailing dotted v4 is rare
  // enough to leave out — such a literal simply isn't matched).
  const halves = s.split("::");
  if (halves.length > 2) return null;
  const head = halves[0] ? halves[0].split(":") : [];
  const tail = halves.length === 2 && halves[1] ? halves[1].split(":") : [];
  const fill = 8 - head.length - tail.length;
  if (fill < 0 || (halves.length === 1 && fill !== 0)) return null;
  const groups = [...head, ...Array(fill).fill("0"), ...tail];
  let value = 0n;
  for (const g of groups) {
    if (!/^[0-9a-f]{1,4}$/.test(g)) return null;
    value = (value << 16n) | BigInt(parseInt(g, 16));
  }
  return { family: 6, value };
}
