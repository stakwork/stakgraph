import { test } from "node:test";
import assert from "node:assert/strict";
import { parseDenyList, denyReason, DEFAULT_DENY } from "./deny.js";

const rules = parseDenyList(DEFAULT_DENY);

test("the default list denies file:, link-local v4 and the fd00::/8 ULA range", () => {
  assert.match(denyReason("file:///etc/passwd", rules)!, /scheme file:/);
  assert.match(denyReason("http://169.254.169.254/latest/meta-data/", rules)!, /denied range/);
  assert.match(denyReason("http://[fd12:3456::1]:8080/", rules)!, /denied range/);
  assert.match(denyReason("http://[FD00::]/", rules)!, /denied range/);
});

test("the default list allows the public web and a private LAN address", () => {
  assert.equal(denyReason("https://example.com/", rules), null);
  assert.equal(denyReason("http://10.0.0.5:3000/login", rules), null);
  assert.equal(denyReason("http://localhost:5173/", rules), null);
  assert.equal(denyReason("http://169.253.255.255/", rules), null);
  assert.equal(denyReason("http://[fe80::1]/", rules), null);
});

test("an unparseable URL is denied", () => {
  assert.match(denyReason("not a url", rules)!, /not a URL/);
});

test("schemes, hosts and single addresses are all rules", () => {
  const custom = parseDenyList("javascript:, Metadata.Google.Internal ,10.1.2.3, 2001:db8::/32");
  assert.equal(custom.length, 4);
  assert.match(denyReason("javascript:alert(1)", custom)!, /scheme javascript:/);
  assert.match(denyReason("http://metadata.google.internal/computeMetadata/v1/", custom)!, /host metadata.google.internal/);
  assert.match(denyReason("http://10.1.2.3/", custom)!, /denied range/);
  assert.equal(denyReason("http://10.1.2.4/", custom), null);
  assert.match(denyReason("http://[2001:db8:1::1]/", custom)!, /denied range/);
  assert.equal(denyReason("http://[2001:db9::1]/", custom), null);
});

test("an IPv4-mapped IPv6 literal of a denied v4 range is denied", () => {
  // ::ffff:a.b.c.d parses as IPv6 but reaches the embedded IPv4 on a dual-stack
  // host; new URL() normalizes the dotted tail to the hex form seen here.
  assert.match(denyReason("http://[::ffff:169.254.169.254]/", rules)!, /denied range/);
  assert.match(denyReason("http://[::ffff:a9fe:a9fe]/", rules)!, /denied range/);
});

test("an IPv4-mapped IPv6 literal of an allowed address is not over-blocked", () => {
  assert.equal(denyReason("http://[::ffff:93.184.216.34]/", rules), null); // mapped public
  assert.equal(denyReason("http://[::ffff:10.0.0.5]/", rules), null); // mapped, not in the default list
  assert.equal(denyReason("http://[2001:db8::a9fe:a9fe]/", rules), null); // low 32 bits coincide, not mapped
});

test("a host rule matches a trailing-dot FQDN, either direction", () => {
  assert.match(denyReason("http://metadata.google.internal./", parseDenyList("metadata.google.internal"))!, /host metadata.google.internal/);
  assert.match(denyReason("http://metadata.google.internal/", parseDenyList("metadata.google.internal."))!, /host metadata.google.internal/);
});

test("an empty list denies nothing; a bad prefix length throws", () => {
  assert.equal(denyReason("file:///x", parseDenyList("")), null);
  assert.throws(() => parseDenyList("10.0.0.0/33"), /bad prefix length/);
});
