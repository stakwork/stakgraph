import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { applyEdits, checkEnvValues, looksMasked, mask, maskPm2, poolName, tagger, unifiedDiff } from "./steps/_pool.js";

// The file hive generates from `services` (its formatPM2Apps), values in clear.
const PM2 = `module.exports = {
  apps: [
    {
      name: "frontend",
      script: "npm run dev",
      cwd: "/workspaces/app",
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        API_KEY: "sk-live-0123456789",
        API_URL: "http://localhost:3000",
        PORT: "3000",
        INSTALL_COMMAND: "npm install",
        POST_START_COMMAND: "node scripts/seed.js"
      }
    }
  ],
};
`;

describe("mask", () => {
  it("is hive's: two characters each side of the stars, four or fewer all stars", () => {
    assert.equal(mask("abcdefyz"), "ab****yz");
    assert.equal(mask("abcdefghijyz"), "ab********yz");
    assert.equal(mask("true"), "****");
    assert.equal(mask("false"), "fa****se");
  });

  it("masks its own output to itself, so a value hive masked survives masking again", () => {
    for (const v of ["sk-live-0123456789", "true", "false", "http://localhost:3000"]) assert.equal(mask(mask(v)), mask(v));
  });

  it("knows a masked value from a value", () => {
    assert.ok(looksMasked("****"));
    assert.ok(looksMasked("ab****yz"));
    assert.ok(looksMasked("ab****yz#3fa9c1"));
    assert.ok(!looksMasked("true"));
    assert.ok(!looksMasked("http://localhost:3000"));
  });
});

describe("maskPm2", () => {
  const plain = (v: string) => `<${mask(v)}>`;

  it("masks the values in the env blocks and leaves the service's config vars and everything else readable", () => {
    const out = maskPm2(PM2, plain);
    assert.match(out, /API_KEY: "<sk\*+89>"/);
    assert.match(out, /API_URL: "<ht\*+00>"/);
    assert.match(out, /PORT: "3000"/);
    assert.match(out, /INSTALL_COMMAND: "npm install"/);
    assert.match(out, /POST_START_COMMAND: "node scripts\/seed.js"/);
    assert.match(out, /script: "npm run dev"/);
    assert.match(out, /cwd: "\/workspaces\/app"/);
    assert.ok(!out.includes("sk-live-0123456789"));
    assert.ok(!out.includes("localhost"));
  });

  it("finds a block's end past template literals and nested braces, in every env block", () => {
    const text = `module.exports = { apps: [{ name: "web", env: { DB: \`postgres://\${host}/db\`, NESTED: { DEEP: 'deep-secret' }, 'QUOTED-KEY': "quoted-secret" }, env_production: { TOKEN: "prod-secret" }, script: "after-the-block" }] };`;
    const out = maskPm2(text, plain);
    for (const secret of ["postgres://", "deep-secret", "quoted-secret", "prod-secret"]) assert.ok(!out.includes(secret), `${secret} leaked: ${out}`);
    assert.match(out, /script: "after-the-block"/);
    assert.match(out, /'QUOTED-KEY': "<qu\*+et>"/);
  });

  it("leaves a file with no env block as it is", () => {
    const text = `module.exports = { apps: [{ name: "web", script: "npm start" }] };`;
    assert.equal(maskPm2(text, plain), text);
  });

  it("tags equal values alike and values that mask alike apart, within one read", () => {
    const tag = tagger();
    assert.equal(tag("http://localhost:3000"), tag("http://localhost:3000"));
    assert.notEqual(tag("http://localhost:3000"), tag("http://localhost:4000"));
    assert.equal(mask("http://localhost:3000"), mask("http://localhost:4000"), "the case the tag is for");
    assert.match(tag("http://localhost:3000"), /^ht\*+00#[0-9a-f]{6}$/);
    assert.notEqual(tagger()("same"), "****#000000");
  });
});

describe("unifiedDiff", () => {
  it("is null for equal texts", () => {
    assert.equal(unifiedDiff("a\nb\n", "a\nb\n", "x", "y"), null);
  });

  it("shows a changed line with three lines of context", () => {
    const a = ["1", "2", "3", "4", "5", "6", "7", "8"].join("\n");
    const b = ["1", "2", "3", "4", "five", "6", "7", "8"].join("\n");
    assert.equal(unifiedDiff(a, b, "hive/f", "pod/f"), "--- hive/f\n+++ pod/f\n@@ -2,7 +2,7 @@\n 2\n 3\n 4\n-5\n+five\n 6\n 7\n 8");
  });

  it("splits far-apart changes into hunks and shows a whole file added", () => {
    const a = Array.from({ length: 20 }, (_, i) => `l${i}`);
    const b = [...a];
    b[1] = "changed-1";
    b[18] = "changed-18";
    const d = unifiedDiff(a.join("\n"), b.join("\n"), "a", "b")!;
    assert.equal(d.match(/^@@/gm)?.length, 2);
    assert.equal(unifiedDiff("", "x\ny\n", "a", "b"), "--- a\n+++ b\n@@ -0,0 +1,2 @@\n+x\n+y");
    assert.equal(unifiedDiff("x\n", "x", "a", "b"), "--- a\n+++ b\n(the files differ only in a final newline)");
  });
});

describe("names", () => {
  it("hive's name for a pod's file; nothing for devcontainer.json", () => {
    assert.equal(poolName("docker-compose.yaml"), "docker-compose.yml");
    assert.equal(poolName("docker-compose.yml"), "docker-compose.yml");
    assert.equal(poolName("pm2.config.cjs"), "pm2.config.js");
    assert.equal(poolName("Dockerfile"), "Dockerfile");
    assert.equal(poolName("devcontainer.json"), null);
  });
});

describe("applyEdits", () => {
  const list = [
    { name: "a", value: "1" },
    { name: "b", value: "2" },
  ];

  it("writes in place, appends the new, removes on null, keeps the rest", () => {
    const r = applyEdits(list, { b: { name: "ignored", value: "20" }, c: { name: "c", value: "3" }, a: null }, "env var");
    assert.deepEqual(r.list, [
      { name: "b", value: "20" },
      { name: "c", value: "3" },
    ]);
    assert.deepEqual([r.written, r.removed], [["b", "c"], ["a"]]);
    assert.deepEqual(list, [{ name: "a", value: "1" }, { name: "b", value: "2" }], "the stored list is not changed");
  });

  it("refuses to remove a name that is not there", () => {
    assert.throws(() => applyEdits(list, { z: null }, "service"), /no service "z" to remove \(stored: a, b\)/);
  });
});

describe("checkEnvValues", () => {
  it("lets a stored value's own mask through and refuses any other masked value", () => {
    checkEnvValues({ KEY: "ab****yz", NEW: "plain" }, { KEY: "ab****yz" }, "env");
    assert.throws(() => checkEnvValues({ KEY: "cd****yz" }, { KEY: "ab****yz" }, "env"), /KEY is a masked value/);
    assert.throws(() => checkEnvValues({ KEY: "ab****yz#3fa9c1" }, { KEY: "ab****yz" }, "env"), /KEY is a masked value/);
    assert.throws(() => checkEnvValues({ OTHER: "****" }, {}, "env"), /OTHER is a masked value/);
  });

  it("refuses a service's config var as an env var, naming where it goes", () => {
    assert.throws(() => checkEnvValues({ POST_START_COMMAND: "x" }, {}, "services.web.env"), /services\.web\.env: POST_START_COMMAND is generated from the service's scripts\.postStart/);
    assert.throws(() => checkEnvValues({ PORT: "3000" }, {}, "env"), /PORT is generated from the service's port/);
  });
});
