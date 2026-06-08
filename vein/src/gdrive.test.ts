import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { buildQuery } from "./steps/lib/gdrive/list-files.js";
import { statusOf, describeDriveError } from "./steps/lib/gdrive/_shared.js";

// gdrive/* steps use the @googleapis/drive SDK directly (not an injectable
// http capability), so we unit-test the pure pieces: the `q` builder and the
// error mapper. (A full run() test would need to mock the SDK import.)

describe("gdrive/list-files buildQuery", () => {
  it("excludes trashed by default", () => {
    assert.equal(buildQuery({ includeTrashed: false }), "trashed = false");
  });

  it("includes trashed → empty filter (list everything)", () => {
    assert.equal(buildQuery({ includeTrashed: true }), "");
  });

  it("AND-combines folder, modifiedAfter, and mimeType", () => {
    const q = buildQuery({
      folderId: "FOLDER1",
      modifiedAfter: "2024-01-01T00:00:00Z",
      mimeType: "application/vnd.google-apps.document",
      includeTrashed: false,
    });
    assert.equal(
      q,
      "'FOLDER1' in parents and modifiedTime > '2024-01-01T00:00:00Z' and mimeType = 'application/vnd.google-apps.document' and trashed = false",
    );
  });

  it("builds the incremental-sync clause from a cursor", () => {
    assert.equal(
      buildQuery({
        folderId: "F",
        modifiedAfter: "2024-06-01T12:00:00Z",
        includeTrashed: false,
      }),
      "'F' in parents and modifiedTime > '2024-06-01T12:00:00Z' and trashed = false",
    );
  });
});

describe("gdrive statusOf", () => {
  it("reads status from .status, .response.status, or numeric .code", () => {
    assert.equal(statusOf({ status: 404 }), 404);
    assert.equal(statusOf({ response: { status: 403 } }), 403);
    assert.equal(statusOf({ code: 401 }), 401);
    assert.equal(statusOf({ code: "500" }), 500);
    assert.equal(statusOf({ code: "ENOTFOUND" }), undefined);
    assert.equal(statusOf(new Error("boom")), undefined);
  });
});

describe("gdrive describeDriveError", () => {
  it("404 → actionable not-found/access message", () => {
    const e = describeDriveError({ status: 404 }, 'file "X"', true);
    assert.match(e.message, /not found.*access/s);
    assert.match(e.message, /file "X"/);
  });

  it("403 → scope / permission hint", () => {
    const e = describeDriveError({ status: 403 }, "file listing", true);
    assert.match(e.message, /access denied \(403\).*drive\.readonly/s);
  });

  it("401 → expired/invalid token hint", () => {
    const e = describeDriveError({ status: 401 }, "file listing", true);
    assert.match(e.message, /auth failed \(401\).*GOOGLE_ACCESS_TOKEN/s);
  });

  it("no creds + ADC failure → 'no credentials' guidance", () => {
    const e = describeDriveError(
      new Error("Could not load the default credentials"),
      "file listing",
      false,
    );
    assert.match(e.message, /No Google credentials available/);
  });

  it("passes unknown errors through unchanged", () => {
    const orig = new Error("network blip");
    assert.equal(describeDriveError(orig, "file listing", true), orig);
  });
});
