mod common;

use common::{fixture_path, run_stakgraph};
use serde_json::Value;

// ── validation errors ─────────────────────────────────────────────────────────

#[test]
fn impact_missing_name_and_file_fails() {
    let dir = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&["impact", &dir]);
    assert_ne!(out.exit_code, 0);
    assert!(
        out.stderr.contains("--name") || out.stderr.contains("--file"),
        "stderr: {}",
        out.stderr
    );
}

#[test]
fn impact_invalid_type_fails() {
    let dir = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&["impact", "--name", "cn", "--type", "NotAType", &dir]);
    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("Unknown node type"), "stderr: {}", out.stderr);
}

#[test]
fn impact_unknown_name_fails() {
    let dir = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&["impact", "--name", "definitely_nonexistent_fn", &dir]);
    assert_ne!(out.exit_code, 0);
    assert!(
        out.stderr.contains("no matching nodes"),
        "stderr: {}",
        out.stderr
    );
}

// ── human output ──────────────────────────────────────────────────────────────

#[test]
fn impact_smoke_name_nextjs() {
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["impact", "--name", "cn", &dir]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("cn"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("affected"), "stdout: {}", out.stdout);
}

#[test]
fn impact_smoke_file_nextjs() {
    let utils = fixture_path("src/testing/nextjs/lib/utils.ts");
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["impact", "--file", &utils, &dir]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("affected"), "stdout: {}", out.stdout);
}

#[test]
fn impact_name_and_file_nextjs() {
    let utils = fixture_path("src/testing/nextjs/lib/utils.ts");
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["impact", "--name", "cn", "--file", &utils, &dir]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("cn"), "stdout: {}", out.stdout);
}

#[test]
fn impact_depth_zero_does_not_crash() {
    let dir = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&["impact", "--name", "cn", "--depth", "0", &dir]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
}

#[test]
fn impact_no_dependents_exits_cleanly() {
    // convertUSDToSats is defined in currency.ts but nothing in the fixture calls it
    let dir = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&["impact", "--name", "convertUSDToSats", &dir]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("No upstream") || out.stdout.contains("0 "),
        "stdout: {}",
        out.stdout
    );
}

// ── JSON output ───────────────────────────────────────────────────────────────

#[test]
fn impact_json_valid_envelope() {
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["--json", "impact", "--name", "cn", &dir]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "impact");
    assert!(v["data"]["summary"]["total"].is_number());
    assert!(v["data"]["seeds"].is_array());
    assert!(v["data"]["affected"].is_array());
}

#[test]
fn impact_json_affected_has_edge_chain() {
    let dir = fixture_path("src/testing/nextjs");
    let out = run_stakgraph(&["--json", "impact", "--name", "cn", &dir]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    let affected = v["data"]["affected"].as_array().expect("affected not array");
    if !affected.is_empty() {
        let first = &affected[0];
        assert!(first["node_type"].is_string());
        assert!(first["name"].is_string());
        assert!(first["file"].is_string());
        assert!(first["depth"].is_number());
        assert!(first["edge_chain"].is_array());
    }
}

#[test]
fn impact_json_no_match_returns_error_envelope() {
    let dir = fixture_path("src/testing/nextjs/lib");
    let out = run_stakgraph(&["--json", "impact", "--name", "nonexistent_xyz", &dir]);
    assert_ne!(out.exit_code, 0);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], false);
    assert!(v["error"]["message"].is_string());
}
