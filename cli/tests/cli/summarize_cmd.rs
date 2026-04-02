mod common;

use common::{fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn summarize_single_file_smoke() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&[&routes, "--max-tokens", "5000"]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Summary:"));
    assert!(out.stdout.contains("File:"));
}

#[test]
fn summarize_directory_with_depth() {
    let rust_dir = fixture_path("src/testing/rust");
    let out = run_stakgraph(&[&rust_dir, "--max-tokens", "5000", "--depth", "1"]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Directory Structure"));
    assert!(out.stdout.contains("File Summaries"));
}

#[test]
fn summarize_token_budget_footer_present() {
    let rust_dir = fixture_path("src/testing/rust");
    let out = run_stakgraph(&[&rust_dir, "--max-tokens", "80"]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("tokens"));
}

#[test]
fn summarize_invalid_path_fails() {
    let out = run_stakgraph(&["./definitely-not-a-real-path-xyz", "--max-tokens", "5000"]);

    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("path does not exist"));
}

#[test]
fn summarize_json_directory_mode_returns_machine_readable_payload() {
    let rust_dir = fixture_path("src/testing/rust");
    let out = run_stakgraph(&["--json", &rust_dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["command"], "parse");
    assert_eq!(payload["data"]["mode"], "parse");
    assert!(payload["data"]["files"].is_array());
}
