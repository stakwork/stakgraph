mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_main_py_exact_counts() {
    let main_py = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&[&main_py]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Var:"), 5);
}

#[test]
fn main_py_contains_exact_named_nodes() {
    let main_py = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&[&main_py]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Function: cleanup (37-41)"), true);
    assert_eq!(out.stdout.contains("Function: run_servers (52-91)"), true);
    assert_eq!(out.stdout.contains("Var: fastapi_app (26-29)"), true);
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_type_filter_endpoints_only_python() {
    let dir = fixture_path("src/testing/python/web");
    let out = run_stakgraph(&["--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint:"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("Function:"), "stdout: {}", out.stdout);
}

#[test]
fn parse_stats_python_dir() {
    let dir = fixture_path("src/testing/python/web");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("9"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("UnitTest"), "stdout: {}", out.stdout);
}

#[test]
fn parse_json_envelope_python() {
    let main = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&["--json", &main]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "parse");
    assert!(v["data"]["files"].is_array());
}

// ── deps ──────────────────────────────────────────────────────────────────────

#[test]
fn deps_python_run_servers() {
    let main = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&["deps", "run_servers", &main]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("run_servers"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("cleanup"), "stdout: {}", out.stdout);
}

#[test]
fn deps_python_json_edges() {
    let main = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&["--json", "deps", "run_servers", &main]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "deps");
    let edges = v["data"]["edges"].as_array().expect("edges array");
    assert!(!edges.is_empty());
}

// ── impact ────────────────────────────────────────────────────────────────────

#[test]
fn impact_python_create_person() {
    let dir = fixture_path("src/testing/python/web");
    let out = run_stakgraph(&["impact", "--name", "create_person", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("2 Endpoints"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("1 IntegrationTest"), "stdout: {}", out.stdout);
}

#[test]
fn impact_python_json_summary() {
    let dir = fixture_path("src/testing/python/web");
    let out = run_stakgraph(&["--json", "impact", "--name", "create_person", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "impact");
    assert_eq!(v["data"]["summary"]["total"], 3);
    assert_eq!(v["data"]["summary"]["by_type"]["Endpoint"], 2);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_python_post_endpoints() {
    let dir = fixture_path("src/testing/python/web");
    let out = run_stakgraph(&["search", "POST", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("3 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("create_person"), "stdout: {}", out.stdout);
}

#[test]
fn search_python_json_total() {
    let dir = fixture_path("src/testing/python/web");
    let out = run_stakgraph(&["--json", "search", "POST", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["data"]["total"], 3);
}

// ── overview ──────────────────────────────────────────────────────────────────

#[test]
fn overview_python_tree_shape() {
    let dir = fixture_path("src/testing/python/web");
    let out = run_stakgraph(&["overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("django_app"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("flask_app"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("fastapi_app"), "stdout: {}", out.stdout);
}

// ── summarize (--max-tokens) ──────────────────────────────────────────────────

#[test]
fn summarize_python_single_file() {
    let main = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&[&main, "--max-tokens", "500"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Summary:"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("main.py"), "stdout: {}", out.stdout);
}
