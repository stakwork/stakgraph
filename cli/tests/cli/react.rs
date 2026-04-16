mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_routes_react_exact_counts() {
    let file = fixture_path("src/testing/react/src/api/routes.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 5);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 4);
}

#[test]
fn routes_react_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/react/src/api/routes.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Endpoint: GET /users (6-9)"), true);
    assert_eq!(out.stdout.contains("Endpoint: POST /users (12-16)"), true);
    assert_eq!(
        out.stdout.contains("Endpoint: PUT /users/:id (19-23)"),
        true
    );
    assert_eq!(
        out.stdout.contains("Endpoint: DELETE /users/:id (26-29)"),
        true
    );
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_type_filter_endpoints_only_react() {
    let src = fixture_path("src/testing/react/src");
    let out = run_stakgraph(&["--type", "Endpoint", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint:"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("Function:"), "stdout: {}", out.stdout);
}

#[test]
fn parse_stats_react_src() {
    let src = fixture_path("src/testing/react/src");
    let out = run_stakgraph(&["--stats", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Page"), "stdout: {}", out.stdout);
}

#[test]
fn parse_json_envelope_react() {
    let app = fixture_path("src/testing/react/src/App.tsx");
    let out = run_stakgraph(&["--json", &app]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "parse");
    assert!(v["data"]["files"].is_array());
}

// ── deps ──────────────────────────────────────────────────────────────────────

#[test]
fn deps_react_usefetch_has_children() {
    let src = fixture_path("src/testing/react/src");
    let out = run_stakgraph(&["deps", "useFetch", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("useFetch"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("useState"), "stdout: {}", out.stdout);
}

#[test]
fn deps_react_json_edges() {
    let src = fixture_path("src/testing/react/src");
    let out = run_stakgraph(&["--json", "deps", "useFetch", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "deps");
    let edges = v["data"]["edges"].as_array().expect("edges array");
    assert!(!edges.is_empty());
}

// ── impact ────────────────────────────────────────────────────────────────────

#[test]
fn impact_react_app_component() {
    let dir = fixture_path("src/testing/react");
    let out = run_stakgraph(&["impact", "--name", "App", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("affected"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("index.tsx"), "stdout: {}", out.stdout);
}

#[test]
fn impact_react_json_summary() {
    let dir = fixture_path("src/testing/react");
    let out = run_stakgraph(&["--json", "impact", "--name", "App", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "impact");
    assert_eq!(v["data"]["summary"]["total"], 1);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_react_get_endpoints() {
    let src = fixture_path("src/testing/react/src");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("2 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: GET"), "stdout: {}", out.stdout);
}

#[test]
fn search_react_json_total() {
    let src = fixture_path("src/testing/react/src");
    let out = run_stakgraph(&["--json", "search", "GET", "--type", "Endpoint", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["data"]["total"], 2);
}

// ── overview ──────────────────────────────────────────────────────────────────

#[test]
fn overview_react_tree_shape() {
    let dir = fixture_path("src/testing/react");
    let out = run_stakgraph(&["overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("src/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("components/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Node.js"), "stdout: {}", out.stdout);
}

// ── summarize (--max-tokens) ──────────────────────────────────────────────────

#[test]
fn summarize_react_single_file() {
    let app = fixture_path("src/testing/react/src/App.tsx");
    let out = run_stakgraph(&[&app, "--max-tokens", "500"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Summary:"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("App.tsx"), "stdout: {}", out.stdout);
}
