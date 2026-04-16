mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_routes_ts_exact_counts() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&[&routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 5);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Datamodel:"), 3);
}

#[test]
fn routes_ts_contains_exact_named_nodes() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&[&routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Endpoint: POST /people/new (18)"), true);
    assert_eq!(out.stdout.contains("Function: getPerson (32-49)"), true);
    assert_eq!(out.stdout.contains("Datamodel: PersonRequest (6)"), true);
}

#[test]
fn comma_separated_single_arg_runs_both_files() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let files = format!("{traits},{routes}");
    let out = run_stakgraph(&[&files]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "File:"), 2);
    assert_eq!(out.stdout.contains("Datamodel: Item (55)"), true);
    assert_eq!(out.stdout.contains("Endpoint: POST /people/new (18)"), true);
}

#[test]
fn multiple_separate_args_runs_both_files() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&[&traits, &routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "File:"), 2);
    assert_eq!(out.stdout.contains("Datamodel: Item (55)"), true);
    assert_eq!(out.stdout.contains("Function: getPerson (32-49)"), true);
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_type_filter_endpoints_only_typescript() {
    let src = fixture_path("src/testing/typescript/src");
    let out = run_stakgraph(&["--type", "Endpoint", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint:"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("Function:"), "stdout: {}", out.stdout);
}

#[test]
fn parse_stats_typescript_dir() {
    let dir = fixture_path("src/testing/typescript");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("22"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("UnitTest"), "stdout: {}", out.stdout);
}

#[test]
fn parse_json_envelope_typescript() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&["--json", &routes]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "parse");
    assert!(v["data"]["files"].is_array());
}

// ── deps ──────────────────────────────────────────────────────────────────────

#[test]
fn deps_typescript_leaf_node() {
    let svc = fixture_path("src/testing/typescript/src/services/user-service.ts");
    let out = run_stakgraph(&["deps", "findAll", &svc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("findAll"), "stdout: {}", out.stdout);
}

#[test]
fn deps_typescript_json_seeds() {
    let src = fixture_path("src/testing/typescript/src");
    let out = run_stakgraph(&["--json", "deps", "findAll", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "deps");
    assert!(v["data"]["seeds"].is_array());
}

// ── impact ────────────────────────────────────────────────────────────────────

#[test]
fn impact_typescript_findall() {
    let dir = fixture_path("src/testing/typescript");
    let out = run_stakgraph(&["impact", "--name", "findAll", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("UserService"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("affected"), "stdout: {}", out.stdout);
}

#[test]
fn impact_typescript_json_summary() {
    let dir = fixture_path("src/testing/typescript");
    let out = run_stakgraph(&["--json", "impact", "--name", "findAll", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "impact");
    assert_eq!(v["data"]["summary"]["total"], 2);
    assert_eq!(v["data"]["summary"]["by_type"]["Class"], 2);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_typescript_get_endpoints() {
    let src = fixture_path("src/testing/typescript/src");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("12 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: GET"), "stdout: {}", out.stdout);
}

#[test]
fn search_typescript_json_total() {
    let src = fixture_path("src/testing/typescript/src");
    let out = run_stakgraph(&["--json", "search", "GET", "--type", "Endpoint", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["data"]["total"], 12);
}

// ── overview ──────────────────────────────────────────────────────────────────

#[test]
fn overview_typescript_tree_shape() {
    let dir = fixture_path("src/testing/typescript");
    let out = run_stakgraph(&["overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("src/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("services/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Node.js"), "stdout: {}", out.stdout);
}

// ── summarize (--max-tokens) ──────────────────────────────────────────────────

#[test]
fn summarize_typescript_single_file() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&[&routes, "--max-tokens", "500"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Summary:"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("routes.ts"), "stdout: {}", out.stdout);
}
