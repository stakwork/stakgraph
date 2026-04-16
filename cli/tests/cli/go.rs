mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_routes_go_exact_counts() {
    let routes = fixture_path("src/testing/go/routes.go");
    let out = run_stakgraph(&[&routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 2);
    assert_eq!(count_prefix(&out.stdout, "  → "), 2);
}

#[test]
fn routes_go_contains_exact_named_nodes() {
    let routes = fixture_path("src/testing/go/routes.go");
    let out = run_stakgraph(&[&routes]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Endpoint: GET /person/{id} (22)"), true);
    assert_eq!(out.stdout.contains("Function: NewRouter (18-45)"), true);
    assert_eq!(out.stdout.contains("Function: initChi (83-98)"), true);
}

#[test]
fn no_nested_filters_nested_functions_exactly() {
    let file = fixture_path("src/testing/go/anonymous_functions.go");
    let normal = run_stakgraph(&[&file]);
    let filtered = run_stakgraph(&["--no-nested", &file]);

    assert_eq!(normal.exit_code, 0);
    assert_eq!(filtered.exit_code, 0);
    assert_eq!(count_prefix(&normal.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&filtered.stdout, "Function:"), 1);

    assert_eq!(
        normal
            .stdout
            .contains("Function: GET_anon-get_func_L8 (9-11)"),
        true
    );
    assert_eq!(
        normal
            .stdout
            .contains("Function: POST_anon-post_func_L13 (14-16)"),
        true
    );
    assert_eq!(
        filtered
            .stdout
            .contains("Function: GET_anon-get_func_L8 (9-11)"),
        false
    );
    assert_eq!(
        filtered
            .stdout
            .contains("Function: POST_anon-post_func_L13 (14-16)"),
        false
    );
}

#[test]
fn skip_calls_removes_call_arrows_exactly() {
    let routes = fixture_path("src/testing/go/routes.go");
    let normal = run_stakgraph(&[&routes]);
    let skipped = run_stakgraph(&["--skip-calls", &routes]);

    assert_eq!(normal.exit_code, 0);
    assert_eq!(skipped.exit_code, 0);
    assert_eq!(count_prefix(&normal.stdout, "  → "), 2);
    assert_eq!(count_prefix(&skipped.stdout, "  → "), 0);
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_type_filter_endpoints_only_go() {
    let dir = fixture_path("src/testing/go");
    let out = run_stakgraph(&["--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint:"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("Function:"), "stdout: {}", out.stdout);
}

#[test]
fn parse_stats_go_dir() {
    let dir = fixture_path("src/testing/go");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("5"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function"), "stdout: {}", out.stdout);
}

#[test]
fn parse_json_envelope_go() {
    let routes = fixture_path("src/testing/go/routes.go");
    let out = run_stakgraph(&["--json", &routes]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "parse");
    assert!(v["data"]["files"].is_array());
}

// ── deps ──────────────────────────────────────────────────────────────────────

#[test]
fn deps_go_cross_file_chain() {
    let dir = fixture_path("src/testing/go");
    let out = run_stakgraph(&["deps", "NewRouter", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("NewRouter"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("initChi"), "stdout: {}", out.stdout);
}

#[test]
fn deps_go_json_edges() {
    let dir = fixture_path("src/testing/go");
    let out = run_stakgraph(&["--json", "deps", "NewRouter", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "deps");
    let edges = v["data"]["edges"].as_array().expect("edges array");
    assert!(!edges.is_empty());
}

// ── impact ────────────────────────────────────────────────────────────────────

#[test]
fn impact_go_newrouter() {
    let dir = fixture_path("src/testing/go");
    let out = run_stakgraph(&["impact", "--name", "NewRouter", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("2 Functions"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("initChi"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("main"), "stdout: {}", out.stdout);
}

#[test]
fn impact_go_json_summary() {
    let dir = fixture_path("src/testing/go");
    let out = run_stakgraph(&["--json", "impact", "--name", "NewRouter", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "impact");
    assert_eq!(v["data"]["summary"]["total"], 2);
    assert_eq!(v["data"]["summary"]["by_type"]["Function"], 2);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_go_get_endpoints() {
    let dir = fixture_path("src/testing/go");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("3 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: GET"), "stdout: {}", out.stdout);
}

#[test]
fn search_go_json_total() {
    let dir = fixture_path("src/testing/go");
    let out = run_stakgraph(&["--json", "search", "GET", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["data"]["total"], 3);
}

// ── overview ──────────────────────────────────────────────────────────────────

#[test]
fn overview_go_tree_shape() {
    let dir = fixture_path("src/testing/go");
    let out = run_stakgraph(&["overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("routes.go"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("main.go"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Go"), "stdout: {}", out.stdout);
}

// ── summarize (--max-tokens) ──────────────────────────────────────────────────

#[test]
fn summarize_go_single_file() {
    let routes = fixture_path("src/testing/go/routes.go");
    let out = run_stakgraph(&[&routes, "--max-tokens", "500"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Summary:"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("routes.go"), "stdout: {}", out.stdout);
}
