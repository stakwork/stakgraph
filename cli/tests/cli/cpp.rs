mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_routes_cpp_exact_counts() {
    let file = fixture_path("src/testing/cpp/web_api/routes.cpp");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 2);
    assert_eq!(count_prefix(&out.stdout, "  → "), 2);
}

#[test]
fn routes_cpp_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/cpp/web_api/routes.cpp");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(
        out.stdout.contains("Endpoint: ANY /person/<int> (26-29)"),
        true
    );
    assert_eq!(out.stdout.contains("Endpoint: POST /person (31-34)"), true);
    assert_eq!(out.stdout.contains("Function: setup_routes (25-35)"), true);
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_stats_cpp_dir() {
    let dir = fixture_path("src/testing/cpp");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint             3"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             112"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Class                1"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_cpp_get_endpoint() {
    let dir = fixture_path("src/testing/cpp");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("1 result"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: ANY /person/<int>"), "stdout: {}", out.stdout);
}
