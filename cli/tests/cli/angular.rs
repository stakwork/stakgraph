mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_app_routes_angular_exact_counts() {
    let file = fixture_path("src/testing/angular/src/app/app.routes.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Var:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 0);
}

#[test]
fn app_routes_angular_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/angular/src/app/app.routes.ts");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Var: routes (6-10)"), true);
    assert_eq!(out.stdout.contains("Class: AppRoutingModule (16)"), true);
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_stats_angular_dir() {
    let dir = fixture_path("src/testing/angular");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint             1"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             14"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Class                5"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("UnitTest             5"), "stdout: {}", out.stdout);
}

#[test]
fn parse_type_filter_endpoint_angular() {
    let dir = fixture_path("src/testing/angular");
    let out = run_stakgraph(&["--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 1);
    assert!(out.stdout.contains("Endpoint: GET **"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_angular_get_endpoint() {
    let dir = fixture_path("src/testing/angular");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("1 result"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: GET **"), "stdout: {}", out.stdout);
}

// ── deps ──────────────────────────────────────────────────────────────────────

#[test]
fn deps_angular_getpeople_is_leaf() {
    let dir = fixture_path("src/testing/angular");
    let out = run_stakgraph(&["deps", "getPeople", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("getPeople"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("└──"), "stdout: {}", out.stdout);
}

// ── overview ──────────────────────────────────────────────────────────────────

#[test]
fn overview_angular_tree_shape() {
    let dir = fixture_path("src/testing/angular");
    let out = run_stakgraph(&["overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("angular/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("app/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Node.js"), "stdout: {}", out.stdout);
}
