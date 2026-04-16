mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_api_service_kotlin_exact_counts() {
    let file = fixture_path(
        "src/testing/kotlin/app/src/main/java/com/kotlintestapp/data/api/ApiService.kt",
    );
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Request:"), 3);
    assert_eq!(count_prefix(&out.stdout, "  → "), 3);
}

#[test]
fn api_service_kotlin_contains_exact_named_nodes() {
    let file = fixture_path(
        "src/testing/kotlin/app/src/main/java/com/kotlintestapp/data/api/ApiService.kt",
    );
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: ApiService (9-18)"), true);
    assert_eq!(
        out.stdout.contains("Function: ApiService.getUsers (10-11)"),
        true
    );
    assert_eq!(out.stdout.contains("Request: GET /users (10-11)"), true);
    assert_eq!(out.stdout.contains("Request: POST /users (16-17)"), true);
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_stats_kotlin_dir() {
    let dir = fixture_path("src/testing/kotlin");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Class                16"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             29"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Request              5"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("UnitTest             2"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_kotlin_get_requests() {
    let dir = fixture_path("src/testing/kotlin");
    let out = run_stakgraph(&["search", "GET", "--type", "Request", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("3 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Request: GET /users"), "stdout: {}", out.stdout);
}

// ── overview ──────────────────────────────────────────────────────────────────

#[test]
fn overview_kotlin_tree_shape() {
    let dir = fixture_path("src/testing/kotlin");
    let out = run_stakgraph(&["overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("kotlin/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("app/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Kotlin"), "stdout: {}", out.stdout);
}
