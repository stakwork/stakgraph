mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_api_php_exact_counts() {
    let api = fixture_path("src/testing/php/routes/api.php");
    let out = run_stakgraph(&[&api]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 15);
}

#[test]
fn api_php_contains_exact_named_endpoints() {
    let api = fixture_path("src/testing/php/routes/api.php");
    let out = run_stakgraph(&[&api]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Endpoint: GET /user (20-22)"), true);
    assert_eq!(out.stdout.contains("Endpoint: POST /login (24)"), true);
    assert_eq!(
        out.stdout.contains("Endpoint: DELETE /posts/{post} (29)"),
        true
    );
    assert_eq!(
        out.stdout
            .contains("Endpoint: POST /posts/{post}/like (32)"),
        true
    );
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_type_filter_endpoints_only_php() {
    let dir = fixture_path("src/testing/php");
    let out = run_stakgraph(&["--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 41);
}

#[test]
fn parse_stats_php_dir() {
    let dir = fixture_path("src/testing/php");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint             41"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Class                9"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             23"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_php_get_endpoints() {
    let dir = fixture_path("src/testing/php");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("20 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: GET /"), "stdout: {}", out.stdout);
}
