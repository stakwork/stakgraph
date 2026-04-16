mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_routes_c_exact_counts() {
    let file = fixture_path("src/testing/c/web-http/routes.c");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 0);
}

#[test]
fn routes_c_contains_exact_named_functions() {
    let file = fixture_path("src/testing/c/web-http/routes.c");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(
        out.stdout.contains("Function: handler_get_user (5-18)"),
        true
    );
    assert_eq!(
        out.stdout.contains("Function: handler_post_user (20-33)"),
        true
    );
    assert_eq!(
        out.stdout
            .contains("Function: handler_list_products (35-39)"),
        true
    );
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_stats_c_dir() {
    let dir = fixture_path("src/testing/c");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Class                23"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint             4"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             47"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_c_hash_functions() {
    let dir = fixture_path("src/testing/c");
    let out = run_stakgraph(&["search", "hash", "--type", "Function", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("4 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function: hash"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function: ht_create"), "stdout: {}", out.stdout);
}
