mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_test_person_svelte_exact_counts() {
    let file = fixture_path("src/testing/svelte/tests/test_person.svelte");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Datamodel:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 0);
}

#[test]
fn test_person_svelte_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/svelte/tests/test_person.svelte");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Datamodel: <script> (1-31)"), true);
    assert_eq!(out.stdout.contains("Datamodel: <svelte:head> (1-31)"), true);
    assert_eq!(out.stdout.contains("Datamodel: <div> (1-31)"), true);
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_stats_svelte_dir() {
    let dir = fixture_path("src/testing/svelte");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint             2"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             10"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Class                4"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_svelte_get_endpoint() {
    let dir = fixture_path("src/testing/svelte");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("1 result"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: GET /api/people"), "stdout: {}", out.stdout);
}
