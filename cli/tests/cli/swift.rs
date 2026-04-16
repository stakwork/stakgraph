mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_api_swift_exact_counts() {
    let file = fixture_path("src/testing/swift/LegacyApp/SphinxTestApp/API.swift");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 2);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Request:"), 2);
    assert_eq!(count_prefix(&out.stdout, "  → "), 2);
}

#[test]
fn api_swift_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/swift/LegacyApp/SphinxTestApp/API.swift");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: API (11-126)"), true);
    assert_eq!(
        out.stdout.contains("Function: API.getPeopleList (62-85)"),
        true
    );
    assert_eq!(out.stdout.contains("Request: GET /people (68-72)"), true);
    assert_eq!(out.stdout.contains("Request: POST /person (103-107)"), true);
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_stats_swift_dir() {
    let dir = fixture_path("src/testing/swift");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Class                26"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             38"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("UnitTest             6"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_swift_person_class() {
    let dir = fixture_path("src/testing/swift");
    let out = run_stakgraph(&["search", "Person", "--type", "Class", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("4 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Class: Person"), "stdout: {}", out.stdout);
}

#[test]
fn search_swift_get_request() {
    let dir = fixture_path("src/testing/swift");
    let out = run_stakgraph(&["search", "GET", "--type", "Request", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("1 result"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Request: GET /people"), "stdout: {}", out.stdout);
}
