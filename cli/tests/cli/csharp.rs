mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_person_controller_cs_exact_counts() {
    let file = fixture_path("src/testing/csharp/Controllers/PersonController.cs");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Var:"), 3);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 10);
    assert_eq!(count_prefix(&out.stdout, "Endpoint:"), 9);
}

#[test]
fn person_controller_cs_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/csharp/Controllers/PersonController.cs");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: PersonController (7-107)"), true);
    assert_eq!(
        out.stdout
            .contains("Function: ApiController.GetById (33-42)"),
        true
    );
    assert_eq!(
        out.stdout.contains("Endpoint: HTTPGET {id:int} (33-42)"),
        true
    );
    assert_eq!(
        out.stdout
            .contains("Endpoint: HTTPPOST {id:int}/avatar (101-106)"),
        true
    );
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_stats_csharp_dir() {
    let dir = fixture_path("src/testing/csharp");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint             81"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Class                164"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             363"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_csharp_get_endpoints() {
    let dir = fixture_path("src/testing/csharp");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("20 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: HTTPGET"), "stdout: {}", out.stdout);
}
