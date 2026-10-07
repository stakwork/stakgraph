mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_person_service_rb_exact_counts() {
    let file = fixture_path("src/testing/ruby/app/services/person_service.rb");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 3);
}

#[test]
fn person_service_rb_contains_exact_named_nodes() {
    let file = fixture_path("src/testing/ruby/app/services/person_service.rb");
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: PersonService (1-16)"), true);
    assert_eq!(
        out.stdout
            .contains("Function: PersonService.get_person_by_id (3-5)"),
        true
    );
    assert_eq!(
        out.stdout
            .contains("Function: PersonService.new_person (8-10)"),
        true
    );
    assert_eq!(
        out.stdout
            .contains("Function: PersonService.delete (13-15)"),
        true
    );
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_stats_ruby_dir() {
    let dir = fixture_path("src/testing/ruby");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint             23"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Class                32"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             61"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("UnitTest             21"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_ruby_get_endpoints() {
    let dir = fixture_path("src/testing/ruby");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("12 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: GET /"), "stdout: {}", out.stdout);
}

#[test]
fn search_ruby_article_cross_type() {
    let dir = fixture_path("src/testing/ruby");
    let out = run_stakgraph(&["search", "article", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("20 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Class: Article"), "stdout: {}", out.stdout);
}
