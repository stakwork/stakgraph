mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_billing_service_java_exact_counts() {
    let file = fixture_path(
        "src/testing/java/src/main/java/graph/stakgraph/java/nonweb/BillingService.java",
    );
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Import:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Class:"), 1);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 0);
}

#[test]
fn billing_service_java_contains_exact_named_nodes() {
    let file = fixture_path(
        "src/testing/java/src/main/java/graph/stakgraph/java/nonweb/BillingService.java",
    );
    let out = run_stakgraph(&[&file]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Class: BillingService (9-15)"), true);
    assert_eq!(
        out.stdout
            .contains("import graph.stakgraph.java.model.Person;"),
        true
    );
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_stats_java_dir() {
    let dir = fixture_path("src/testing/java");
    let out = run_stakgraph(&["--stats", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Class                10"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Function             1"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("UnitTest             1"), "stdout: {}", out.stdout);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_java_billingservice_class() {
    let dir = fixture_path("src/testing/java");
    let out = run_stakgraph(&["search", "BillingService", "--type", "Class", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("1 result"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Class: BillingService"), "stdout: {}", out.stdout);
}

// ── overview ──────────────────────────────────────────────────────────────────

#[test]
fn overview_java_tree_shape() {
    let dir = fixture_path("src/testing/java");
    let out = run_stakgraph(&["overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("java/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Java"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains(".java"), "stdout: {}", out.stdout);
}
