mod common;

use common::{count_prefix, fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn smoke_traits_rs_exact_counts() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&[&traits]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 12);
    assert_eq!(count_prefix(&out.stdout, "Datamodel:"), 5);
    assert_eq!(count_prefix(&out.stdout, "Trait:"), 4);

    let test_nodes =
        count_prefix(&out.stdout, "UnitTest:") + count_prefix(&out.stdout, "IntegrationTest:");
    assert_eq!(test_nodes, 10);
}

#[test]
fn no_nested_removes_item_datamodel_exactly() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--no-nested", &traits]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(count_prefix(&out.stdout, "Function:"), 12);
    assert_eq!(count_prefix(&out.stdout, "Datamodel:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Trait:"), 4);
    assert_eq!(count_prefix(&out.stdout, "Datamodel: Item"), 0);
}

#[test]
fn traits_rs_contains_exact_named_nodes() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&[&traits]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.contains("Trait: Greet (4-6)"), true);
    assert_eq!(out.stdout.contains("Datamodel: Greeter (9-11)"), true);
    assert_eq!(
        out.stdout.contains("Function: Greeter::greet (20-22)"),
        true
    );
}

#[test]
fn invalid_path_exits_nonzero_with_exact_error() {
    let out = run_stakgraph(&["nonexistent_file.rs"]);

    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("file does not exist"));
}

#[test]
fn quiet_mode_has_empty_stderr() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["-q", &traits]);

    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stderr, "");
}

// ── parse ─────────────────────────────────────────────────────────────────────

#[test]
fn parse_type_filter_endpoints_only_rust() {
    let src = fixture_path("src/testing/rust/src");
    let out = run_stakgraph(&["--type", "Endpoint", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint:"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("Function:"), "stdout: {}", out.stdout);
}

#[test]
fn parse_stats_rust_src() {
    let src = fixture_path("src/testing/rust/src");
    let out = run_stakgraph(&["--stats", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("21"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("UnitTest"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("41"), "stdout: {}", out.stdout);
}

#[test]
fn parse_json_envelope_rust() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--json", &traits]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "parse");
    assert!(v["data"]["files"].is_array());
}

// ── deps ──────────────────────────────────────────────────────────────────────

#[test]
fn deps_rust_batch_process_chain() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["deps", "batch_process", &traits]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("batch_process"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("process"), "stdout: {}", out.stdout);
}

#[test]
fn deps_rust_json_edges() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--json", "deps", "batch_process", &traits]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "deps");
    let edges = v["data"]["edges"].as_array().expect("edges array");
    assert!(!edges.is_empty());
    assert_eq!(edges[0]["target_name"], "process");
}

// ── impact ────────────────────────────────────────────────────────────────────

#[test]
fn impact_rust_batch_process() {
    let dir = fixture_path("src/testing/rust");
    let out = run_stakgraph(&["impact", "--name", "batch_process", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("2 UnitTests"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("test_simple_processor_batch"), "stdout: {}", out.stdout);
}

#[test]
fn impact_rust_json_summary() {
    let dir = fixture_path("src/testing/rust");
    let out = run_stakgraph(&["--json", "impact", "--name", "batch_process", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "impact");
    assert_eq!(v["data"]["summary"]["total"], 3);
    assert_eq!(v["data"]["summary"]["by_type"]["UnitTest"], 2);
}

// ── search ────────────────────────────────────────────────────────────────────

#[test]
fn search_rust_get_endpoints() {
    let src = fixture_path("src/testing/rust/src");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("11 results"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Endpoint: GET"), "stdout: {}", out.stdout);
}

#[test]
fn search_rust_json_total() {
    let src = fixture_path("src/testing/rust/src");
    let out = run_stakgraph(&["--json", "search", "GET", "--type", "Endpoint", &src]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let v: Value = serde_json::from_str(&out.stdout).expect("invalid JSON");
    assert_eq!(v["data"]["total"], 11);
}

// ── overview ──────────────────────────────────────────────────────────────────

#[test]
fn overview_rust_tree_shape() {
    let dir = fixture_path("src/testing/rust");
    let out = run_stakgraph(&["overview", &dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("src/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("routes/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Rust"), "stdout: {}", out.stdout);
}

// ── summarize (--max-tokens) ──────────────────────────────────────────────────

#[test]
fn summarize_rust_single_file() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&[&traits, "--max-tokens", "500"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Summary:"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("traits.rs"), "stdout: {}", out.stdout);
}
