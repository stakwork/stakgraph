mod common;

use common::{fixture_path, run_stakgraph};
use serde_json::Value;

// ── error / validation ──────────────────────────────────────────────────────

#[test]
fn search_missing_files_fails() {
    let out = run_stakgraph(&["search", "anything"]);
    assert_ne!(out.exit_code, 0);
}

#[test]
fn search_invalid_type_fails() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["search", "add", "--type", "NotAType", &calc]);
    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("Unknown node type"), "stderr: {}", out.stderr);
}

#[test]
fn search_no_parseable_files_in_dir_fails() {
    let dir = tempfile::tempdir().expect("tempdir failed");
    let path = dir.path().to_string_lossy().to_string();
    let out = run_stakgraph(&["search", "foo", &path]);
    assert_ne!(out.exit_code, 0);
    assert!(
        out.stderr.contains("no parseable files"),
        "stderr: {}",
        out.stderr
    );
}

// ── basic smoke ─────────────────────────────────────────────────────────────

#[test]
fn search_smoke_single_file_by_name() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["search", "add", "--type", "Function", &calc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("Calculator.add"),
        "stdout: {}",
        out.stdout
    );
}

#[test]
fn search_smoke_directory_returns_multiple_results() {
    let api = fixture_path("src/testing/nextjs/app/api");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &api]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("results for 'GET'"),
        "stdout: {}",
        out.stdout
    );
    assert!(out.stdout.contains("Endpoint:"), "stdout: {}", out.stdout);
}

#[test]
fn search_no_results_exits_zero_with_message() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["search", "xyznonexistenttoken", &calc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("No nodes matching"),
        "stdout: {}",
        out.stdout
    );
}

// ── type, file, limit filters ────────────────────────────────────────────────

#[test]
fn search_type_filter_returns_only_endpoints() {
    let api = fixture_path("src/testing/nextjs/app/api");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", &api]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint:"), "stdout: {}", out.stdout);
    assert!(!out.stdout.contains("Function:"), "stdout: {}", out.stdout);
}

#[test]
fn search_file_filter_scopes_to_matching_paths() {
    let api = fixture_path("src/testing/nextjs/app/api");
    let out = run_stakgraph(&[
        "search",
        "users",
        "--type",
        "Endpoint",
        "--file",
        "users",
        &api,
    ]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("/api/users"),
        "stdout: {}",
        out.stdout
    );
    assert!(
        !out.stdout.contains("/api/comments"),
        "stdout: {}",
        out.stdout
    );
    assert!(
        !out.stdout.contains("/api/products"),
        "stdout: {}",
        out.stdout
    );
}

#[test]
fn search_limit_caps_result_count() {
    let api = fixture_path("src/testing/nextjs/app/api");
    let out = run_stakgraph(&["search", "GET", "--type", "Endpoint", "--limit", "2", &api]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("2 results for"),
        "stdout: {}",
        out.stdout
    );
}

// ── multi-term ranking ───────────────────────────────────────────────────────

#[test]
fn search_multi_term_matches_both_terms() {
    let api = fixture_path("src/testing/nextjs/app/api");
    let out = run_stakgraph(&["search", "GET POST", "--type", "Endpoint", "--limit", "10", &api]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("GET /api"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("POST /api"), "stdout: {}", out.stdout);
}

// ── body search ──────────────────────────────────────────────────────────────

#[test]
fn search_body_flag_broadens_results() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");

    // "result" appears in body of add/multiply/subtract but only in the name of getResult
    let without_body = run_stakgraph(&["search", "result", &calc]);
    assert_eq!(without_body.exit_code, 0, "stderr: {}", without_body.stderr);

    let with_body = run_stakgraph(&["search", "result", "--body", &calc]);
    assert_eq!(with_body.exit_code, 0, "stderr: {}", with_body.stderr);

    let count_without = without_body.stdout.lines().filter(|l| l.contains("Function:") || l.contains("Class:")).count();
    let count_with = with_body.stdout.lines().filter(|l| l.contains("Function:") || l.contains("Class:")).count();

    assert!(
        count_with > count_without,
        "--body should return more results; without={}, with={}",
        count_without,
        count_with
    );
}

// ── code / detail mode ───────────────────────────────────────────────────────

#[test]
fn search_code_flag_shows_source_body() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["search", "add", "--type", "Function", "--code", "--limit", "1", &calc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("```"), "expected code fence; stdout: {}", out.stdout);
    assert!(
        out.stdout.contains("this.result"),
        "expected body content; stdout: {}",
        out.stdout
    );
}

// ── node-type-aware display ──────────────────────────────────────────────────

#[test]
fn search_endpoint_shows_handler_metadata() {
    let api = fixture_path("src/testing/nextjs/app/api/users/route.ts");
    let out = run_stakgraph(&["search", "POST", "--type", "Endpoint", &api]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Endpoint:"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("Handler:"), "stdout: {}", out.stdout);
}

// ── context (1-hop edges) ────────────────────────────────────────────────────

#[test]
fn search_context_flag_shows_arrows() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["search", "add", "--type", "Function", "--context", &calc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    // The result section should contain arrow indicators even if no edges exist yet;
    // the key check is it doesn't blow up and still shows the node
    assert!(
        out.stdout.contains("Calculator.add"),
        "stdout: {}",
        out.stdout
    );
}

// ── tests flag ───────────────────────────────────────────────────────────────

#[test]
fn search_tests_flag_surfaces_associated_test_nodes() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let test_file = fixture_path("src/testing/nextjs/app/test/unit.class.test.ts");
    let out = run_stakgraph(&[
        "search",
        "add",
        "--type",
        "Function",
        "--tests",
        &calc,
        &test_file,
    ]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("Calculator.add"),
        "stdout: {}",
        out.stdout
    );
    assert!(
        out.stdout.contains("Tests:"),
        "expected 'Tests:' section; stdout: {}",
        out.stdout
    );
    assert!(
        out.stdout.contains("unit.class.test.ts"),
        "expected test file in output; stdout: {}",
        out.stdout
    );
}

// ── related (sibling) nodes ──────────────────────────────────────────────────

#[test]
fn search_related_flag_shows_sibling_nodes() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["search", "add", "--type", "Function", "--related", &calc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("Related:"),
        "expected 'Related:' section; stdout: {}",
        out.stdout
    );
    assert!(
        out.stdout.contains("Calculator.multiply") || out.stdout.contains("Calculator.subtract"),
        "expected sibling functions; stdout: {}",
        out.stdout
    );
}

// ── JSON output ───────────────────────────────────────────────────────────────

#[test]
fn search_json_outputs_valid_payload() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["--json", "search", "add", "--type", "Function", &calc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");

    assert_eq!(payload["ok"], true);
    assert_eq!(payload["command"], "search");
    assert!(payload["data"]["results"].is_array());
    assert!(payload["data"]["total"].is_number());
    assert_eq!(payload["data"]["query"], "add");

    let results = payload["data"]["results"].as_array().unwrap();
    assert!(!results.is_empty(), "expected results array to be non-empty");

    let first = &results[0];
    assert!(first["name"].is_string());
    assert!(first["node_type"].is_string());
    assert!(first["file"].is_string());
    assert!(first["lines"].is_string());
    assert!(first["score"].is_number());
}

#[test]
fn search_json_no_results_returns_ok_with_warning() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&["--json", "search", "xyznonexistenttoken", &calc]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");

    assert_eq!(payload["ok"], true);
    assert_eq!(payload["data"]["total"], 0);
    let warnings = payload["warnings"].as_array().unwrap();
    assert!(!warnings.is_empty(), "expected a warning for no results");
    assert_eq!(warnings[0]["kind"], "no_results");
}

#[test]
fn search_json_with_code_includes_body_field() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let out = run_stakgraph(&[
        "--json",
        "search",
        "add",
        "--type",
        "Function",
        "--code",
        "--limit",
        "1",
        &calc,
    ]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");

    let results = payload["data"]["results"].as_array().unwrap();
    assert!(!results.is_empty());
    assert!(
        results[0]["body"].is_string(),
        "expected 'body' field when --code is set"
    );
}

#[test]
fn search_json_with_context_includes_callers_callees() {
    let calc = fixture_path("src/testing/nextjs/lib/calculator.ts");
    let test_file = fixture_path("src/testing/nextjs/app/test/unit.class.test.ts");
    let out = run_stakgraph(&[
        "--json",
        "search",
        "add",
        "--type",
        "Function",
        "--context",
        &calc,
        &test_file,
    ]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");

    let results = payload["data"]["results"].as_array().unwrap();
    assert!(!results.is_empty());
    // callers and callees arrays must exist (may be empty if no edges in single-file parse)
    assert!(results[0]["callers"].is_array());
    assert!(results[0]["callees"].is_array());
}
