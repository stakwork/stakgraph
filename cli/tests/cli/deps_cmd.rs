mod common;

use common::{fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn deps_missing_name_fails() {
    let out = run_stakgraph(&["deps"]);
    assert_ne!(out.exit_code, 0);
}

#[test]
fn deps_invalid_type_fails() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["deps", "batch_process", "--type", "NotAType", &traits]);
    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("Unknown node type"));
}

#[test]
fn deps_unknown_function_fails() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["deps", "definitely_nonexistent_fn", &traits]);
    assert_ne!(out.exit_code, 0);
    assert!(
        out.stderr.contains("no node named") || out.stderr.contains("no Function named"),
        "stderr: {}",
        out.stderr
    );
}

#[test]
fn deps_smoke_single_file() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["deps", "batch_process", &traits]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("batch_process"),
        "stdout: {}",
        out.stdout
    );
}

#[test]
fn deps_smoke_directory() {
    let dir = fixture_path("src/testing/rust/src");
    let out = run_stakgraph(&["deps", "batch_process", &dir]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("batch_process"),
        "stdout: {}",
        out.stdout
    );
}

#[test]
fn deps_depth_zero_unlimited() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["deps", "batch_process", "--depth", "0", &traits]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("batch_process"), "stdout: {}", out.stdout);
}

#[test]
fn deps_allow_true_includes_verified_and_unverified_python_calls() {
    let file = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&["deps", "run_servers", "--allow", "true", &file]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("cleanup  [") && out.stdout.contains("main.py:37"),
        "stdout: {}",
        out.stdout
    );
    assert!(
        out.stdout.contains("chdir"),
        "stdout: {}",
        out.stdout
    );
}

#[test]
fn deps_allow_false_prefers_verified_python_calls() {
    let file = fixture_path("src/testing/python/web/main.py");
    let out = run_stakgraph(&["deps", "run_servers", "--allow", "false", &file]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("cleanup"), "stdout: {}", out.stdout);
    assert!(
        out.stdout.contains("main.py:37"),
        "stdout: {}",
        out.stdout
    );
    assert!(
        !out.stdout.contains("cleanup  [unverified]"),
        "stdout: {}",
        out.stdout
    );
}

#[test]
fn deps_skips_ts_test_framework_calls() {
    let out = run_stakgraph(&["deps", "structureFinalAnswer", "../mcp/src/repo"]);
    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        !out.stdout.contains("expect  [unverified]"),
        "stdout: {}",
        out.stdout
    );
    assert!(
        !out.stdout.contains("test  [unverified]"),
        "stdout: {}",
        out.stdout
    );
    assert!(
        !out.stdout.contains("describe  [unverified]"),
        "stdout: {}",
        out.stdout
    );
    assert!(
        !out.stdout.contains("│   │   "),
        "stdout: {}",
        out.stdout
    );
}

#[test]
fn deps_json_outputs_machine_readable_payload() {
    let traits = fixture_path("src/testing/rust/src/traits.rs");
    let out = run_stakgraph(&["--json", "deps", "batch_process", &traits]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["command"], "deps");
    assert!(payload["data"]["seeds"].is_array());
    assert!(payload["data"]["edges"].is_array());
}
