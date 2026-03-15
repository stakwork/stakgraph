mod common;

use common::{fixture_path, run_stakgraph};

#[test]
fn summarize_single_file_smoke() {
    let routes = fixture_path("src/testing/typescript/src/routes.ts");
    let out = run_stakgraph(&["summarize", &routes]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Summary:"));
    assert!(out.stdout.contains("File:"));
}

#[test]
fn summarize_directory_with_depth() {
    let rust_dir = fixture_path("src/testing/rust");
    let out = run_stakgraph(&["summarize", "--depth", "1", &rust_dir]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Directory Structure"));
    assert!(out.stdout.contains("File Summaries"));
}

#[test]
fn summarize_token_budget_footer_present() {
    let rust_dir = fixture_path("src/testing/rust");
    let out = run_stakgraph(&["summarize", "--max-tokens", "80", &rust_dir]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("tokens"));
}

#[test]
fn summarize_invalid_path_fails() {
    let out = run_stakgraph(&["summarize", "./definitely-not-a-real-path-xyz"]);

    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("path does not exist"));
}
