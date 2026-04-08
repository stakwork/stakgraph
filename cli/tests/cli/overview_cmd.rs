mod common;

use std::fs;

use common::{fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn overview_directory_smoke() {
    let ts_dir = fixture_path("src/testing/typescript");
    let out = run_stakgraph(&["overview", &ts_dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("Overview:"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("typescript/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("src/"), "stdout: {}", out.stdout);
}

#[test]
fn overview_collapses_repetitive_dirs() {
    let ts_dir = fixture_path("src/testing/typescript");
    let out = run_stakgraph(&["overview", &ts_dir, "--max-lines", "40"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains(".ts)") || out.stdout.contains("files:"),
        "stdout: {}",
        out.stdout
    );
}

#[test]
fn overview_json_output() {
    let ts_dir = fixture_path("src/testing/typescript");
    let out = run_stakgraph(&["--json", "overview", &ts_dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["command"], "overview");
    assert_eq!(payload["data"]["mode"], "overview");
    assert!(payload["data"]["tree"].is_string());
}

#[test]
fn overview_collapses_large_mixed_source_dirs() {
    let temp = tempfile::tempdir().expect("tempdir failed");
    let root = temp.path();
    let components = root.join("src/components");
    fs::create_dir_all(&components).expect("create dir failed");
    fs::write(root.join("package.json"), "{}\n").expect("write package failed");

    for i in 0..12 {
        fs::write(
            components.join(format!("Comp{i}.tsx")),
            "export function Comp() { return null; }\n",
        )
        .expect("write tsx failed");
    }
    for i in 0..4 {
        fs::write(
            components.join(format!("util{i}.ts")),
            "export const value = 1;\n",
        )
        .expect("write ts failed");
    }
    fs::write(components.join("styles.css"), ".root {}\n").expect("write css failed");

    let root_str = root.to_string_lossy().to_string();
    let out = run_stakgraph(&["overview", &root_str, "--max-lines", "30"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("components/  ("),
        "stdout: {}",
        out.stdout
    );
    assert!(
        out.stdout.contains(".tsx") || out.stdout.contains(".ts"),
        "stdout: {}",
        out.stdout
    );
}

#[test]
fn overview_invalid_path_fails() {
    let out = run_stakgraph(&["overview", "./definitely-not-a-real-path-xyz"]);
    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("path does not exist"));
}
