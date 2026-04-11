mod common;

use std::fs;

use common::{fixture_path, run_stakgraph};
use serde_json::Value;

#[test]
fn overview_directory_smoke() {
    let ts_dir = fixture_path("src/testing/typescript");
    let out = run_stakgraph(&["overview", &ts_dir]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("typescript/"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("src/"), "stdout: {}", out.stdout);
    assert!(
        out.stdout.contains("├── ") || out.stdout.contains("└── "),
        "stdout: {}",
        out.stdout
    );
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

#[test]
fn overview_grep_filters_tree() {
    let temp = tempfile::tempdir().expect("tempdir failed");
    let root = temp.path();
    fs::create_dir_all(root.join("src/auth")).expect("create dir failed");
    fs::create_dir_all(root.join("src/utils")).expect("create dir failed");
    fs::write(root.join("src/auth/handler.ts"), "export function auth() {}").expect("write failed");
    fs::write(root.join("src/utils/helpers.ts"), "export function help() {}").expect("write failed");

    let root_str = root.to_string_lossy().to_string();
    let out = run_stakgraph(&["overview", &root_str, "--grep", "auth"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("auth"), "stdout: {}", out.stdout);
    assert!(
        !out.stdout.contains("helpers.ts"),
        "helpers.ts should be filtered: {}",
        out.stdout
    );
}

#[test]
fn overview_json_fingerprint_field() {
    let temp = tempfile::tempdir().expect("tempdir failed");
    let root = temp.path();
    fs::write(root.join("package.json"), r#"{"dependencies":{"react":"^18"}}"#)
        .expect("write failed");
    fs::create_dir_all(root.join("src")).expect("create dir failed");
    fs::write(root.join("src/index.ts"), "export const x = 1;").expect("write failed");

    let root_str = root.to_string_lossy().to_string();
    let out = run_stakgraph(&["--json", "overview", &root_str]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json");
    assert!(
        payload["data"]["fingerprint"].is_string(),
        "fingerprint should be present: {:?}",
        payload
    );
    assert!(
        payload["data"]["fingerprint"]
            .as_str()
            .unwrap()
            .contains("Node.js"),
        "fingerprint should mention Node.js: {:?}",
        payload["data"]["fingerprint"]
    );
}

#[test]
fn overview_zoom_expands_subdirectory() {
    let temp = tempfile::tempdir().expect("tempdir failed");
    let root = temp.path();
    fs::create_dir_all(root.join("src/api")).expect("create dir failed");
    fs::create_dir_all(root.join("src/utils")).expect("create dir failed");
    for i in 0..5 {
        fs::write(
            root.join(format!("src/api/route{i}.ts")),
            "export function route() {}",
        )
        .expect("write failed");
    }
    fs::write(root.join("src/utils/helpers.ts"), "export function h() {}").expect("write failed");

    let zoom_path = root.join("src/api").to_string_lossy().to_string();
    let out = run_stakgraph(&["overview", &zoom_path]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(out.stdout.contains("route0.ts"), "stdout: {}", out.stdout);
    assert!(out.stdout.contains("route1.ts"), "stdout: {}", out.stdout);
}
