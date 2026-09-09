mod common;

use std::fs;
use std::path::Path;
use std::process::Command;

use common::run_stakgraph_in_cwd;
use serde_json::Value;

fn run_cmd(cwd: &Path, args: &[&str]) {
    let output = Command::new(args[0])
        .args(&args[1..])
        .current_dir(cwd)
        .output()
        .expect("failed to run command");
    assert!(
        output.status.success(),
        "command failed: {:?}\nstdout: {}\nstderr: {}",
        args,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn write_file(repo: &Path, rel: &str, content: &str) {
    let full = repo.join(rel);
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).expect("create dir failed");
    }
    fs::write(full, content).expect("write file failed");
}

fn init_git_repo() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir failed");
    let root = dir.path();

    run_cmd(root, &["git", "init"]);
    run_cmd(root, &["git", "config", "user.email", "test@example.com"]);
    run_cmd(root, &["git", "config", "user.name", "Test User"]);

    write_file(root, "src/lib.rs", "pub fn one() -> i32 {\n    1\n}\n");
    run_cmd(root, &["git", "add", "."]);
    run_cmd(root, &["git", "commit", "-m", "initial"]);

    write_file(
        root,
        "src/lib.rs",
        "pub fn one() -> i32 {\n    2\n}\n\npub fn two() -> i32 {\n    one()\n}\n",
    );

    dir
}

#[test]
fn changes_list_smoke() {
    let repo = init_git_repo();
    let cwd = repo.path().to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(&cwd, &["changes", "list"]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("Found"));
    assert!(out.stdout.contains("commits affecting"));
}

#[test]
fn changes_diff_working_tree_smoke() {
    let repo = init_git_repo();
    let cwd = repo.path().to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(&cwd, &["changes", "diff"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    // Verify working-tree file detection works; graph-level diff output varies
    // depending on whether tree-sitter resolves nodes from isolated temp files.
    assert!(
        out.stdout.contains("file(s) changed"),
        "expected file-count header; stdout: {}\nstderr: {}",
        out.stdout,
        out.stderr
    );
}

#[test]
fn changes_diff_staged_mode_smoke() {
    let repo = init_git_repo();
    let root = repo.path();

    run_cmd(root, &["git", "add", "src/lib.rs"]);

    let cwd = root.to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(&cwd, &["changes", "diff", "--staged"]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("staged changes"));
}

#[test]
fn changes_diff_last_and_types_filter_smoke() {
    let repo = init_git_repo();
    let root = repo.path();

    run_cmd(root, &["git", "add", "src/lib.rs"]);
    run_cmd(root, &["git", "commit", "-m", "second"]);

    write_file(
        root,
        "src/lib.rs",
        "pub fn one() -> i32 {\n    3\n}\n\npub fn two() -> i32 {\n    one()\n}\n",
    );

    let cwd = root.to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(
        &cwd,
        &["changes", "diff", "--last", "1", "--types", "FUNCTION"],
    );

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("last 1 commit(s)"),
        "stdout: {}\nstderr: {}",
        out.stdout,
        out.stderr
    );
    // HEAD~1 had only fn one; HEAD added fn two — graph diff must report file changed
    assert!(
        out.stdout.contains("file(s) changed"),
        "expected file-count header; stdout: {}\nstderr: {}",
        out.stdout,
        out.stderr
    );
}

#[test]
fn changes_diff_invalid_range_fails() {
    let repo = init_git_repo();
    let cwd = repo.path().to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(&cwd, &["changes", "diff", "--range", "invalid"]);

    assert_ne!(out.exit_code, 0);
    assert!(out.stderr.contains("range must be in format"));
}

#[test]
fn changes_diff_scope_warning_for_missing_path() {
    let repo = init_git_repo();
    let cwd = repo.path().to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(&cwd, &["changes", "diff", "not/a/real/path"]);

    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.contains("warning:"));
    assert!(out
        .stdout
        .contains("No changes found in the specified scope"));
}

fn init_git_repo_with_path_scope_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir failed");
    let root = dir.path();

    run_cmd(root, &["git", "init"]);
    run_cmd(root, &["git", "config", "user.email", "test@example.com"]);
    run_cmd(root, &["git", "config", "user.name", "Test User"]);

    write_file(root, "src/app/main.rs", "pub fn app() -> i32 {\n    1\n}\n");
    write_file(
        root,
        "src/app-utils/main.rs",
        "pub fn app_utils() -> i32 {\n    1\n}\n",
    );
    run_cmd(root, &["git", "add", "."]);
    run_cmd(root, &["git", "commit", "-m", "initial"]);

    write_file(root, "src/app/main.rs", "pub fn app() -> i32 {\n    2\n}\n");
    write_file(
        root,
        "src/app-utils/main.rs",
        "pub fn app_utils() -> i32 {\n    2\n}\n",
    );

    dir
}

#[test]
fn changes_diff_scope_does_not_match_sibling_prefix() {
    let repo = init_git_repo_with_path_scope_fixture();
    let cwd = repo.path().to_string_lossy().to_string();

    let out = run_stakgraph_in_cwd(&cwd, &["changes", "diff", "src/app"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("Found 1 file(s) changed"),
        "expected exact scoped file count; stdout: {}\nstderr: {}",
        out.stdout,
        out.stderr
    );
    assert!(
        !out.stdout.contains("Found 2 file(s) changed"),
        "unexpected sibling scope match count; stdout: {}\nstderr: {}",
        out.stdout,
        out.stderr
    );
}

#[test]
fn changes_diff_scope_normalizes_dot_slash_and_trailing_slash() {
    let repo = init_git_repo_with_path_scope_fixture();
    let cwd = repo.path().to_string_lossy().to_string();

    let out = run_stakgraph_in_cwd(&cwd, &["changes", "diff", "./src/app/"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    assert!(
        out.stdout.contains("Found 1 file(s) changed"),
        "expected normalized scope to match app dir only; stdout: {}\nstderr: {}",
        out.stdout,
        out.stderr
    );
}

fn init_git_repo_with_committed_node_changes() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir failed");
    let root = dir.path();

    run_cmd(root, &["git", "init"]);
    run_cmd(root, &["git", "config", "user.email", "test@example.com"]);
    run_cmd(root, &["git", "config", "user.name", "Test User"]);

    write_file(
        root,
        "src/lib.rs",
        "pub fn one() -> i32 {\n    1\n}\n\npub fn gone() -> i32 {\n    0\n}\n",
    );
    run_cmd(root, &["git", "add", "."]);
    run_cmd(root, &["git", "commit", "-m", "initial"]);

    write_file(
        root,
        "src/lib.rs",
        "pub fn one() -> i32 {\n    2\n}\n\npub fn two() -> i32 {\n    one()\n}\n",
    );
    run_cmd(root, &["git", "add", "."]);
    run_cmd(root, &["git", "commit", "-m", "second"]);

    dir
}

fn node_names(payload: &Value, key: &str) -> Vec<String> {
    payload["data"][key]
        .as_array()
        .expect("node array")
        .iter()
        .filter_map(|n| {
            n["name"]
                .as_str()
                .or_else(|| n["after"]["name"].as_str())
                .map(str::to_string)
        })
        .collect()
}

#[test]
fn changes_diff_last_reports_removed_and_modified_nodes() {
    let repo = init_git_repo_with_committed_node_changes();
    let cwd = repo.path().to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(&cwd, &["--json", "changes", "diff", "--last", "1"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");
    assert_eq!(payload["ok"], true);

    let added = node_names(&payload, "added_nodes");
    let removed = node_names(&payload, "removed_nodes");
    let modified = node_names(&payload, "modified_nodes");

    assert!(
        added.iter().any(|n| n == "two"),
        "fn two must be reported added; stdout: {}",
        out.stdout
    );
    assert!(
        removed.iter().any(|n| n == "gone"),
        "fn gone must be reported removed; stdout: {}",
        out.stdout
    );
    assert!(
        modified.iter().any(|n| n == "one"),
        "fn one must be reported modified; stdout: {}",
        out.stdout
    );
    assert!(
        !added.iter().any(|n| n == "one"),
        "fn one must not be reported added; stdout: {}",
        out.stdout
    );
    assert!(
        payload["data"]["summary"]["nodes_removed"].as_u64().unwrap_or(0) >= 1
            && payload["data"]["summary"]["nodes_modified"].as_u64().unwrap_or(0) >= 1,
        "summary must count removed and modified nodes; stdout: {}",
        out.stdout
    );
}

fn init_git_repo_with_rewired_call() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir failed");
    let root = dir.path();

    run_cmd(root, &["git", "init"]);
    run_cmd(root, &["git", "config", "user.email", "test@example.com"]);
    run_cmd(root, &["git", "config", "user.name", "Test User"]);

    write_file(
        root,
        "src/lib.rs",
        "pub fn one() -> i32 {\n    1\n}\n\npub fn caller() -> i32 {\n    one()\n}\n",
    );
    run_cmd(root, &["git", "add", "."]);
    run_cmd(root, &["git", "commit", "-m", "initial"]);

    write_file(
        root,
        "src/lib.rs",
        "pub fn one() -> i32 {\n    1\n}\n\npub fn helper() -> i32 {\n    5\n}\n\npub fn caller() -> i32 {\n    helper()\n}\n",
    );
    run_cmd(root, &["git", "add", "."]);
    run_cmd(root, &["git", "commit", "-m", "rewire caller"]);

    dir
}

fn edge_pairs(payload: &Value, key: &str) -> Vec<(String, String)> {
    payload["data"][key]
        .as_array()
        .expect("edge array")
        .iter()
        .filter_map(|e| {
            Some((
                e["source_name"].as_str()?.to_string(),
                e["target_name"].as_str()?.to_string(),
            ))
        })
        .collect()
}

#[test]
fn changes_diff_range_reports_edge_changes() {
    let repo = init_git_repo_with_rewired_call();
    let cwd = repo.path().to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(
        &cwd,
        &["--json", "changes", "diff", "--range", "HEAD~1..HEAD"],
    );

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");

    let removed = edge_pairs(&payload, "removed_edges");
    let added = edge_pairs(&payload, "added_edges");
    assert!(
        removed
            .iter()
            .any(|(s, t)| s == "caller" && t == "one"),
        "call edge caller -> one must be reported removed; stdout: {}",
        out.stdout
    );
    assert!(
        added
            .iter()
            .any(|(s, t)| s == "caller" && t == "helper"),
        "call edge caller -> helper must be reported added; stdout: {}",
        out.stdout
    );
}

#[test]
fn changes_list_json_outputs_machine_readable_payload() {
    let repo = init_git_repo();
    let cwd = repo.path().to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(&cwd, &["--json", "changes", "list"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["command"], "changes");
    assert!(payload["data"]["commits"].is_array());
}

#[test]
fn changes_diff_json_outputs_machine_readable_payload() {
    let repo = init_git_repo();
    let cwd = repo.path().to_string_lossy().to_string();
    let out = run_stakgraph_in_cwd(&cwd, &["--json", "changes", "diff"]);

    assert_eq!(out.exit_code, 0, "stderr: {}", out.stderr);
    let payload: Value = serde_json::from_str(&out.stdout).expect("valid json stdout");
    assert_eq!(payload["ok"], true);
    assert_eq!(payload["command"], "changes");
    assert!(payload["data"]["summary"].is_object());
    assert!(payload["data"]["files"].is_array());
}
