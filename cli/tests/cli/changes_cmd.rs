mod common;

use std::fs;
use std::path::Path;
use std::process::Command;

use common::run_stakgraph_in_cwd;

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

    write_file(
        root,
        "src/lib.rs",
        "pub fn one() -> i32 {\n    1\n}\n",
    );
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
    let out = run_stakgraph_in_cwd(&cwd, &["changes", "diff", "--last", "1", "--types", "FUNCTION"]);

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
    assert!(out.stdout.contains("No changes found in the specified scope"));
}
