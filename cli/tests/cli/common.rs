#![allow(dead_code)]

use std::path::PathBuf;
use std::process::Command;

pub struct CliOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn manifest_path() -> PathBuf {
    manifest_dir().join("Cargo.toml")
}

pub fn workspace_path(relative: &str) -> String {
    let p = manifest_dir().join("..").join(relative);
    std::fs::canonicalize(&p).unwrap_or(p).to_string_lossy().to_string()
}

pub fn fixture_path(relative: &str) -> String {
    let p = manifest_dir().join("..").join("ast").join(relative);
    std::fs::canonicalize(&p).unwrap_or(p).to_string_lossy().to_string()
}

pub fn run_stakgraph(args: &[&str]) -> CliOutput {
    let output = Command::new("cargo")
        .current_dir(manifest_dir())
        .args([
            "run",
            "--quiet",
            "--manifest-path",
            manifest_path().to_string_lossy().as_ref(),
            "--bin",
            "stakgraph",
            "--",
        ])
        .args(args)
        .output()
        .expect("failed to run stakgraph");

    CliOutput {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
    }
}

pub fn run_stakgraph_in_cwd(cwd: &str, args: &[&str]) -> CliOutput {
    let output = Command::new("cargo")
        .current_dir(cwd)
        .args([
            "run",
            "--quiet",
            "--manifest-path",
            manifest_path().to_string_lossy().as_ref(),
            "--bin",
            "stakgraph",
            "--",
        ])
        .args(args)
        .output()
        .expect("failed to run stakgraph");

    CliOutput {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code().unwrap_or(-1),
    }
}

pub fn count_prefix(output: &str, prefix: &str) -> usize {
    output.lines().filter(|line| line.starts_with(prefix)).count()
}
