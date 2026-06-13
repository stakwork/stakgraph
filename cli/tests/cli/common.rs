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

pub fn workspace_path(relative: &str) -> String {
    let p = manifest_dir().join("..").join(relative);
    std::fs::canonicalize(&p)
        .unwrap_or(p)
        .to_string_lossy()
        .to_string()
}

pub fn fixture_path(relative: &str) -> String {
    let p = manifest_dir().join("..").join("ast").join(relative);
    std::fs::canonicalize(&p)
        .unwrap_or(p)
        .to_string_lossy()
        .to_string()
}

pub fn run_stakgraph(args: &[&str]) -> CliOutput {
    let output = Command::new(env!("CARGO_BIN_EXE_stakgraph"))
        .current_dir(manifest_dir())
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
    let output = Command::new(env!("CARGO_BIN_EXE_stakgraph"))
        .current_dir(cwd)
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
    output
        .lines()
        .filter(|line| line.starts_with(prefix))
        .count()
}
