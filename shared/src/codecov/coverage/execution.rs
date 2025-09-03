use crate::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub struct CommandRunner;

impl CommandRunner {
    pub fn run(repo_path: &Path, cmd: &str, args: &[&str]) -> Result<()> {
        use std::process::Command;
        let output = Command::new(cmd)
            .args(args)
            .current_dir(repo_path)
            .output()
            .map_err(|e| Error::Custom(format!("Failed to execute {}: {}", cmd, e)))?;
        Self::log_output(repo_path, cmd, args, &output)?;
        if output.status.success() {
            Ok(())
        } else {
            Err(Error::Custom(format!(
                "Command failed: {} {}",
                cmd,
                args.join(" ")
            )))
        }
    }

    pub fn run_with_string_args(repo_path: &Path, cmd: &str, args: &[String]) -> Result<()> {
        let str_args: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        Self::run(repo_path, cmd, &str_args)
    }

    fn log_output(
        repo_path: &Path,
        cmd: &str,
        args: &[&str],
        output: &std::process::Output,
    ) -> Result<()> {
        let cov_dir = repo_path.join("coverage");
        let _ = fs::create_dir_all(&cov_dir);

        let mut log = String::new();
        log.push_str(&format!("$ {} {}\n", cmd, args.join(" ")));
        log.push_str(&format!("exit: {:?}\n", output.status.code()));
        if !output.stdout.is_empty() {
            log.push_str("--- stdout ---\n");
            log.push_str(&String::from_utf8_lossy(&output.stdout));
        }
        if !output.stderr.is_empty() {
            log.push_str("--- stderr ---\n");
            log.push_str(&String::from_utf8_lossy(&output.stderr));
        }

        use std::io::Write;
        if let Ok(mut f) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(cov_dir.join("coverage-run.log"))
        {
            let _ = f.write_all(log.as_bytes());
        }
        Ok(())
    }
}

pub fn find_test_files(repo_path: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut test_files = Vec::new();

    fn scan_dir(dir: &Path, test_files: &mut Vec<PathBuf>, extensions: &[&str]) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    if !["node_modules", ".git", "coverage", "target"].contains(&name) {
                        scan_dir(&path, test_files, extensions);
                    }
                } else if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.contains(".test.")
                        || name.contains(".spec.")
                        || path
                            .parent()
                            .and_then(|p| p.file_name())
                            .and_then(|s| s.to_str())
                            .map(|s| s == "test" || s == "tests" || s == "__tests__")
                            .unwrap_or(false)
                    {
                        for ext in extensions {
                            if name.ends_with(ext) {
                                test_files.push(path.clone());
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    scan_dir(repo_path, &mut test_files, extensions);
    test_files
}


