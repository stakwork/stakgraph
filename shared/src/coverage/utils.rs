use crate::Result;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;


pub fn clone_repo(git_url: &str) -> Result<PathBuf> {
    let repo_name = get_repo_name_from_url(git_url)?;
    let repo_path = PathBuf::from("/tmp").join(format!("stakgraph_test_{}", repo_name));
    
    if repo_path.exists() {
        fs::remove_dir_all(&repo_path)?;
    }
    
    let output = Command::new("git")
        .args(&["clone", "--depth", "1", git_url, repo_path.to_str().unwrap()])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(crate::Error::Custom(format!(
            "Failed to clone repository {}: {}", git_url, stderr
        )));
    }

    Ok(repo_path)
}


pub fn get_repo_name_from_url(git_url: &str) -> Result<String> {
    let url = git_url.trim_end_matches('/');
    let name = url.split('/').last()
        .ok_or_else(|| crate::Error::Custom("Invalid git URL".to_string()))?;

    let clean_name = name.strip_suffix(".git").unwrap_or(name);
    Ok(clean_name.to_string())
}


pub fn dir_exists_and_not_empty(path: &Path) -> bool {
    path.exists() && path.is_dir() && 
    path.read_dir().map(|mut i| i.next().is_some()).unwrap_or(false)
}

pub fn run_command(cmd: &str, args: &[&str], working_dir: &Path) -> Result<()> {
    let output = Command::new(cmd)
        .args(args)
        .current_dir(working_dir)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(crate::Error::Custom(format!(
            "Command '{}' failed in {:?}:\nSTDERR: {}\nSTDOUT: {}", 
            cmd, working_dir, stderr, stdout
        )));
    }

    Ok(())
}

pub fn install_dependencies(repo_path: &Path) -> Result<()> {
    if repo_path.join("pnpm-lock.yaml").exists() {
        run_command("pnpm", &["install", "--dev"], repo_path)
    } else if repo_path.join("yarn.lock").exists() {
        run_command("yarn", &["install"], repo_path)
    } else if repo_path.join("package-lock.json").exists() {
        run_command("npm", &["install"], repo_path)
    } else if repo_path.join("package.json").exists() {
        run_command("npm", &["install"], repo_path)
    } else {
        Err(crate::Error::Custom("No package.json found for dependency installation".to_string()))
    }
}

pub fn has_coverage_output(repo_path: &Path) -> bool {
    let coverage_dir = repo_path.join("coverage");
    coverage_dir.exists() && (
        coverage_dir.join("coverage-final.json").exists() ||
        coverage_dir.join("coverage-summary.json").exists() ||
        coverage_dir.join("lcov.info").exists()
    )
}

pub fn clean_coverage_dir(repo_path: &Path) -> Result<()> {
    let coverage_dir = repo_path.join("coverage");
    if coverage_dir.exists() {
        std::fs::remove_dir_all(&coverage_dir)?;
    }
    Ok(())
}

pub fn list_coverage_files(repo_path: &Path) -> Vec<PathBuf> {
    let coverage_dir = repo_path.join("coverage");
    let mut files = Vec::new();
    
    if let Ok(entries) = std::fs::read_dir(&coverage_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                files.push(entry.path());
            }
        }
    }
    
    files
}

pub fn cleanup_test_repo(git_url: &str) -> Result<()> {
    let repo_name = get_repo_name_from_url(git_url)?;
    let repo_path = PathBuf::from("/tmp").join(format!("stakgraph_test_{}", repo_name));
    
    if repo_path.exists() {
        std::fs::remove_dir_all(&repo_path)?;
    }
    
    Ok(())
}