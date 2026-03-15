use shared::{Error, Result};

pub fn get_repo_root(start_dir: &str) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .current_dir(start_dir)
        .output()
        .map_err(|e| Error::internal(format!("Failed to run git: {}", e)))?;
    if !output.status.success() {
        return Err(Error::internal(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub struct CommitInfo {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub date: String,
}

fn run_git(repo_path: &str, args: &[&str]) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(repo_path)
        .output()
        .map_err(|e| Error::internal(format!("Failed to run git: {}", e)))?;
    if !output.status.success() {
        return Err(Error::internal(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub fn get_changed_files(repo_path: &str, old_rev: &str, new_rev: &str) -> Result<Vec<String>> {
    let out = run_git(repo_path, &["diff", "--name-only", old_rev, new_rev])?;
    Ok(out.lines().filter(|l| !l.is_empty()).map(String::from).collect())
}

pub fn get_working_tree_changes(repo_path: &str) -> Result<Vec<String>> {
    let modified = run_git(repo_path, &["diff", "--name-only", "HEAD"])?;
    let untracked = run_git(
        repo_path,
        &["ls-files", "--others", "--exclude-standard"],
    )?;
    let mut files: Vec<String> = modified
        .lines()
        .chain(untracked.lines())
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
    files.sort();
    files.dedup();
    Ok(files)
}

pub fn get_staged_changes(repo_path: &str) -> Result<Vec<String>> {
    let out = run_git(repo_path, &["diff", "--name-only", "--cached"])?;
    Ok(out.lines().filter(|l| !l.is_empty()).map(String::from).collect())
}

pub fn list_commits_for_paths(
    repo_path: &str,
    paths: &[String],
    max: Option<usize>,
) -> Result<Vec<CommitInfo>> {
    let limit = max.unwrap_or(20).to_string();
    let mut args = vec!["log", "--format=%H%x00%s%x00%an%x00%ar", "-n", &limit, "--"];
    let path_refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
    args.extend_from_slice(&path_refs);

    let out = run_git(repo_path, &args)?;

    let commits = out
        .lines()
        .filter(|l| !l.is_empty())
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(4, '\x00').collect();
            if parts.len() < 3 {
                return None;
            }
            Some(CommitInfo {
                hash: parts[0].to_string(),
                message: parts[1].to_string(),
                author: parts[2].to_string(),
                date: parts.get(3).unwrap_or(&"").to_string(),
            })
        })
        .collect();

    Ok(commits)
}

pub fn read_file_at_rev(
    repo_path: &str,
    rev: &str,
    file_path: &str,
) -> Result<Option<Vec<u8>>> {
    let spec = format!("{}:{}", rev, file_path);
    let output = std::process::Command::new("git")
        .args(["show", &spec])
        .current_dir(repo_path)
        .output()
        .map_err(|e| Error::internal(format!("Failed to run git show: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("does not exist") || stderr.contains("exists on disk") || stderr.contains("Path '") {
            return Ok(None);
        }
        let exit_code = output.status.code().unwrap_or(-1);
        if exit_code == 128 {
            return Ok(None);
        }
        return Err(Error::internal(stderr.to_string()));
    }

    Ok(Some(output.stdout))
}

pub fn filter_paths_by_scope(files: Vec<String>, scope: &[String]) -> Vec<String> {
    if scope.is_empty() {
        return files;
    }
    files
        .into_iter()
        .filter(|f| {
            scope.iter().any(|s| {
                f == s || f.starts_with(&format!("{}/", s)) || f.starts_with(s.as_str())
            })
        })
        .collect()
}
