use crate::utils::{run, run_res_in_dir};
use shared::error::{Context, Result};
use std::path::Path;
use tracing::info;

pub async fn git_clone(
    repo: &str,
    path: &str,
    username: Option<String>,
    pat: Option<String>,
    commit: Option<&str>,
) -> Result<()> {
    let repo_url = if username.is_some() && pat.is_some() {
        let username = username.unwrap();
        let pat = pat.unwrap();
        let repo_end = &repo.to_string()[8..];
        format!("https://{}:{}@{}", username, pat, repo_end)
    } else {
        repo.to_string()
    };
    let repo_path = Path::new(path);

    if repo_path.exists() && repo_path.join(".git").exists() {
        info!("Repository exists at {}, pulling latest changes", path);
        run_res_in_dir("git", &["pull"], path).await?;
    } else {
        info!("Repository doesn't exist at {}, cloning it", path);
        let output = run("git", &["clone", &repo_url, "--single-branch", path]).await;
        match output {
            Ok(_) => {
                tracing::info!("Cloned repo to {}", path);
            }
            Err(e) => {
                tracing::error!("git clone failed for {}: {}", repo_url, e);
                return Err(e);
            }
        }
    }
    if let Some(commit) = commit {
        checkout_commit(path, commit)
            .await
            .context("git checkout failed")?;
    }
    Ok(())
}

pub async fn get_commit_hash(dir: &str) -> Result<String> {
    let log = run_res_in_dir("git", &["log", "-1"], dir).await?;
    let hash = log
        .lines()
        .next()
        .context("empty res")?
        .split_whitespace()
        .nth(1)
        .context("no hash")?;
    Ok(hash.to_string())
}

pub async fn push(msg: &str, branch: &str) -> Result<()> {
    run("git", &["add", "."]).await?;
    run("git", &["commit", "-m", msg]).await?;
    run("git", &["push", "origin", branch]).await?;
    Ok(())
}
pub async fn checkout_commit(repo_path: &str, commit: &str) -> Result<()> {
    crate::utils::run_res_in_dir("git", &["checkout", commit], repo_path).await?;
    Ok(())
}

pub async fn get_changed_files_between(
    repo_path: &str,
    old_commit: &str,
    new_commit: &str,
) -> Result<Vec<String>> {
    let output = crate::utils::run_res_in_dir(
        "git",
        &["diff", "--name-only", old_commit, new_commit],
        repo_path,
    )
    .await?;
    Ok(output.lines().map(|s| s.to_string()).collect())
}
