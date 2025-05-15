use crate::utils::{run, run_res_in_dir};
use anyhow::{Context, Result};
use std::path::Path;
use tracing::info;

pub async fn git_clone(
    repo: &str,
    path: &str,
    username: Option<String>,
    pat: Option<String>,
) -> Result<()> {
    let repo_url = if username.is_some() && pat.is_some() {
        let username = username.unwrap();
        let pat = pat.unwrap();
        let repo_end = &repo.to_string()[8..];
        format!("https://{}:{}@{}", username, pat, repo_end)
    } else {
        repo.to_string()
    };
    run("git", &["clone", &repo_url, "--single-branch", path]).await?;
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

pub async fn git_pull_or_clone(
    repo: &str,
    path: &str,
    username: Option<String>,
    pat: Option<String>,
) -> Result<()> {
    let repo_path = Path::new(path);

    if repo_path.exists() && repo_path.join(".git").exists() {
        info!("Repository exists at {}, pulling latest changes", path);

        run_res_in_dir("git", &["reset", "--hard", "HEAD"], path).await?;

        run_res_in_dir("git", &["pull"], path).await?;

        Ok(())
    } else {
        info!("Repository doesn't exist at {}, cloning it", path);
        git_clone(repo, path, username, pat).await
    }
}
