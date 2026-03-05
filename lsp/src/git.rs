use crate::utils::{remove_dir, run, run_res_in_dir};
use shared::error::{Context, Error, Result};
use std::path::Path;
use tracing::{debug, info};

fn required(username: &Option<String>, pat: &Option<String>) -> Result<()> {
    match (username.as_ref(), pat.as_ref()) {
        (Some(_), None) | (None, Some(_)) => Err(Error::Custom(
            "Both username and PAT must be provided together, or neither".to_string(),
        )),
        _ => Ok(()),
    }
}

fn build_auth(repo: &str, username: &str, pat: &str) -> Result<String> {
    if !repo.starts_with("https://") {
        return Err(Error::Custom(format!(
            "Authenticated clones require an https:// URL, got: {}",
            &repo[..repo.len().min(32)]
        )));
    }
    let rest = &repo["https://".len()..];
    Ok(format!("https://{}:{}@{}", username, pat, rest))
}

fn classify_git_error(e: &Error) -> Error {
    let msg = e.to_string().to_lowercase();
    if msg.contains("authentication failed")
        || msg.contains("invalid username or password")
        || msg.contains("bad credentials")
        || msg.contains("access denied")
        || msg.contains("unauthorized")
        || msg.contains("403")
        || msg.contains("401")
    {
        Error::Custom(
            "Git authentication failed. Please check your PAT and username.".to_string(),
        )
    } else if msg.contains("repository not found") || msg.contains("404") {
        Error::Custom("Repository not found or access denied.".to_string())
    } else {
        Error::Custom(format!("Git operation failed: {}", e))
    }
}

pub async fn validate_git_credentials(
    repo: &str,
    username: Option<String>,
    pat: Option<String>,
) -> Result<()> {
    required(&username, &pat)?;
    let repo_url = match (username.as_ref(), pat.as_ref()) {
        (Some(u), Some(p)) => build_auth(repo, u, p)?,
        _ => repo.to_string(),
    };
    debug!("Validating git credentials for repository");

    match run("git", &["ls-remote", "--heads", &repo_url]).await {
        Ok(_) => {
            debug!("Git credentials validation successful");
            Ok(())
        }
        Err(e) => Err(classify_git_error(&e)),
    }
}

pub async fn validate_git_credentials_multi(
    repos: &[String],
    username: Option<String>,
    pat: Option<String>,
) -> Result<()> {
    let mut errors = Vec::new();

    for repo in repos {
        if let Err(e) = validate_git_credentials(repo, username.clone(), pat.clone()).await {
            errors.push(format!("Repo '{}': {}", repo, e));
        }
    }

    if !errors.is_empty() {
        return Err(Error::Custom(format!(
            "Git validation failed for {} repository(ies):\n{}",
            errors.len(),
            errors.join("\n")
        )));
    }

    Ok(())
}
pub async fn git_clone(
    repo: &str,
    path: &str,
    username: Option<String>,
    pat: Option<String>,
    commit: Option<&str>,
    branch: Option<&str>,
) -> Result<()> {
    required(&username, &pat)?;
    let repo_url = match (username.as_ref(), pat.as_ref()) {
        (Some(u), Some(p)) => build_auth(repo, u, p)?,
        _ => repo.to_string(),
    };
    let repo_path = Path::new(path);

    if repo_path.exists() && repo_path.join(".git").exists() {
        info!("Repository exists at {}, pulling latest changes", path);
        run_res_in_dir("git", &["pull"], path).await?;
    } else {
        info!("Repository doesn't exist at {}, cloning it", path);
        remove_dir(path)?;

        let mut clone_args = vec![
            "clone",
            &repo_url,
            "--single-branch",
            "--recurse-submodules",
        ];
        if let Some(branch) = branch {
            clone_args.extend(&["--branch", branch]);
        }
        clone_args.push(path);
        run("git", &clone_args)
            .await
            .map_err(|e| classify_git_error(&e))?;
        tracing::info!("Cloned repo to {}", path);
    }
    if let Some(commit) = commit {
        checkout_commit(path, commit)
            .await
            .context("git checkout failed")?;
    }
    Ok(())
}

pub async fn get_commit_hash(dir: &str) -> Result<String> {
    let log = run_res_in_dir("git", &["log", "-1"], dir)
        .await
        .map_err(|e| {
            let error_msg = e.to_string().to_lowercase();
            if error_msg.contains("no such file or directory") {
                Error::Custom(format!(
                    "Repository directory '{}' not found or incomplete. Error: {}",
                    dir, e
                ))
            } else if error_msg.contains("not a git repository") {
                Error::Custom(format!(
                    "Directory '{}' is not a valid git repository. Error: {}",
                    dir, e
                ))
            } else {
                Error::Custom(format!("Failed to get commit hash from '{}': {}", dir, e))
            }
        })?;
    let hash = log
        .lines()
        .next()
        .context("empty git log result")?
        .split_whitespace()
        .nth(1)
        .context("no commit hash found in git log")?;
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
