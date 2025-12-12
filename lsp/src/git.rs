use crate::utils::{remove_dir, run, run_res_in_dir};
use shared::error::{Context, Error, Result};
use std::path::Path;
use tracing::{debug, info};

pub async fn validate_git_credentials(
    repo: &str,
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

    debug!("Validating git credentials for repository");

    match run("git", &["ls-remote", "--heads", &repo_url]).await {
        Ok(_) => {
            debug!("Git credentials validation successful");
            Ok(())
        }
        Err(e) => {
            let error_msg = e.to_string().to_lowercase();

            // Check for common authentication error patterns
            if error_msg.contains("authentication failed")
                || error_msg.contains("invalid username or password")
                || error_msg.contains("bad credentials")
                || error_msg.contains("access denied")
                || error_msg.contains("unauthorized")
                || error_msg.contains("403")
                || error_msg.contains("401")
            {
                return Err(Error::Custom(format!(
                    "Git authentication failed. Please check your PAT and username. Error: {}",
                    e
                )));
            } else if error_msg.contains("repository not found") || error_msg.contains("404") {
                return Err(Error::Custom(format!(
                    "Repository not found or access denied. Error: {}",
                    e
                )));
            } else {
                return Err(Error::Custom(format!(
                    "Failed to validate git credentials: {}",
                    e
                )));
            }
        }
    }
}
pub async fn git_clone(
    repo: &str,
    path: &str,
    username: Option<String>,
    pat: Option<String>,
    commit: Option<&str>,
    branch: Option<&str>,
) -> Result<()> {
    let repo_url = if username.is_some() && pat.is_some() {
        let username = username.as_ref().unwrap();
        let pat = pat.as_ref().unwrap();
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
        remove_dir(path)?;

        // Configure git to use HTTPS with authentication instead of SSH for GitHub submodules
        // if let (Some(username), Some(pat)) = (username.as_ref(), pat.as_ref()) {
        //     let insteadof_key = format!("url.https://{}:{}@github.com/.insteadOf", username, pat);
        //     run(
        //         "git",
        //         &["config", "--global", &insteadof_key, "git@github.com:"],
        //     )
        //     .await?;
        // }

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
        let output = run("git", &clone_args).await;
        match output {
            Ok(_) => {
                tracing::info!("Cloned repo to {}", path);
                // Clean up global config with credentials
                // if let (Some(username), Some(pat)) = (username.as_ref(), pat.as_ref()) {
                //     let insteadof_key =
                //         format!("url.https://{}:{}@github.com/.insteadOf", username, pat);
                //     let _ = run("git", &["config", "--global", "--unset", &insteadof_key]).await;
                // }
            }
            Err(e) => {
                let error_msg = e.to_string().to_lowercase();

                if error_msg.contains("authentication failed")
                    || error_msg.contains("invalid username or password")
                    || error_msg.contains("bad credentials")
                    || error_msg.contains("access denied")
                    || error_msg.contains("unauthorized")
                    || error_msg.contains("403")
                    || error_msg.contains("401")
                {
                    tracing::error!("git clone authentication failed for {}: {}", repo_url, e);
                    return Err(Error::Custom(format!(
                        "Git authentication failed during clone. Please check your PAT (Personal Access Token) and username. Error: {}", 
                        e
                    )));
                } else if error_msg.contains("repository not found") || error_msg.contains("404") {
                    tracing::error!("git clone repository not found for {}: {}", repo_url, e);
                    return Err(Error::Custom(format!(
                        "Repository not found or access denied during clone. Error: {}",
                        e
                    )));
                } else {
                    tracing::error!("git clone failed for {}: {}", repo_url, e);
                    return Err(Error::Custom(format!("Git clone failed: {}", e)));
                }
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
