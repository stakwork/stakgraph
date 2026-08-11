use crate::utils::{remove_dir, run, run_res_in_dir};
use shared::error::{Context, Error, Result};
use std::path::Path;
use tracing::{debug, info, warn};

/// Options controlling how a git repository is cloned.
///
/// `Default` (all `None`) is semantically identical to today's full-clone
/// behaviour — no `--depth` and no `--filter` are appended to the clone
/// command.
#[derive(Debug, Clone, Default)]
pub struct CloneOpts {
    /// When `Some(n)`, clone with `--depth n` (shallow clone).
    /// Must be ≥ 1; validated before reaching `git_clone`.
    pub depth: Option<u32>,
    /// When `Some(spec)`, clone with `--filter=<spec>` (partial clone).
    /// Passed through verbatim to git, e.g. `"blob:limit=1m"`.
    /// Requires the remote to support git protocol v2 /
    /// `uploadpack.allowFilter`.
    pub filter: Option<String>,
}

/// Pure helper: appends optional `--depth <n>` and `--filter=<spec>` to
/// `base_args` and returns an owned `Vec<String>`.  The function is kept
/// free of any I/O so it can be unit-tested without shelling out.
pub fn build_clone_args(
    base_args: &[&str],
    depth: Option<u32>,
    filter: Option<&str>,
) -> Vec<String> {
    let mut args: Vec<String> = base_args.iter().map(|s| s.to_string()).collect();
    if let Some(d) = depth {
        args.push("--depth".to_string());
        args.push(d.to_string());
    }
    if let Some(f) = filter {
        args.push(format!("--filter={}", f));
    }
    args
}

pub async fn validate_git_credentials(
    repo: &str,
    username: Option<String>,
    pat: Option<String>,
) -> Result<()> {
    let repo_url = match (username.as_ref(), pat.as_ref()) {
        (Some(username), Some(pat)) => {
            let repo_end = repo.strip_prefix("https://").unwrap_or(repo);
            format!("https://{}:{}@{}", username, pat, repo_end)
        }
        _ => repo.to_string(),
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
                Err(Error::auth(
                    "Git authentication failed. Please check your username and Personal Access Token."
                ))
            } else if error_msg.contains("repository not found") || error_msg.contains("404") {
                Err(Error::not_found(
                    "Repository not found or access denied. Please check the repository URL."
                ))
            } else {
                Err(Error::dependency(format!(
                    "Failed to validate git credentials: {}",
                    e.to_string().lines().next().unwrap_or("unknown error")
                )))
            }
        }
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
        return Err(Error::validation(format!(
            "Git validation failed for {} repository(ies):\n{}",
            errors.len(),
            errors.join("\n")
        )));
    }

    Ok(())
}

/// Attempt to check out `commit` in `repo_path`.  If the first attempt fails
/// (the commit may be absent from a shallow/partial clone), run a targeted
/// single-object fetch and retry once before propagating the error.
async fn checkout_commit_with_fallback(repo_path: &str, commit: &str) -> Result<()> {
    if let Err(e) = checkout_commit(repo_path, commit).await {
        warn!(
            "checkout {} failed ({:?}); attempting targeted fetch of that commit",
            commit, e
        );
        // Fetch only the specific commit, deepening just enough to retrieve it.
        // We deliberately avoid `--unshallow` (which would pull full history
        // and negate the cost savings of a shallow/partial clone).
        let _ = run_res_in_dir(
            "git",
            &["fetch", "origin", commit, "--depth", "1"],
            repo_path,
        )
        .await;
        // Retry after the fetch; surface the original error if this also fails.
        checkout_commit(repo_path, commit)
            .await
            .context("git checkout failed even after targeted fetch")?;
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
    opts: &CloneOpts,
) -> Result<()> {
    let repo_url = match (username.as_ref(), pat.as_ref()) {
        (Some(username), Some(pat)) => {
            let repo_end = repo.strip_prefix("https://").unwrap_or(repo);
            format!("https://{}:{}@{}", username, pat, repo_end)
        }
        _ => repo.to_string(),
    };
    let repo_path = Path::new(path);

    if repo_path.exists() && repo_path.join(".git").exists() {
        info!("Repository exists at {}, pulling latest changes", path);
        run_res_in_dir("git", &["pull"], path).await?;
        // Existing-repo path: commit checkout still needs the targeted-fetch
        // fallback because this repo may already be a shallow clone.
        if let Some(commit) = commit {
            checkout_commit_with_fallback(path, commit).await?;
        }
    } else {
        info!(
            "cloning {} with depth={:?} filter={:?}",
            repo, opts.depth, opts.filter
        );
        remove_dir(path)?;

        let base_args: Vec<&str> = vec!["clone", &repo_url, "--single-branch", "--recurse-submodules"];

        // Build args using the pure helper so argument construction is testable.
        // We need owned strings for depth/filter; hold them here so references stay valid.
        let depth_str;
        let filter_str;
        let mut base: Vec<String> = vec![
            "clone".to_string(),
            repo_url.clone(),
            "--single-branch".to_string(),
            "--recurse-submodules".to_string(),
        ];
        if let Some(b) = branch {
            base.push("--branch".to_string());
            base.push(b.to_string());
        }
        let base_refs: Vec<&str> = base.iter().map(String::as_str).collect();
        let mut final_args =
            build_clone_args(&base_refs, opts.depth, opts.filter.as_deref());
        final_args.push(path.to_string());

        // Drop base_args (unused placeholder above)
        let _ = base_args;
        depth_str = opts.depth.map(|d| d.to_string()).unwrap_or_default();
        filter_str = opts.filter.clone().unwrap_or_default();

        let final_arg_strs: Vec<&str> = final_args.iter().map(String::as_str).collect();
        let output = run("git", &final_arg_strs).await;
        match output {
            Ok(_) => {
                tracing::info!("Cloned repo to {}", path);
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
                    tracing::error!("git clone authentication failed");
                    return Err(Error::auth(
                        "Git authentication failed during clone. Please check your PAT (Personal Access Token) and username.".to_string(),
                    ));
                } else if error_msg.contains("repository not found")
                    || error_msg.contains("404")
                {
                    tracing::error!("git clone repository not found or access denied");
                    return Err(Error::not_found(
                        "Repository not found or access denied during clone.".to_string(),
                    ));
                } else {
                    tracing::error!(
                        "git clone failed (depth={:?} filter={:?}): {}",
                        opts.depth,
                        opts.filter,
                        e
                    );
                    let hint = if opts.filter.is_some() {
                        " Note: partial-clone --filter requires the remote to support \
git protocol v2 / uploadpack.allowFilter."
                    } else {
                        ""
                    };
                    return Err(Error::dependency(format!(
                        "Git clone failed (depth={} filter={}).{}",
                        if depth_str.is_empty() {
                            "none".to_string()
                        } else {
                            depth_str
                        },
                        if filter_str.is_empty() {
                            "none".to_string()
                        } else {
                            filter_str
                        },
                        hint
                    )));
                }
            }
        }

        if let Some(commit) = commit {
            checkout_commit_with_fallback(path, commit).await?;
        }
    }
    Ok(())
}

pub async fn get_commit_hash(dir: &str) -> Result<String> {
    let log = run_res_in_dir("git", &["log", "-1"], dir)
        .await
        .map_err(|e| {
            let error_msg = e.to_string().to_lowercase();
            if error_msg.contains("no such file or directory") {
                Error::not_found(format!(
                    "Repository directory '{}' not found or incomplete. Error: {}",
                    dir, e
                ))
            } else if error_msg.contains("not a git repository") {
                Error::validation(format!(
                    "Directory '{}' is not a valid git repository. Error: {}",
                    dir, e
                ))
            } else {
                Error::dependency(format!("Failed to get commit hash from '{}': {}", dir, e))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_clone_args_neither() {
        let base = &["clone", "https://example.com/repo.git", "--single-branch", "--recurse-submodules"];
        let args = build_clone_args(base, None, None);
        assert_eq!(
            args,
            vec![
                "clone",
                "https://example.com/repo.git",
                "--single-branch",
                "--recurse-submodules",
            ]
        );
    }

    #[test]
    fn build_clone_args_depth_only() {
        let base = &["clone", "https://example.com/repo.git", "--single-branch", "--recurse-submodules"];
        let args = build_clone_args(base, Some(1), None);
        assert_eq!(
            args,
            vec![
                "clone",
                "https://example.com/repo.git",
                "--single-branch",
                "--recurse-submodules",
                "--depth",
                "1",
            ]
        );
    }

    #[test]
    fn build_clone_args_filter_only() {
        let base = &["clone", "https://example.com/repo.git", "--single-branch", "--recurse-submodules"];
        let args = build_clone_args(base, None, Some("blob:limit=1m"));
        assert_eq!(
            args,
            vec![
                "clone",
                "https://example.com/repo.git",
                "--single-branch",
                "--recurse-submodules",
                "--filter=blob:limit=1m",
            ]
        );
    }

    #[test]
    fn build_clone_args_both() {
        let base = &["clone", "https://example.com/repo.git", "--single-branch", "--recurse-submodules"];
        let args = build_clone_args(base, Some(5), Some("blob:limit=1m"));
        assert_eq!(
            args,
            vec![
                "clone",
                "https://example.com/repo.git",
                "--single-branch",
                "--recurse-submodules",
                "--depth",
                "5",
                "--filter=blob:limit=1m",
            ]
        );
    }
}
