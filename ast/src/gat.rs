use git2::{DiffOptions, Repository, Status, StatusOptions};
use shared::error::{Context, Error, Result};
use std::path::Path;

pub fn get_changed_files(repo_path: &str, old_rev: &str, new_rev: &str) -> Result<Vec<String>> {
    // Open the repository
    let repo = Repository::open(repo_path)
        .map_err(|e| Error::not_found(format!("Failed to open git repository: {}", e)))?;

    // Look up the two commits
    let old_commit = repo
        .revparse_single(old_rev)
        .context("Failed to find old revision")?
        .peel_to_commit()
        .context("Failed to peel old revision to commit")?;
    let new_commit = repo
        .revparse_single(new_rev)
        .context("Failed to find new revision")?
        .peel_to_commit()
        .context("Failed to peel new revision to commit")?;

    // Get the trees for both commits
    let old_tree = old_commit
        .tree()
        .context("Failed to get tree for old commit")?;
    let new_tree = new_commit
        .tree()
        .context("Failed to get tree for new commit")?;

    // Create diff options
    let mut diff_opts = DiffOptions::new();

    // Get the diff between the two trees
    let diff = repo
        .diff_tree_to_tree(Some(&old_tree), Some(&new_tree), Some(&mut diff_opts))
        .context("Failed to generate diff")?;

    // Collect changed files
    let mut changed_files = Vec::new();

    // Iterate through diff deltas
    diff.foreach(
        &mut |delta, _| {
            if let Some(new_file) = delta.new_file().path() {
                if let Some(path_str) = new_file.to_str() {
                    changed_files.push(path_str.to_string());
                }
            }
            true
        },
        None,
        None,
        None,
    )
    .map_err(|e| shared::Error::internal(format!("Failed to iterate diff deltas: {e}")))?;

    Ok(changed_files)
}

pub fn get_working_tree_changes(repo_path: &str) -> Result<Vec<String>> {
    let repo = Repository::open(repo_path)
        .map_err(|e| Error::not_found(format!("Failed to open git repository: {}", e)))?;

    let mut status_opts = StatusOptions::new();
    status_opts.include_untracked(true);
    status_opts.recurse_untracked_dirs(true);

    let statuses = repo
        .statuses(Some(&mut status_opts))
        .context("Failed to get repository status")?;

    let mut changed_files = Vec::new();
    for entry in statuses.iter() {
        let status = entry.status();
        if status.intersects(
            Status::WT_NEW
                | Status::WT_MODIFIED
                | Status::WT_DELETED
                | Status::WT_RENAMED
                | Status::WT_TYPECHANGE,
        ) {
            if let Some(path) = entry.path() {
                changed_files.push(path.to_string());
            }
        }
    }

    Ok(changed_files)
}

pub fn get_staged_changes(repo_path: &str) -> Result<Vec<String>> {
    let repo = Repository::open(repo_path)
        .map_err(|e| Error::not_found(format!("Failed to open git repository: {}", e)))?;

    let mut status_opts = StatusOptions::new();
    let statuses = repo
        .statuses(Some(&mut status_opts))
        .context("Failed to get repository status")?;

    let mut changed_files = Vec::new();
    for entry in statuses.iter() {
        let status = entry.status();
        if status.intersects(
            Status::INDEX_NEW
                | Status::INDEX_MODIFIED
                | Status::INDEX_DELETED
                | Status::INDEX_RENAMED
                | Status::INDEX_TYPECHANGE,
        ) {
            if let Some(path) = entry.path() {
                changed_files.push(path.to_string());
            }
        }
    }

    Ok(changed_files)
}

pub struct CommitInfo {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: i64,
}

pub fn list_commits_for_paths(
    repo_path: &str,
    paths: &[String],
    max_count: Option<usize>,
) -> Result<Vec<CommitInfo>> {
    let repo = Repository::open(repo_path)
        .map_err(|e| Error::not_found(format!("Failed to open git repository: {}", e)))?;

    let mut revwalk = repo.revwalk().context("Failed to create revwalk")?;
    revwalk
        .push_head()
        .context("Failed to push HEAD to revwalk")?;

    let mut commits = Vec::new();
    let count_limit = max_count.unwrap_or(100);

    for oid_result in revwalk {
        if commits.len() >= count_limit {
            break;
        }

        let oid = oid_result.context("Failed to get commit OID")?;
        let commit = repo
            .find_commit(oid)
            .context("Failed to find commit")?;

        if paths.is_empty() {
            commits.push(CommitInfo {
                hash: commit.id().to_string(),
                message: commit.message().unwrap_or("").to_string(),
                author: commit.author().name().unwrap_or("").to_string(),
                timestamp: commit.time().seconds(),
            });
            continue;
        }

        let tree = commit.tree().context("Failed to get commit tree")?;
        let parent_tree = if commit.parent_count() > 0 {
            Some(
                commit
                    .parent(0)
                    .context("Failed to get parent commit")?
                    .tree()
                    .context("Failed to get parent tree")?,
            )
        } else {
            None
        };

        let mut diff_opts = DiffOptions::new();
        for path in paths {
            diff_opts.pathspec(path);
        }

        let diff = repo
            .diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), Some(&mut diff_opts))
            .context("Failed to create diff")?;

        if diff.deltas().len() > 0 {
            commits.push(CommitInfo {
                hash: commit.id().to_string(),
                message: commit.message().unwrap_or("").to_string(),
                author: commit.author().name().unwrap_or("").to_string(),
                timestamp: commit.time().seconds(),
            });
        }
    }

    Ok(commits)
}

/// Read the content of a file at a specific git revision.
/// Returns `None` if the file did not exist at that revision (e.g. it was added).
pub fn read_file_at_rev(repo_path: &str, rev: &str, file_path: &str) -> Result<Option<Vec<u8>>> {
    let repo = Repository::open(repo_path)
        .map_err(|e| Error::not_found(format!("Failed to open git repository: {}", e)))?;

    let obj = repo
        .revparse_single(rev)
        .context("Failed to find revision")?;
    let commit = obj.peel_to_commit().context("Failed to peel to commit")?;
    let tree = commit.tree().context("Failed to get tree")?;

    let entry = match tree.get_path(std::path::Path::new(file_path)) {
        Ok(e) => e,
        Err(_) => return Ok(None),
    };

    let blob = entry
        .to_object(&repo)
        .context("Failed to get object")?
        .into_blob()
        .map_err(|_| Error::internal("Entry is not a blob"))?;

    Ok(Some(blob.content().to_vec()))
}

pub fn filter_paths_by_scope(files: Vec<String>, scope: &[String]) -> Vec<String> {
    if scope.is_empty() {
        return files;
    }

    files
        .into_iter()
        .filter(|file| {
            let file_path = Path::new(file);
            scope.iter().any(|scope_path| {
                let scope_p = Path::new(scope_path);
                file_path.starts_with(scope_p)
            })
        })
        .collect()
}
