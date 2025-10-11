use git2::{DiffOptions, Repository};
use shared::error::{Result, Error, Context};


pub fn get_changed_files(repo_path: &str, old_rev: &str, new_rev: &str) -> Result<Vec<String>> {
    // Open the repository
    let repo = Repository::open(repo_path)
        .map_err(|e| Error::Custom(format!("Failed to open git repository: {}", e)))?;

    // Look up the two commits
    let old_commit = repo.revparse_single(old_rev)
        .context("Failed to find old revision")?
        .peel_to_commit()
        .context("Failed to peel old revision to commit")?;
    let new_commit = repo.revparse_single(new_rev)
        .context("Failed to find new revision")?
        .peel_to_commit()
        .context("Failed to peel new revision to commit")?;

    // Get the trees for both commits
    let old_tree = old_commit.tree()
        .context("Failed to get tree for old commit")?;
    let new_tree = new_commit.tree()
        .context("Failed to get tree for new commit")?;

    // Create diff options
    let mut diff_opts = DiffOptions::new();

    // Get the diff between the two trees
    let diff = repo.diff_tree_to_tree(Some(&old_tree), Some(&new_tree), Some(&mut diff_opts))
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
    ).map_err(|e| shared::Error::Custom(format!("Failed to iterate diff deltas: {e}")))?;

    Ok(changed_files)
}
