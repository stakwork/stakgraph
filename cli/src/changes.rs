use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::git::{
    filter_paths_by_scope, get_changed_files, get_staged_changes, get_working_tree_changes,
    list_commits_for_paths, read_file_at_rev,
};
use ast::lang::graphs::{ArrayGraph, Node, NodeType};
use ast::lang::Lang;
use ast::repo::{Repo, Repos};
use console::style;
use lsp::Language;
use shared::{Error, Result};

use super::args::{ChangesArgs, ChangesCommand, DiffArgs};
use super::output::Output;
use super::progress::CliSpinner;
use super::utils::common_ancestor;

pub async fn run(args: &ChangesArgs, out: &mut Output, show_progress: bool) -> Result<()> {
    let repo_path = std::env::current_dir()
        .map_err(|e| Error::internal(format!("Failed to get current directory: {}", e)))?;
    let repo_str = repo_path.to_string_lossy().to_string();

    match &args.command {
        ChangesCommand::List(list_args) => {
            run_list_commits(&repo_str, &list_args.paths, list_args.max, out, show_progress).await
        }
        ChangesCommand::Diff(diff_args) => {
            run_diff(
                &repo_str,
                &diff_args.paths,
                &diff_args.types,
                diff_args,
                out,
                show_progress,
            )
            .await
        }
    }
}

async fn run_list_commits(
    repo_path: &str,
    paths: &[String],
    max: usize,
    out: &mut Output,
    show_progress: bool,
) -> Result<()> {
    let spinner = if show_progress {
        Some(CliSpinner::new("Scanning commit history..."))
    } else {
        None
    };
    let commits = list_commits_for_paths(repo_path, paths, Some(max))?;
    if let Some(sp) = &spinner {
        sp.finish_and_clear();
    }

    if commits.is_empty() {
        out.writeln(format!("{}", style("No commits found for the specified paths").yellow()))?;
        return Ok(());
    }

    let scope_label = if paths.is_empty() {
        "all files".to_string()
    } else {
        paths.join(", ")
    };

    out.writeln(format!(
        "{} {} commits affecting: {}",
        style("Found").bold().cyan(),
        style(commits.len()).bold().green(),
        style(&scope_label).cyan()
    ))?;
    out.newline()?;

    for commit in commits {
        let short_hash = &commit.hash[..7.min(commit.hash.len())];
        let message_line = commit.message.lines().next().unwrap_or("");

        out.writeln(format!(
            "{} {} {}",
            style(short_hash).yellow(),
            style(&commit.date).dim(),
            style(message_line).white()
        ))?;
        out.writeln(format!(
            "  {} {}",
            style("Author:").dim(),
            style(&commit.author).cyan()
        ))?;
    }

    Ok(())
}

async fn run_diff(
    repo_path: &str,
    paths: &[String],
    types: &[String],
    args: &DiffArgs,
    out: &mut Output,
    show_progress: bool,
) -> Result<()> {
    let mode_description = if args.staged {
        "staged changes".to_string()
    } else if let Some(n) = args.last {
        format!("last {} commit(s)", n)
    } else if let Some(ref r) = args.since {
        format!("changes since {}", r)
    } else if let Some(ref r) = args.range {
        format!("range {}", r)
    } else {
        "working tree changes".to_string()
    };

    let type_scope = if types.is_empty() {
        "all node types".to_string()
    } else {
        format!("{}", types.join(", "))
    };

    let spinner = if show_progress {
        Some(CliSpinner::new(&format!(
            "Preparing {} summary for {}...",
            mode_description, type_scope
        )))
    } else {
        None
    };

    // before_rev: the "old" snapshot; after_rev: Some(rev) means read from git, None means read from disk
    if let Some(sp) = &spinner {
        sp.set_message("Collecting changed files...");
    }
    let (changed_files, before_rev, after_rev) = if args.staged {
        (get_staged_changes(repo_path)?, "HEAD".to_string(), None)
    } else if let Some(last_n) = args.last {
        let old_rev = format!("HEAD~{}", last_n);
        let files = get_changed_files(repo_path, &old_rev, "HEAD")?;
        (files, old_rev, None)
    } else if let Some(ref since_ref) = args.since {
        let files = get_changed_files(repo_path, since_ref, "HEAD")?;
        (files, since_ref.clone(), None)
    } else if let Some(ref range) = args.range {
        let parts: Vec<&str> = range.split("..").collect();
        if parts.len() != 2 {
            return Err(Error::validation("Range must be in format <a>..<b>"));
        }
        let files = get_changed_files(repo_path, parts[0], parts[1])?;
        (files, parts[0].to_string(), Some(parts[1].to_string()))
    } else {
        (get_working_tree_changes(repo_path)?, "HEAD".to_string(), None)
    };

    let scoped_files = filter_paths_by_scope(changed_files, paths);

    if scoped_files.is_empty() {
        if !paths.is_empty() {
            for p in paths {
                let abs = Path::new(repo_path).join(p);
                if !abs.exists() {
                    out.writeln(format!(
                        "{}",
                        style(format!("warning: '{}' does not exist in this repository", p))
                            .yellow()
                    ))?;
                }
            }
        }
        out.writeln(format!(
            "{}",
            style("No changes found in the specified scope").yellow()
        ))?;
        return Ok(());
    }

    let scope_label = if paths.is_empty() {
        "all files".to_string()
    } else {
        paths.join(", ")
    };

    out.writeln(format!(
        "{} {} file(s) changed in {} (scope: {})",
        style("Found").bold().cyan(),
        style(scoped_files.len()).bold().green(),
        style(mode_description).yellow(),
        style(&scope_label).cyan()
    ))?;
    out.newline()?;

    let parseable_count = scoped_files
        .iter()
        .filter(|file| Language::from_path(file).is_some())
        .count();
    let mut printed_file_list = false;

    for file in &scoped_files {
        let parseable = Language::from_path(file).is_some();
        if parseable {
            if parseable_count > 5 {
                out.writeln(format!("  {}", style(file).cyan()))?;
                printed_file_list = true;
            }
        } else {
            out.writeln(format!("  {} {}", style(file).cyan(), style("(not parsed)").dim()))?;
            printed_file_list = true;
        }
    }
    if printed_file_list {
        out.newline()?;
    }

    let tmp_after_dir = tempfile::tempdir()
        .map_err(|e| Error::internal(format!("Failed to create temp dir: {}", e)))?;

    // Build "after" snapshot: from disk when after_rev is None (HEAD), from git blobs otherwise
    if let Some(sp) = &spinner {
        sp.set_message("Loading current snapshot files...");
    }
    let after_files: Vec<String> = if let Some(ref rev) = after_rev {
        let mut files = Vec::new();
        for rel in &scoped_files {
            if Language::from_path(rel).is_none() {
                continue;
            }
            if let Some(content) = read_file_at_rev(repo_path, rev, rel)? {
                let dest = tmp_after_dir.path().join(rel);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        Error::internal(format!("Failed to create temp dir structure: {}", e))
                    })?;
                }
                std::fs::write(&dest, &content).map_err(|e| {
                    Error::internal(format!("Failed to write temp file: {}", e))
                })?;
                files.push(dest.to_string_lossy().to_string());
            }
        }
        files
    } else {
        scoped_files
            .iter()
            .filter(|f| {
                let abs = Path::new(repo_path).join(f);
                abs.exists() && Language::from_path(f).is_some()
            })
            .map(|f| Path::new(repo_path).join(f).to_string_lossy().to_string())
            .collect()
    };

    if let Some(sp) = &spinner {
        sp.set_message("Building current snapshot graph...");
    }
    let after_graph = build_graph_for_files(&after_files).await?;

    // Build "before" graph from git blobs written to a temp directory
    let tmp_dir = tempfile::tempdir()
        .map_err(|e| Error::internal(format!("Failed to create temp dir: {}", e)))?;
    let mut before_files: Vec<String> = Vec::new();
    if let Some(sp) = &spinner {
        sp.set_message("Loading previous snapshot files...");
    }
    for rel_path in &scoped_files {
        if Language::from_path(rel_path).is_none() {
            continue;
        }
        match read_file_at_rev(repo_path, &before_rev, rel_path)? {
            None => {} // file was added (didn't exist before)
            Some(content) => {
                let dest = tmp_dir.path().join(rel_path);
                if let Some(parent) = dest.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        Error::internal(format!("Failed to create temp dir structure: {}", e))
                    })?;
                }
                std::fs::write(&dest, &content).map_err(|e| {
                    Error::internal(format!("Failed to write temp file: {}", e))
                })?;
                before_files.push(dest.to_string_lossy().to_string());
            }
        }
    }

    if let Some(sp) = &spinner {
        sp.set_message("Building previous snapshot graph...");
    }
    let before_graph = build_graph_for_files(&before_files).await?;

    if let Some(sp) = &spinner {
        sp.set_message(format!("Computing {} delta...", type_scope));
    }

    // Canonicalize roots to resolve symlinks (e.g., macOS /var -> /private/var)
    let canon_repo = std::fs::canonicalize(repo_path)
        .unwrap_or_else(|_| std::path::PathBuf::from(repo_path));
    let canon_tmp = tmp_dir
        .path()
        .canonicalize()
        .unwrap_or_else(|_| tmp_dir.path().to_path_buf());
    let canon_tmp_after = tmp_after_dir
        .path()
        .canonicalize()
        .unwrap_or_else(|_| tmp_after_dir.path().to_path_buf());

    let canon_repo_str = canon_repo.to_str().unwrap_or(repo_path);
    let canon_tmp_str = canon_tmp.to_str().unwrap_or("");
    // For "after": use the git-blob temp dir root when reading from a rev, else the repo root
    let canon_after_root = if after_rev.is_some() {
        canon_tmp_after.to_str().unwrap_or("").to_string()
    } else {
        canon_repo_str.to_string()
    };

    let changed_abs: HashSet<String> = after_files
        .iter()
        .filter_map(|f| std::fs::canonicalize(f).ok())
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let changed_tmp: HashSet<String> = before_files
        .iter()
        .filter_map(|f| std::fs::canonicalize(f).ok())
        .map(|p| p.to_string_lossy().to_string())
        .collect();

    let after_by_key = index_graph_by_norm_key(&after_graph, &canon_after_root, &changed_abs);
    let before_by_key = index_graph_by_norm_key(&before_graph, canon_tmp_str, &changed_tmp);

    let after_keys: HashSet<String> = after_by_key.keys().cloned().collect();
    let before_keys: HashSet<String> = before_by_key.keys().cloned().collect();

    let added: Vec<&Node> = after_keys
        .difference(&before_keys)
        .filter_map(|k| after_by_key.get(k.as_str()).copied())
        .collect();
    let removed: Vec<&Node> = before_keys
        .difference(&after_keys)
        .filter_map(|k| before_by_key.get(k.as_str()).copied())
        .collect();
    let mut modified: Vec<(&Node, &Node)> = Vec::new();
    for k in after_keys.intersection(&before_keys) {
        if let (Some(a), Some(b)) = (after_by_key.get(k.as_str()), before_by_key.get(k.as_str())) {
            if a.node_data.body != b.node_data.body {
                modified.push((a, b));
            }
        }
    }

    // Apply --types filter
    let type_filter: Vec<String> = types.iter().map(|t| t.to_lowercase()).collect();
    let filter_node = |n: &&Node| -> bool {
        type_filter.is_empty()
            || type_filter.contains(&n.node_type.to_string().to_lowercase())
    };
    let added: Vec<&Node> = added.into_iter().filter(filter_node).collect();
    let removed: Vec<&Node> = removed.into_iter().filter(filter_node).collect();
    let modified: Vec<(&Node, &Node)> = modified
        .into_iter()
        .filter(|(a, _)| filter_node(&a))
        .collect();

    if added.is_empty() && removed.is_empty() && modified.is_empty() {
        if let Some(sp) = &spinner {
            sp.finish_with_message("No graph-level changes found");
        }
        out.writeln(format!(
            "{}",
            style("No graph-level changes detected in the scoped files").dim()
        ))?;
        return Ok(());
    }

    if let Some(sp) = &spinner {
        sp.set_message("Rendering graph change summary...");
    }
    print_delta(out, &added, &removed, &modified, canon_repo_str, &canon_after_root, canon_tmp_str)?;
    if let Some(sp) = &spinner {
        sp.finish_with_message("Graph change summary ready");
    }

    Ok(())
}

async fn build_graph_for_files(files: &[String]) -> Result<ArrayGraph> {
    if files.is_empty() {
        return Ok(ArrayGraph::default());
    }

    let mut files_by_lang: Vec<(Language, Vec<String>)> = Vec::new();
    for file_path in files {
        if let Some(lang) = Language::from_path(file_path) {
            if let Some((_, list)) = files_by_lang.iter_mut().find(|(l, _)| *l == lang) {
                list.push(file_path.clone());
            } else {
                files_by_lang.push((lang, vec![file_path.clone()]));
            }
        }
    }

    let mut repos_vec: Vec<Repo> = Vec::new();
    for (language, file_list) in &files_by_lang {
        let lang = Lang::from_language(language.clone());
        if let Some(root) = common_ancestor(file_list) {
            let file_refs: Vec<&str> = file_list.iter().map(|s| s.as_str()).collect();
            repos_vec.push(Repo::from_files(
                &file_refs,
                root,
                lang,
                false,
                false,
                false,
            )?);
        } else {
            for file_path in file_list {
                let file_lang = Lang::from_language(language.clone());
                repos_vec.push(Repo::from_single_file(file_path, file_lang, false, false, false)?);
            }
        }
    }

    if repos_vec.is_empty() {
        return Ok(ArrayGraph::default());
    }

    let repos = Repos(repos_vec);
    repos.build_graphs_array().await
}

const MAX_SIG: usize = 100;

fn signature_line(body: &str) -> Option<String> {
    body.lines()
        .find(|l| !l.trim().is_empty())
        .map(|l| {
            let s = l.trim_end();
            if s.chars().count() > MAX_SIG {
                format!("{}…", s.chars().take(MAX_SIG).collect::<String>())
            } else {
                s.to_string()
            }
        })
}

fn norm_key(node: &Node, root: &str) -> String {
    let file = &node.node_data.file;
    let rel_file = if root.is_empty() {
        file.as_str()
    } else {
        file.strip_prefix(root)
            .map(|s| s.trim_start_matches('/'))
            .unwrap_or(file.as_str())
    };
    format!("{}-{}-{}", node.node_type, node.node_data.name, rel_file)
}

fn index_graph_by_norm_key<'a>(
    graph: &'a ArrayGraph,
    root: &str,
    allowed_files: &HashSet<String>,
) -> HashMap<String, &'a Node> {
    let mut map = HashMap::new();
    for node in &graph.nodes {
        if matches!(
            node.node_type,
            NodeType::Repository
                | NodeType::File
                | NodeType::Directory
                | NodeType::Import
                | NodeType::Language
                | NodeType::Package
        ) {
            continue;
        }
        if !allowed_files.is_empty() {
            let canon = std::fs::canonicalize(&node.node_data.file)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            if !allowed_files.contains(&canon) {
                continue;
            }
        }
        let key = norm_key(node, root);
        map.entry(key).or_insert(node);
    }
    map
}

fn rel_path(file: &str, root: &str) -> String {
    Path::new(file)
        .strip_prefix(root)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| {
            Path::new(file)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(file)
                .to_string()
        })
}

fn print_delta(
    out: &mut Output,
    added: &[&Node],
    removed: &[&Node],
    modified: &[(&Node, &Node)],
    repo_root: &str,
    after_root: &str,
    before_root: &str,
) -> Result<()> {
    // Collect per-file node lists keyed by repo-relative path
    let mut file_added: HashMap<String, Vec<&Node>> = HashMap::new();
    let mut file_removed: HashMap<String, Vec<&Node>> = HashMap::new();
    let mut file_modified: HashMap<String, Vec<(&Node, &Node)>> = HashMap::new();

    for node in added {
        // Added nodes may live in tmp_after_dir (--range) or repo dir (HEAD-based)
        let rp = rel_path(&node.node_data.file, after_root);
        file_added.entry(rp).or_default().push(node);
    }
    for node in removed {
        // Removed nodes always live in tmp (before) dir
        let rp = rel_path(&node.node_data.file, before_root);
        file_removed.entry(rp).or_default().push(node);
    }
    for (after_node, before_node) in modified {
        let rp = rel_path(&after_node.node_data.file, after_root);
        file_modified.entry(rp).or_default().push((after_node, before_node));
    }
    let _ = repo_root; // kept for future use

    // Collect all unique files and determine file-level status
    let mut all_files: Vec<String> = file_added
        .keys()
        .chain(file_removed.keys())
        .chain(file_modified.keys())
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    all_files.sort();

    let total_file_count = all_files.len();

    for file in &all_files {
        let has_added = file_added.contains_key(file);
        let has_removed = file_removed.contains_key(file);
        let has_modified = file_modified.contains_key(file);

        // File-level status: Added if only in added, Removed if only in removed, else Modified
        let file_status = if has_modified || (has_added && has_removed) {
            style("[Modified]".to_string()).bold().yellow()
        } else if has_added {
            style("[Added]".to_string()).bold().green()
        } else {
            style("[Removed]".to_string()).bold().red()
        };

        let node_count = file_added.get(file).map(|v| v.len()).unwrap_or(0)
            + file_removed.get(file).map(|v| v.len()).unwrap_or(0)
            + file_modified.get(file).map(|v| v.len()).unwrap_or(0);

        out.writeln(format!(
            "{}  {}  {}",
            style(file).bold(),
            file_status,
            style(format!("({} node{})", node_count, if node_count == 1 { "" } else { "s" })).dim()
        ))?;

        // Modified nodes (show with ~)
        if let Some(nodes) = file_modified.get(file) {
            let mut sorted = nodes.clone();
            sorted.sort_by_key(|(a, _)| a.node_data.start);
            for (after_node, before_node) in sorted {
                let line_range = style(format!("L{}-L{}", after_node.node_data.start, after_node.node_data.end)).dim();
                out.writeln(format!(
                    "  {} {} {}  {}",
                    style("~").yellow().bold(),
                    style(after_node.node_type.to_string()).yellow(),
                    style(&after_node.node_data.name).bold(),
                    line_range
                ))?;
                let before_sig = signature_line(&before_node.node_data.body);
                let after_sig = signature_line(&after_node.node_data.body);
                match (before_sig, after_sig) {
                    (Some(b), Some(a)) if b != a => {
                        out.writeln(format!("    {} {}", style("-").red(), style(&b).red().bright()))?;
                        out.writeln(format!("    {} {}", style("+").green(), style(&a).green().bright()))?;
                    }
                    _ => {}
                }
            }
        }

        // Added nodes (show with +)
        if let Some(nodes) = file_added.get(file) {
            let mut sorted = nodes.clone();
            sorted.sort_by_key(|n| n.node_data.start);
            for node in sorted {
                let location = style(format!("L{}", node.node_data.start)).dim();
                out.writeln(format!(
                    "  {} {} {}  {}",
                    style("+").green().bold(),
                    style(node.node_type.to_string()).green(),
                    style(&node.node_data.name).bold(),
                    location
                ))?;
                if let Some(sig) = signature_line(&node.node_data.body) {
                    out.writeln(format!("    {}", style(sig).dim()))?;
                }
            }
        }

        // Removed nodes (show with -)
        if let Some(nodes) = file_removed.get(file) {
            let mut sorted = nodes.clone();
            sorted.sort_by_key(|n| n.node_data.start);
            for node in sorted {
                let location = style(format!("L{}", node.node_data.start)).dim();
                out.writeln(format!(
                    "  {} {} {}  {}",
                    style("-").red().bold(),
                    style(node.node_type.to_string()).red(),
                    style(&node.node_data.name).bold(),
                    location
                ))?;
                if let Some(sig) = signature_line(&node.node_data.body) {
                    out.writeln(format!("    {}", style(sig).dim()))?;
                }
            }
        }

        out.newline()?;
    }

    out.writeln(format!(
        "{}  {}  {}  {}",
        style(format!("{} file{}", total_file_count, if total_file_count == 1 { "" } else { "s" })).bold(),
        style(format!("{} added", added.len())).green(),
        style(format!("{} removed", removed.len())).red(),
        style(format!("{} modified", modified.len())).yellow()
    ))?;

    Ok(())
}
