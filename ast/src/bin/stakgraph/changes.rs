use std::collections::{HashMap, HashSet};
use std::path::Path;

use ast::gat::{
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
use super::utils::common_ancestor;

pub async fn run(args: &ChangesArgs, out: &mut Output) -> Result<()> {
    let repo_path = std::env::current_dir()
        .map_err(|e| Error::internal(format!("Failed to get current directory: {}", e)))?;

    let repo_str = repo_path.to_string_lossy().to_string();

    match &args.command {
        ChangesCommand::List(list_args) => {
            run_list_commits(&repo_str, &list_args.paths, list_args.max, out).await
        }
        ChangesCommand::Diff(diff_args) => {
            run_diff(&repo_str, &diff_args.paths, diff_args, out).await
        }
    }
}

async fn run_list_commits(
    repo_path: &str,
    paths: &[String],
    max: usize,
    out: &mut Output,
) -> Result<()> {
    let commits = list_commits_for_paths(repo_path, paths, Some(max))?;

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
            style("(timestamp hidden)").dim(),
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
    args: &DiffArgs,
    out: &mut Output,
) -> Result<()> {
    let (changed_files, before_rev) = if args.staged {
        (get_staged_changes(repo_path)?, "HEAD".to_string())
    } else if let Some(last_n) = args.last {
        let old_rev = format!("HEAD~{}", last_n);
        let files = get_changed_files(repo_path, &old_rev, "HEAD")?;
        (files, old_rev)
    } else if let Some(ref since_ref) = args.since {
        let files = get_changed_files(repo_path, since_ref, "HEAD")?;
        (files, since_ref.clone())
    } else if let Some(ref range) = args.range {
        let parts: Vec<&str> = range.split("..").collect();
        if parts.len() != 2 {
            return Err(Error::validation("Range must be in format <a>..<b>"));
        }
        let files = get_changed_files(repo_path, parts[0], parts[1])?;
        (files, parts[0].to_string())
    } else {
        (get_working_tree_changes(repo_path)?, "HEAD".to_string())
    };

    let scoped_files = filter_paths_by_scope(changed_files, paths);

    if scoped_files.is_empty() {
        out.writeln(format!(
            "{}",
            style("No changes found in the specified scope").yellow()
        ))?;
        return Ok(());
    }

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

    for file in &scoped_files {
        out.writeln(format!("  {}", style(file).cyan()))?;
    }
    out.newline()?;

    // Build "after" graph from current files on disk (only parseable files)
    let after_files: Vec<String> = scoped_files
        .iter()
        .filter(|f| {
            let abs = Path::new(repo_path).join(f);
            abs.exists() && Language::from_path(f).is_some()
        })
        .map(|f| Path::new(repo_path).join(f).to_string_lossy().to_string())
        .collect();

    let after_graph = build_graph_for_files(&after_files).await?;

    // Build "before" graph from git blobs written to a temp directory
    let tmp_dir = tempfile::tempdir()
        .map_err(|e| Error::internal(format!("Failed to create temp dir: {}", e)))?;

    let mut before_files: Vec<String> = Vec::new();
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

    let before_graph = build_graph_for_files(&before_files).await?;

    // Compute delta using a normalized key (node_type + name + relative_file)
    // Canonicalize both roots to resolve symlinks (e.g., macOS /var -> /private/var)
    let canon_repo = std::fs::canonicalize(repo_path)
        .unwrap_or_else(|_| std::path::PathBuf::from(repo_path));
    let canon_tmp = tmp_dir
        .path()
        .canonicalize()
        .unwrap_or_else(|_| tmp_dir.path().to_path_buf());
    let after_by_key = index_graph_by_norm_key(&after_graph, canon_repo.to_str().unwrap_or(repo_path));
    let before_by_key = index_graph_by_norm_key(&before_graph, canon_tmp.to_str().unwrap_or(""));

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

    if added.is_empty() && removed.is_empty() && modified.is_empty() {
        out.writeln(format!(
            "{}",
            style("No graph-level changes detected in the scoped files").dim()
        ))?;
        return Ok(());
    }

    print_delta(out, &added, &removed, &modified)?;

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

fn index_graph_by_norm_key<'a>(graph: &'a ArrayGraph, root: &str) -> HashMap<String, &'a Node> {
    let mut map = HashMap::new();
    for node in &graph.nodes {
        if matches!(
            node.node_type,
            NodeType::Repository | NodeType::File | NodeType::Directory | NodeType::Import
        ) {
            continue;
        }
        let key = norm_key(node, root);
        map.entry(key).or_insert(node);
    }
    map
}

fn print_delta(
    out: &mut Output,
    added: &[&Node],
    removed: &[&Node],
    modified: &[(&Node, &Node)],
) -> Result<()> {
    if !added.is_empty() {
        out.writeln(format!(
            "{} ({}):",
            style("Added").bold().green(),
            added.len()
        ))?;
        let mut sorted = added.to_vec();
        sorted.sort_by_key(|n| (&n.node_data.file, n.node_data.start));
        for node in sorted {
            out.writeln(format!(
                "  {} {} {}",
                style("+").green().bold(),
                style(node.node_type.to_string()).green(),
                style(&node.node_data.name).bold()
            ))?;
            let filename = Path::new(&node.node_data.file)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&node.node_data.file);
            out.writeln(format!(
                "    {}",
                style(format!("{}:{}", filename, node.node_data.start + 1)).dim()
            ))?;
        }
        out.newline()?;
    }

    if !removed.is_empty() {
        out.writeln(format!(
            "{} ({}):",
            style("Removed").bold().red(),
            removed.len()
        ))?;
        let mut sorted = removed.to_vec();
        sorted.sort_by_key(|n| (&n.node_data.file, n.node_data.start));
        for node in sorted {
            out.writeln(format!(
                "  {} {} {}",
                style("-").red().bold(),
                style(node.node_type.to_string()).red(),
                style(&node.node_data.name).bold()
            ))?;
            let filename = Path::new(&node.node_data.file)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&node.node_data.file);
            out.writeln(format!(
                "    {}",
                style(format!("{}:{}", filename, node.node_data.start + 1)).dim()
            ))?;
        }
        out.newline()?;
    }

    if !modified.is_empty() {
        out.writeln(format!(
            "{} ({}):",
            style("Modified").bold().yellow(),
            modified.len()
        ))?;
        let mut sorted = modified.to_vec();
        sorted.sort_by_key(|(a, _)| (&a.node_data.file, a.node_data.start));
        for (after_node, _before_node) in sorted {
            out.writeln(format!(
                "  {} {} {}",
                style("~").yellow().bold(),
                style(after_node.node_type.to_string()).yellow(),
                style(&after_node.node_data.name).bold()
            ))?;
            let filename = Path::new(&after_node.node_data.file)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&after_node.node_data.file);
            out.writeln(format!(
                "    {}",
                style(format!("{}:{}", filename, after_node.node_data.start + 1)).dim()
            ))?;
        }
        out.newline()?;
    }

    Ok(())
}
