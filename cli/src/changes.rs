use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::git::{
    filter_paths_by_scope, get_changed_files, get_repo_root, get_staged_changes,
    get_working_tree_changes, list_commits_for_paths, read_file_at_rev,
};
use ast::lang::graphs::{ArrayGraph, Node, NodeType, Edge, EdgeType};
use console::style;
use lsp::Language;
use serde::Serialize;
use shared::{Error, Result};

use super::args::{ChangesArgs, ChangesCommand, DiffArgs};
use super::output::{write_json_success, JsonWarning, Output, OutputMode};
use super::progress::CliSpinner;
use super::utils::{build_graph_for_files, parse_node_types};

#[derive(Serialize)]
struct CommitSummary {
    hash: String,
    short_hash: String,
    message: String,
    author: String,
    date: String,
}

#[derive(Serialize)]
struct ChangesListData {
    repo_path: String,
    scope: Vec<String>,
    commits: Vec<CommitSummary>,
}

#[derive(Serialize)]
struct DeltaSummary {
    files_changed: usize,
    nodes_added: usize,
    nodes_removed: usize,
    nodes_modified: usize,
    edges_added: usize,
    edges_removed: usize,
}

#[derive(Serialize)]
struct ChangedNodeSummary {
    node_type: String,
    name: String,
    file: String,
    start_line: usize,
    end_line: usize,
    signature: Option<String>,
}

#[derive(Serialize)]
struct ModifiedNodeSummary {
    before: ChangedNodeSummary,
    after: ChangedNodeSummary,
}

#[derive(Serialize)]
struct EdgeSummary {
    edge_type: String,
    source_name: String,
    source_file: String,
    target_name: String,
    target_file: String,
}

#[derive(Serialize)]
struct ChangesDiffData {
    repo_path: String,
    mode: String,
    scope: Vec<String>,
    files: Vec<String>,
    summary: DeltaSummary,
    added_nodes: Vec<ChangedNodeSummary>,
    removed_nodes: Vec<ChangedNodeSummary>,
    modified_nodes: Vec<ModifiedNodeSummary>,
    added_edges: Vec<EdgeSummary>,
    removed_edges: Vec<EdgeSummary>,
}

pub async fn run(
    args: &ChangesArgs,
    out: &mut Output,
    show_progress: bool,
    output_mode: OutputMode,
) -> Result<()> {
    let cwd = std::env::current_dir()
        .map_err(|e| Error::internal(format!("Failed to get current directory: {}", e)))?;
    let cwd_str = cwd.to_string_lossy().to_string();
    let repo_str = get_repo_root(&cwd_str).unwrap_or(cwd_str);

    match &args.command {
        ChangesCommand::List(list_args) => {
            run_list_commits(
                &repo_str,
                &list_args.paths,
                list_args.max,
                out,
                show_progress,
                output_mode,
            )
            .await
        }
        ChangesCommand::Diff(diff_args) => {
            run_diff(
                &repo_str,
                &diff_args.paths,
                &diff_args.types,
                diff_args,
                out,
                show_progress,
                output_mode,
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
    output_mode: OutputMode,
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

    if output_mode.is_json() {
        let data = ChangesListData {
            repo_path: repo_path.to_string(),
            scope: paths.to_vec(),
            commits: commits
                .into_iter()
                .map(|commit| CommitSummary {
                    short_hash: commit.hash[..7.min(commit.hash.len())].to_string(),
                    hash: commit.hash,
                    message: commit.message,
                    author: commit.author,
                    date: commit.date,
                })
                .collect(),
        };
        write_json_success(out, "changes", data, Vec::new())?;
        return Ok(());
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
    output_mode: OutputMode,
) -> Result<()> {

    let validated_types = parse_node_types(types)?;

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
            if let Some(sp) = &spinner {
                sp.finish_and_clear();
            }
            return Err(Error::validation(
                "range must be in format <a>..<b> (e.g. HEAD~3..HEAD)",
            ));
        }
        let files = get_changed_files(repo_path, parts[0], parts[1])?;
        (files, parts[0].to_string(), Some(parts[1].to_string()))
    } else {
        (get_working_tree_changes(repo_path)?, "HEAD".to_string(), None)
    };

    let scoped_files = filter_paths_by_scope(changed_files, paths);

    if scoped_files.is_empty() {
        let mut warnings = Vec::new();
        if !paths.is_empty() {
            for p in paths {
                let abs = Path::new(repo_path).join(p);
                if !abs.exists() {
                    if output_mode.is_json() {
                        warnings.push(JsonWarning::new(
                            "missing_path",
                            format!("'{}' does not exist in this repository", p),
                        ));
                    } else {
                        out.writeln(format!(
                            "{}",
                            style(format!("warning: '{}' does not exist in this repository", p))
                                .yellow()
                        ))?;
                    }
                }
            }
        }
        if output_mode.is_json() {
            let data = ChangesDiffData {
                repo_path: repo_path.to_string(),
                mode: mode_description,
                scope: paths.to_vec(),
                files: Vec::new(),
                summary: DeltaSummary {
                    files_changed: 0,
                    nodes_added: 0,
                    nodes_removed: 0,
                    nodes_modified: 0,
                    edges_added: 0,
                    edges_removed: 0,
                },
                added_nodes: Vec::new(),
                removed_nodes: Vec::new(),
                modified_nodes: Vec::new(),
                added_edges: Vec::new(),
                removed_edges: Vec::new(),
            };
            write_json_success(out, "changes", data, warnings)?;
            return Ok(());
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

    if !output_mode.is_json() {
        out.writeln(format!(
            "{} {} file(s) changed in {} (scope: {})",
            style("Found").bold().cyan(),
            style(scoped_files.len()).bold().green(),
            style(&mode_description).yellow(),
            style(&scope_label).cyan()
        ))?;
        out.newline()?;
    }

    let parseable_count = scoped_files
        .iter()
        .filter(|file| Language::from_path(file).is_some())
        .count();
    let mut printed_file_list = false;

    for file in &scoped_files {
        let parseable = Language::from_path(file).is_some();
        if parseable {
            if parseable_count > 5 && !output_mode.is_json() {
                out.writeln(format!("  {}", style(file).cyan()))?;
                printed_file_list = true;
            }
        } else {
            if !output_mode.is_json() {
                out.writeln(format!("  {} {}", style(file).cyan(), style("(not parsed)").dim()))?;
                printed_file_list = true;
            }
        }
    }
    if printed_file_list && !output_mode.is_json() {
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

    let after_edge_by_key = index_edges_by_key(&after_graph, &canon_after_root, &changed_abs);
    let before_edge_by_key = index_edges_by_key(&before_graph, canon_tmp_str, &changed_tmp);

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

    // Edge diff: Calls + Handler edges whose source node was not itself added/removed
    let after_edge_keys: HashSet<String> = after_edge_by_key.keys().cloned().collect();
    let before_edge_keys: HashSet<String> = before_edge_by_key.keys().cloned().collect();
    let added_node_keys: HashSet<String> = added
        .iter()
        .map(|n| norm_key(n, &canon_after_root))
        .collect();
    let removed_node_keys: HashSet<String> = removed
        .iter()
        .map(|n| norm_key(n, canon_tmp_str))
        .collect();
    let added_edges: Vec<&Edge> = after_edge_keys
        .difference(&before_edge_keys)
        .filter_map(|k| after_edge_by_key.get(k.as_str()).copied())
        .filter(|e| {
            let src_key = norm_key_from_ref(&e.source, &canon_after_root);
            !added_node_keys.contains(&src_key)
        })
        .collect();
    let removed_edges: Vec<&Edge> = before_edge_keys
        .difference(&after_edge_keys)
        .filter_map(|k| before_edge_by_key.get(k.as_str()).copied())
        .filter(|e| {
            let src_key = norm_key_from_ref(&e.source, canon_tmp_str);
            !removed_node_keys.contains(&src_key)
        })
        .collect();

    let filter_node = |n: &&Node| -> bool {
        validated_types.is_empty() || validated_types.contains(&n.node_type)
    };
    let added: Vec<&Node> = added.into_iter().filter(filter_node).collect();
    let removed: Vec<&Node> = removed.into_iter().filter(filter_node).collect();
    let modified: Vec<(&Node, &Node)> = modified
        .into_iter()
        .filter(|(a, _)| filter_node(&a))
        .collect();

    if added.is_empty() && removed.is_empty() && modified.is_empty() && added_edges.is_empty() && removed_edges.is_empty() {
        if let Some(sp) = &spinner {
            sp.finish_with_message("No graph-level changes found");
        }
        if output_mode.is_json() {
            let data = ChangesDiffData {
                repo_path: repo_path.to_string(),
                mode: mode_description,
                scope: paths.to_vec(),
                files: scoped_files,
                summary: DeltaSummary {
                    files_changed: 0,
                    nodes_added: 0,
                    nodes_removed: 0,
                    nodes_modified: 0,
                    edges_added: 0,
                    edges_removed: 0,
                },
                added_nodes: Vec::new(),
                removed_nodes: Vec::new(),
                modified_nodes: Vec::new(),
                added_edges: Vec::new(),
                removed_edges: Vec::new(),
            };
            write_json_success(out, "changes", data, Vec::new())?;
            return Ok(());
        }
        out.writeln(format!(
            "{}",
            style("No graph-level changes detected in the scoped files").dim()
        ))?;
        return Ok(());
    }

    if output_mode.is_json() {
        let data = ChangesDiffData {
            repo_path: repo_path.to_string(),
            mode: mode_description,
            scope: paths.to_vec(),
            files: scoped_files,
            summary: DeltaSummary {
                files_changed: total_changed_file_count(&added, &removed, &modified, &added_edges, &removed_edges, &canon_after_root, canon_tmp_str),
                nodes_added: added.len(),
                nodes_removed: removed.len(),
                nodes_modified: modified.len(),
                edges_added: added_edges.len(),
                edges_removed: removed_edges.len(),
            },
            added_nodes: added.iter().map(|n| json_node_summary(n)).collect(),
            removed_nodes: removed.iter().map(|n| json_node_summary(n)).collect(),
            modified_nodes: modified
                .iter()
                .map(|(after, before)| ModifiedNodeSummary {
                    before: json_node_summary(before),
                    after: json_node_summary(after),
                })
                .collect(),
            added_edges: added_edges.iter().map(|e| json_edge_summary(e)).collect(),
            removed_edges: removed_edges.iter().map(|e| json_edge_summary(e)).collect(),
        };
        write_json_success(out, "changes", data, Vec::new())?;
        return Ok(());
    }

    if let Some(sp) = &spinner {
        sp.set_message("Rendering graph change summary...");
    }
    print_delta(out, &added, &removed, &modified, &added_edges, &removed_edges, canon_repo_str, &canon_after_root, canon_tmp_str)?;
    if let Some(sp) = &spinner {
        sp.finish_with_message("Graph change summary ready");
    }

    Ok(())
}

fn json_node_summary(node: &Node) -> ChangedNodeSummary {
    ChangedNodeSummary {
        node_type: node.node_type.to_string(),
        name: node.node_data.name.clone(),
        file: node.node_data.file.clone(),
        start_line: node.node_data.start + 1,
        end_line: node.node_data.end + 1,
        signature: node_signature(node),
    }
}

fn json_edge_summary(edge: &Edge) -> EdgeSummary {
    EdgeSummary {
        edge_type: format!("{:?}", edge.edge).to_uppercase(),
        source_name: edge.source.node_data.name.clone(),
        source_file: edge.source.node_data.file.clone(),
        target_name: edge.target.node_data.name.clone(),
        target_file: edge.target.node_data.file.clone(),
    }
}

fn total_changed_file_count(
    added: &[&Node],
    removed: &[&Node],
    modified: &[(&Node, &Node)],
    added_edges: &[&Edge],
    removed_edges: &[&Edge],
    after_root: &str,
    before_root: &str,
) -> usize {
    added
        .iter()
        .map(|n| rel_path(&n.node_data.file, after_root))
        .chain(removed.iter().map(|n| rel_path(&n.node_data.file, before_root)))
        .chain(modified.iter().map(|(n, _)| rel_path(&n.node_data.file, after_root)))
        .chain(added_edges.iter().map(|e| rel_path(&e.source.node_data.file, after_root)))
        .chain(removed_edges.iter().map(|e| rel_path(&e.source.node_data.file, before_root)))
        .collect::<HashSet<_>>()
        .len()
}

const MAX_SIG: usize = 100;
fn node_signature(node: &Node) -> Option<String> {
    let raw = if let Some(iface) = node.node_data.meta.get("interface") {
        iface.as_str()
    } else {
        node.node_data.body.lines().find(|l| !l.trim().is_empty())?
    };
    let sig: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let sig = sig.trim().to_string();
    if sig.is_empty() {
        return None;
    }
    Some(if sig.chars().count() > MAX_SIG {
        format!("{}…", sig.chars().take(MAX_SIG).collect::<String>())
    } else {
        sig
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

fn norm_key_from_ref(node_ref: &ast::lang::graphs::NodeRef, root: &str) -> String {
    let file = &node_ref.node_data.file;
    let rel_file = if root.is_empty() {
        file.as_str()
    } else {
        file.strip_prefix(root)
            .map(|s| s.trim_start_matches('/'))
            .unwrap_or(file.as_str())
    };
    format!("{}-{}-{}", node_ref.node_type, node_ref.node_data.name, rel_file)
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

fn edge_key(edge: &Edge, src_root: &str, tgt_root: &str) -> String {
    let src_file = &edge.source.node_data.file;
    let tgt_file = &edge.target.node_data.file;
    let rel_src = if src_root.is_empty() {
        src_file.as_str()
    } else {
        src_file
            .strip_prefix(src_root)
            .map(|s| s.trim_start_matches('/'))
            .unwrap_or(src_file.as_str())
    };
    let rel_tgt = if tgt_root.is_empty() {
        tgt_file.as_str()
    } else {
        tgt_file
            .strip_prefix(tgt_root)
            .map(|s| s.trim_start_matches('/'))
            .unwrap_or(tgt_file.as_str())
    };
    format!(
        "{}-{}-{}→{}-{}-{}",
        edge.source.node_type,
        edge.source.node_data.name,
        rel_src,
        edge.target.node_type,
        edge.target.node_data.name,
        rel_tgt,
    )
}

fn index_edges_by_key<'a>(
    graph: &'a ArrayGraph,
    root: &str,
    allowed_files: &HashSet<String>,
) -> HashMap<String, &'a Edge> {
    let mut map = HashMap::new();
    for edge in &graph.edges {
        if !matches!(edge.edge, EdgeType::Calls | EdgeType::Handler) {
            continue;
        }
        // Only include edges where at least one endpoint is in the changed files set
        if !allowed_files.is_empty() {
            let src_canon = std::fs::canonicalize(&edge.source.node_data.file)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            let tgt_canon = std::fs::canonicalize(&edge.target.node_data.file)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            if !allowed_files.contains(&src_canon) && !allowed_files.contains(&tgt_canon) {
                continue;
            }
        }
        let key = edge_key(edge, root, root);
        map.entry(key).or_insert(edge);
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
    added_edges: &[&Edge],
    removed_edges: &[&Edge],
    repo_root: &str,
    after_root: &str,
    before_root: &str,
) -> Result<()> {
    // Collect per-file node lists keyed by repo-relative path
    let mut file_added: HashMap<String, Vec<&Node>> = HashMap::new();
    let mut file_removed: HashMap<String, Vec<&Node>> = HashMap::new();
    let mut file_modified: HashMap<String, Vec<(&Node, &Node)>> = HashMap::new();
    // Edge changes bucketed by the source node's file
    let mut file_added_edges: HashMap<String, Vec<&Edge>> = HashMap::new();
    let mut file_removed_edges: HashMap<String, Vec<&Edge>> = HashMap::new();

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
    for edge in added_edges {
        let rp = rel_path(&edge.source.node_data.file, after_root);
        file_added_edges.entry(rp).or_default().push(edge);
    }
    for edge in removed_edges {
        let rp = rel_path(&edge.source.node_data.file, before_root);
        file_removed_edges.entry(rp).or_default().push(edge);
    }
    let _ = repo_root; // kept for future use

    // Collect all unique files and determine file-level status
    let mut all_files: Vec<String> = file_added
        .keys()
        .chain(file_removed.keys())
        .chain(file_modified.keys())
        .chain(file_added_edges.keys())
        .chain(file_removed_edges.keys())
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
        let edge_add_count = file_added_edges.get(file).map(|v| v.len()).unwrap_or(0);
        let edge_rem_count = file_removed_edges.get(file).map(|v| v.len()).unwrap_or(0);

        // File-level status: Added if only in added, Removed if only in removed, else Modified
        let file_status = if has_modified || (has_added && has_removed) {
            style("[Modified]".to_string()).bold().yellow()
        } else if has_added {
            style("[Added]".to_string()).bold().green()
        } else if has_removed {
            style("[Removed]".to_string()).bold().red()
        } else {
            style("[Edge changes]".to_string()).bold().cyan()
        };

        let node_count = file_added.get(file).map(|v| v.len()).unwrap_or(0)
            + file_removed.get(file).map(|v| v.len()).unwrap_or(0)
            + file_modified.get(file).map(|v| v.len()).unwrap_or(0);

        let mut summary_parts: Vec<String> = Vec::new();
        if node_count > 0 {
            summary_parts.push(format!("{} node{}", node_count, if node_count == 1 { "" } else { "s" }));
        }
        if edge_add_count + edge_rem_count > 0 {
            summary_parts.push(format!("{} edge change{}", edge_add_count + edge_rem_count, if edge_add_count + edge_rem_count == 1 { "" } else { "s" }));
        }
        let summary = summary_parts.join(", ");

        out.writeln(format!(
            "{}  {}  {}",
            style(file).bold(),
            file_status,
            style(format!("({})", summary)).dim()
        ))?;

        // Modified nodes (show with ~)
        if let Some(nodes) = file_modified.get(file) {
            let mut sorted = nodes.clone();
            sorted.sort_by_key(|(a, _)| a.node_data.start);
            for (after_node, before_node) in sorted {
                let line_range = style(format!("L{}-L{}", after_node.node_data.start + 1, after_node.node_data.end + 1)).dim();
                out.writeln(format!(
                    "  {} {} {}  {}",
                    style("~").yellow().bold(),
                    style(after_node.node_type.to_string()).yellow(),
                    style(&after_node.node_data.name).bold(),
                    line_range
                ))?;
                let before_sig = node_signature(&before_node);
                let after_sig = node_signature(&after_node);
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
                let location = style(format!("L{}", node.node_data.start + 1)).dim();
                out.writeln(format!(
                    "  {} {} {}  {}",
                    style("+").green().bold(),
                    style(node.node_type.to_string()).green(),
                    style(&node.node_data.name).bold(),
                    location
                ))?;
                if let Some(sig) = node_signature(&node) {
                    out.writeln(format!("    {}", style(sig).dim()))?;
                }
            }
        }

        // Removed nodes (show with -)
        if let Some(nodes) = file_removed.get(file) {
            let mut sorted = nodes.clone();
            sorted.sort_by_key(|n| n.node_data.start);
            for node in sorted {
                let location = style(format!("L{}", node.node_data.start + 1)).dim();
                out.writeln(format!(
                    "  {} {} {}  {}",
                    style("-").red().bold(),
                    style(node.node_type.to_string()).red(),
                    style(&node.node_data.name).bold(),
                    location
                ))?;
                if let Some(sig) = node_signature(&node) {
                    out.writeln(format!("    {}", style(sig).dim()))?;
                }
            }
        }

        // Added edges (show with ↗)
        if let Some(edges) = file_added_edges.get(file) {
            let mut sorted = edges.clone();
            sorted.sort_by_key(|e| &e.source.node_data.name);
            for edge in sorted {
                let tgt_file = &edge.target.node_data.file;
                let tgt_rel = rel_path(tgt_file, after_root);
                let tgt_display = if tgt_rel != *file {
                    format!("{} [{}]", edge.target.node_data.name, tgt_rel)
                } else {
                    edge.target.node_data.name.clone()
                };
                let edge_label = match edge.edge {
                    EdgeType::Handler => "handler",
                    _ => "calls",
                };
                out.writeln(format!(
                    "  {} {} {} {} {}",
                    style("↗").green().bold(),
                    style("new").green(),
                    style(edge_label).dim(),
                    style(&edge.source.node_data.name).bold(),
                    style(format!("→ {}", tgt_display)).cyan()
                ))?;
            }
        }

        // Removed edges (show with ↘)
        if let Some(edges) = file_removed_edges.get(file) {
            let mut sorted = edges.clone();
            sorted.sort_by_key(|e| &e.source.node_data.name);
            for edge in sorted {
                let tgt_file = &edge.target.node_data.file;
                let tgt_rel = rel_path(tgt_file, before_root);
                let tgt_display = if tgt_rel != *file {
                    format!("{} [{}]", edge.target.node_data.name, tgt_rel)
                } else {
                    edge.target.node_data.name.clone()
                };
                let edge_label = match edge.edge {
                    EdgeType::Handler => "handler",
                    _ => "calls",
                };
                out.writeln(format!(
                    "  {} {} {} {} {}",
                    style("↘").red().bold(),
                    style("dropped").red(),
                    style(edge_label).dim(),
                    style(&edge.source.node_data.name).bold(),
                    style(format!("→ {}", tgt_display)).cyan()
                ))?;
            }
        }

        out.newline()?;
    }

    out.writeln(format!(
        "{}  {}  {}  {}  {}  {}",
        style(format!("{} file{}", total_file_count, if total_file_count == 1 { "" } else { "s" })).bold(),
        style(format!("{} added", added.len())).green(),
        style(format!("{} removed", removed.len())).red(),
        style(format!("{} modified", modified.len())).yellow(),
        style(format!("{} new edge{}", added_edges.len(), if added_edges.len() == 1 { "" } else { "s" })).green(),
        style(format!("{} dropped edge{}", removed_edges.len(), if removed_edges.len() == 1 { "" } else { "s" })).red(),
    ))?;

    Ok(())
}
