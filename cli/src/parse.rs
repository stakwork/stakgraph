use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use ast::lang::graphs::{ArrayGraph, Node, NodeType};
use ast::repo::{Repo, Repos};
use ast::Lang;
use console::style;
use lsp::Language;
use serde::Serialize;
use shared::{Error, Result};
use walkdir::WalkDir;

use super::args::CliArgs;
use super::output::{write_json_success, JsonWarning, Output, OutputMode};
use super::progress::{CliSpinner, ProgressTracker};
use super::render::{
    build_call_index, node_code_preview, node_display_name, print_named_node,
    print_single_file_nodes, print_single_file_nodes_filtered, resolve_call_ref,
    CallRef,
};
use super::summarize::run_summarize;
use super::utils::{common_ancestor, parse_node_types, read_text_preview};

#[derive(Serialize)]
struct UnsupportedFileSummary {
    file: String,
    preview: Option<String>,
    reason: String,
}

#[derive(Serialize)]
struct NodeSummary {
    node_type: NodeType,
    name: String,
    start: usize,
    end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    docs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    handler: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    calls: Vec<CallRef>,
}

#[derive(Serialize)]
struct FileNodeEntry {
    file: String,
    nodes: Vec<NodeSummary>,
}

#[derive(Serialize)]
struct ParseJsonData {
    mode: String,
    files: Vec<FileNodeEntry>,
    unsupported_files: Vec<UnsupportedFileSummary>,
    stats: Option<BTreeMap<String, usize>>,
}

fn expand_dirs(inputs: &[String]) -> Result<(Vec<String>, HashSet<String>)> {
    let mut expanded: Vec<String> = Vec::new();
    let mut dir_files: HashSet<String> = HashSet::new();
    let mut seen: HashSet<String> = HashSet::new();

    for input in inputs {
        let path = Path::new(input);
        if path.is_dir() {
            for entry in WalkDir::new(path) {
                let entry = entry?;
                if entry.file_type().is_file() {
                    let p = entry.path().to_string_lossy().to_string();
                    if Language::from_path(&p).is_some() && seen.insert(p.clone()) {
                        expanded.push(p.clone());
                        dir_files.insert(p);
                    }
                }
            }
        } else {
            if seen.insert(input.clone()) {
                expanded.push(input.clone());
            }
        }
    }

    Ok((expanded, dir_files))
}

fn parse_goal_phrase(node_types: &[NodeType], stats: bool) -> String {
    let focus = if node_types.is_empty() {
        "all node types".to_string()
    } else {
        let names: Vec<String> = node_types.iter().map(ToString::to_string).collect();
        format!("{}", names.join(", "))
    };

    if stats {
        format!("{} and node counts", focus)
    } else {
        focus
    }
}

fn collect_nodes_for_file(
    graph: &ArrayGraph,
    file_path: &str,
    node_name: Option<&str>,
    type_filter: Option<&[NodeType]>,
) -> Vec<NodeSummary> {
    let canonical = std::fs::canonicalize(file_path)
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| file_path.to_string());

    let in_file = |node: &&Node| -> bool {
        let nf = std::fs::canonicalize(&node.node_data.file)
            .map(|p| p.to_string_lossy().to_string());
        if let Ok(ref nf) = nf {
            if *nf == canonical {
                return true;
            }
        }
        canonical.ends_with(&node.node_data.file)
    };

    let endpoint_lines: std::collections::HashSet<usize> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Endpoint) && in_file(n))
        .map(|n| n.node_data.start)
        .collect();

    let edges_by_source = build_call_index(&graph.edges);

    let mut nodes: Vec<_> = graph
        .nodes
        .iter()
        .filter(|node| {
            if matches!(node.node_type, NodeType::File | NodeType::Directory) {
                return false;
            }
            if matches!(node.node_type, NodeType::Request)
                && endpoint_lines.contains(&node.node_data.start)
            {
                return false;
            }
            if let Some(types) = type_filter {
                if !types.contains(&node.node_type) {
                    return false;
                }
            }
            if let Some(name) = node_name {
                if node.node_data.name != name {
                    return false;
                }
            }
            in_file(node)
        })
        .collect();

    nodes.sort_by_key(|n| n.node_data.start);

    nodes
        .into_iter()
        .map(|node| {
            let nd = &node.node_data;
            let calls = if matches!(node.node_type, NodeType::Function) {
                let source_key = ast::utils::create_node_key(node).to_lowercase();
                edges_by_source
                    .get(&source_key)
                    .map(|edges| {
                        edges
                            .iter()
                            .map(|edge| resolve_call_ref(edge, &nd.file, graph))
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            NodeSummary {
                node_type: node.node_type.clone(),
                name: node_display_name(node),
                start: nd.start + 1,
                end: nd.end + 1,
                docs: nd.docs.clone(),
                handler: if matches!(node.node_type, NodeType::Endpoint) {
                    nd.meta.get("handler").cloned()
                } else {
                    None
                },
                code: node_code_preview(node),
                calls,
            }
        })
        .collect()
}

pub async fn run(cli: &CliArgs, out: &mut Output, output_mode: OutputMode) -> Result<()> {
    let first = cli.files.first().map(|s| s.as_str()).unwrap_or(".");
    let is_dir_input = Path::new(first).is_dir();
    let has_filters = !cli.r#type.is_empty() || cli.name.is_some() || cli.stats;

    if (cli.max_tokens.is_some() || (is_dir_input && !has_filters)) && !output_mode.is_json() {
        let max_tokens = cli.max_tokens.unwrap_or(8000);
        return run_summarize(first, max_tokens, cli.depth, out, cli.verbose || cli.perf).await;
    }

    let (files, dir_files) = expand_dirs(&cli.files)?;
    let allow_unverified_calls = cli.allow;
    let skip_calls = cli.skip_calls;
    let no_nested = cli.no_nested;
    let node_types = parse_node_types(&cli.r#type)?;

    let mut files_by_lang: Vec<(Language, Vec<String>)> = Vec::new();
    let mut files_to_print: Vec<String> = Vec::new();
    let mut unsupported_files: Vec<UnsupportedFileSummary> = Vec::new();

    for file_path in &files {
        if !dir_files.contains(file_path) && !Path::new(file_path).exists() {
            return Err(Error::validation(format!("file does not exist: {}", file_path)));
        }

        let canonical_path = std::fs::canonicalize(file_path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| file_path.clone());
        let language = Language::from_path(file_path);
        match language {
            Some(lang) => {
                if let Some((_, file_list)) = files_by_lang.iter_mut().find(|(l, _)| *l == lang) {
                    file_list.push(canonical_path.clone());
                } else {
                    files_by_lang.push((lang, vec![canonical_path.clone()]));
                }
                files_to_print.push(canonical_path);
            }
            None => {
                if !dir_files.contains(file_path) {
                    let file_label = style("File:").bold().cyan();
                    let rel = crate::utils::rel_path_from_cwd(file_path);
                    let preview = read_text_preview(file_path);
                    if output_mode.is_json() {
                        unsupported_files.push(UnsupportedFileSummary {
                            file: rel,
                            preview,
                            reason: "unsupported_or_unparseable".to_string(),
                        });
                    } else {
                        let msg = match preview {
                            Some(preview) => format!(
                                "{}  {}\n{}\n",
                                file_label,
                                style(&rel).cyan(),
                                preview
                            ),
                            None => format!(
                                "{}  {}\n[binary or unprintable content skipped]\n",
                                file_label,
                                style(&rel).cyan()
                            ),
                        };
                        out.writeln(msg)?;
                    }
                }
            }
        }
    }

    if files_by_lang.is_empty() {
        if output_mode.is_json() {
            let warnings = if unsupported_files.is_empty() {
                vec![JsonWarning::new("no_parseable_files", "No parseable files found")]
            } else {
                Vec::new()
            };
            let data = ParseJsonData {
                mode: "parse".to_string(),
                files: Vec::new(),
                unsupported_files,
                stats: None,
            };
            write_json_success(out, "parse", data, warnings)?;
        }
        return Ok(());
    }

    let goal_phrase = parse_goal_phrase(&node_types, cli.stats);

    let spinner = if cli.verbose || cli.perf {
        Some(CliSpinner::new(&format!("Preparing {} summary...", goal_phrase)))
    } else {
        None
    };

    let (progress_tracker, status_tx) = ProgressTracker::new(cli.verbose || cli.perf);
    let progress_handle = tokio::spawn(progress_tracker.run());

    let mut repos_vec: Vec<Repo> = Vec::new();
    for (language, file_list) in files_by_lang.iter() {
        let lang = Lang::from_language(language.clone());

        if let Some(root) = common_ancestor(file_list) {
            let file_refs: Vec<&str> = file_list.iter().map(|s| s.as_str()).collect();
            let repo = Repo::from_files(
                &file_refs,
                root,
                lang,
                allow_unverified_calls,
                skip_calls,
                no_nested,
            )?;
            repos_vec.push(repo);
        } else {
            for file_path in file_list {
                let file_lang = Lang::from_language(language.clone());
                let repo = Repo::from_single_file(
                    file_path,
                    file_lang,
                    allow_unverified_calls,
                    skip_calls,
                    no_nested,
                )?;
                repos_vec.push(repo);
            }
        }
    }

    let mut repos = Repos(repos_vec);
    repos.set_status_tx(status_tx).await;

    if let Some(sp) = &spinner {
        sp.set_message(format!("Building graph for {}...", goal_phrase));
    }

    let graph = repos.build_graphs_array().await?;

    drop(repos);
    let _ = progress_handle.await;

    if let Some(sp) = &spinner {
        sp.set_message(format!("Rendering {} output...", goal_phrase));
    }

    let mut file_entries: Vec<FileNodeEntry> = Vec::new();

    for file_path in &files_to_print {
        if output_mode.is_json() {
            let type_filter = if node_types.is_empty() { None } else { Some(node_types.as_slice()) };
            let name_filter = cli.name.as_deref();
            let nodes = collect_nodes_for_file(&graph, file_path, name_filter, type_filter);
            file_entries.push(FileNodeEntry {
                file: crate::utils::rel_path_from_cwd(file_path),
                nodes,
            });
        } else if let Some(node_name) = &cli.name {
            let type_filter = if node_types.is_empty() { None } else { Some(node_types.as_slice()) };
            print_named_node(out, &graph, file_path, node_name, type_filter)?;
        } else if node_types.is_empty() {
            print_single_file_nodes(out, &graph, file_path)?;
        } else {
            print_single_file_nodes_filtered(out, &graph, file_path, &node_types)?;
        }
    }

    let mut stats_out: Option<BTreeMap<String, usize>> = None;
    if cli.stats {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for node in &graph.nodes {
            if matches!(node.node_type, NodeType::File | NodeType::Directory) {
                continue;
            }
            let file_path = std::fs::canonicalize(&node.node_data.file)
                .unwrap_or_else(|_| std::path::PathBuf::from(&node.node_data.file))
                .to_string_lossy()
                .to_string();
            if !files_to_print.iter().any(|f| *f == file_path) {
                continue;
            }
            *counts.entry(node.node_type.to_string()).or_insert(0) += 1;
        }
        if output_mode.is_json() {
            stats_out = Some(counts);
        } else {
            out.writeln(style("\n--- Node type counts ---").bold().to_string())?;
            for (type_name, count) in &counts {
                out.writeln(format!("  {:<20} {}", type_name, count))?;
            }
        }
    }

    if let Some(sp) = &spinner {
        sp.finish_with_message("Node summary ready");
    }

    if output_mode.is_json() {
        let warnings = if file_entries.is_empty() {
            vec![JsonWarning::new(
                "no_results",
                "No nodes matched the requested parse filters",
            )]
        } else {
            Vec::new()
        };
        let data = ParseJsonData {
            mode: "parse".to_string(),
            files: file_entries,
            unsupported_files,
            stats: stats_out,
        };
        write_json_success(out, "parse", data, warnings)?;
    }

    Ok(())
}
