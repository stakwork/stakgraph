use std::collections::{HashMap, HashSet, VecDeque};

use ast::lang::graphs::EdgeType;
use console::style;
use serde::Serialize;
use shared::{Error, Result};

use super::args::DepsArgs;
use super::output::{write_json_success, JsonWarning, Output, OutputMode};
use super::progress::CliSpinner;
use super::utils::{
    build_graph_for_files_with_options, expand_dirs_for_parse, parse_node_types,
    rel_path_from_cwd,
};

#[derive(Serialize)]
struct DependencySeed {
    node_type: String,
    name: String,
    file: String,
    line: usize,
}

#[derive(Serialize)]
struct DependencyEdge {
    source_name: String,
    source_file: String,
    target_name: String,
    target_file: String,
    depth: usize,
    verified: bool,
}

#[derive(Serialize)]
struct DependencyTreeData {
    query: String,
    files: Vec<String>,
    depth: usize,
    allow_unverified: bool,
    seeds: Vec<DependencySeed>,
    edges: Vec<DependencyEdge>,
}

pub async fn run(
    args: &DepsArgs,
    out: &mut Output,
    show_progress: bool,
    output_mode: OutputMode,
) -> Result<()> {
    let node_type_filter = args
        .r#type
        .as_deref()
        .map(|t| {
            parse_node_types(&[t.to_string()]).and_then(|v| {
                v.into_iter()
                    .next()
                    .ok_or_else(|| Error::validation("--type must specify exactly one node type"))
            })
        })
        .transpose()?;

    let files = expand_dirs_for_parse(&args.files);
    if files.is_empty() {
        return Err(Error::validation(
            "no parseable files found in the given paths",
        ));
    }

    let spinner = if show_progress {
        Some(CliSpinner::new(&format!(
            "Parsing {} file(s)...",
            files.len()
        )))
    } else {
        None
    };

    let graph = build_graph_for_files_with_options(&files, args.allow).await?;

    if let Some(sp) = &spinner {
        sp.finish_and_clear();
    }

    // Find the seed node(s) matching the requested name and type
    let seeds: Vec<_> = graph
        .nodes
        .iter()
        .filter(|n| {
            node_type_filter.as_ref().map_or(true, |nt| n.node_type == *nt)
                && n.node_data.name == args.name
        })
        .collect();

    if seeds.is_empty() {
        let type_label = args.r#type.as_deref().unwrap_or("node");
        return Err(Error::validation(format!(
            "no {} named '{}' found in the parsed files",
            type_label, args.name
        )));
    }

    if output_mode.is_json() {
        let mut result_edges = Vec::new();
        for seed in &seeds {
            collect_dependency_edges(&graph, seed, args.depth, args.allow, &mut result_edges);
        }
        let data = DependencyTreeData {
            query: args.name.clone(),
            files: files.clone(),
            depth: args.depth,
            allow_unverified: args.allow,
            seeds: seeds
                .iter()
                .map(|seed| DependencySeed {
                    node_type: seed.node_type.to_string(),
                    name: seed.node_data.name.clone(),
                    file: seed.node_data.file.clone(),
                    line: seed.node_data.start + 1,
                })
                .collect(),
            edges: result_edges,
        };
        let warnings = if data.edges.is_empty() {
            vec![JsonWarning::new(
                "no_dependencies",
                format!("No outgoing dependency edges found for '{}'", args.name),
            )]
        } else {
            Vec::new()
        };
        write_json_success(out, "deps", data, warnings)?;
        return Ok(());
    }

    for seed in seeds {
        let file = &seed.node_data.file;
        let line = seed.node_data.start + 1;
        out.writeln(format!(
            "{}: {}  [{}:{}]",
            style(seed.node_type.to_string()).bold().cyan(),
            style(&args.name).bold().white(),
            style(rel_path_from_cwd(file)).dim(),
            style(line).dim()
        ))?;

        let seed_key = (&seed.node_data.name, &seed.node_data.file);

        // BFS over Calls edges
        // Queue: (source_name, source_file, depth, prefix_str, is_last)
        struct QueueItem {
            name: String,
            file: String,
            depth: usize,
            prefix: String,
            is_last: bool,
        }

        let mut queue: VecDeque<QueueItem> = VecDeque::new();
        let mut visited: HashSet<(String, String)> = HashSet::new();
        visited.insert((seed_key.0.clone(), seed_key.1.clone()));

        // Seed callees
        let callees = direct_callees(&graph, &args.name, file, args.allow);
        for (i, (callee_name, callee_file)) in callees.iter().enumerate() {
            let is_last = i == callees.len() - 1;
            queue.push_back(QueueItem {
                name: callee_name.clone(),
                file: callee_file.clone(),
                depth: 1,
                prefix: String::new(),
                is_last,
            });
        }

        while let Some(item) = queue.pop_front() {
            let connector = if item.is_last { "└── " } else { "├── " };
            let node_line = if item.file == "unverified" {
                format!(
                    "{}{}{}",
                    item.prefix,
                    connector,
                    style(&item.name).white()
                )
            } else {
                let found = graph
                    .nodes
                    .iter()
                    .find(|n| n.node_data.name == item.name && n.node_data.file == item.file);
                let display_line = found.map(|n| n.node_data.start + 1).unwrap_or(0);
                let node_type_label = found
                    .map(|n| n.node_type.to_string())
                    .unwrap_or_default();
                format!(
                    "{}{}{}  [{}]  [{}:{}]",
                    item.prefix,
                    connector,
                    style(&item.name).white(),
                    style(&node_type_label).cyan(),
                    style(rel_path_from_cwd(&item.file)).dim(),
                    style(display_line).dim()
                )
            };
            out.writeln(node_line)?;

            let key = (item.name.clone(), item.file.clone());
            let max_depth = args.depth;
            if (max_depth == 0 || item.depth < max_depth) && !visited.contains(&key) {
                visited.insert(key);
                if item.file == "unverified" {
                    continue;
                }
                let child_callees = direct_callees(&graph, &item.name, &item.file, args.allow);
                let child_prefix = format!(
                    "{}{}",
                    item.prefix,
                    if item.is_last { "    " } else { "│   " }
                );
                for (i, (cn, cf)) in child_callees.iter().enumerate() {
                    let is_last = i == child_callees.len() - 1;
                    queue.push_back(QueueItem {
                        name: cn.clone(),
                        file: cf.clone(),
                        depth: item.depth + 1,
                        prefix: child_prefix.clone(),
                        is_last,
                    });
                }
            } else if !visited.contains(&key) {
                visited.insert(key);
            }
        }

        out.newline()?;
    }

    Ok(())
}

fn collect_dependency_edges(
    graph: &ast::lang::graphs::ArrayGraph,
    seed: &ast::lang::Node,
    max_depth: usize,
    allow_unverified: bool,
    result_edges: &mut Vec<DependencyEdge>,
) {
    let mut queue: VecDeque<(String, String, usize)> = VecDeque::new();
    let mut visited: HashSet<(String, String)> = HashSet::new();

    let seed_name = seed.node_data.name.clone();
    let seed_file = seed.node_data.file.clone();
    visited.insert((seed_name.clone(), seed_file.clone()));

    for (callee_name, callee_file) in direct_callees(graph, &seed_name, &seed_file, allow_unverified)
    {
        result_edges.push(DependencyEdge {
            source_name: seed_name.clone(),
            source_file: seed_file.clone(),
            target_name: callee_name.clone(),
            target_file: callee_file.clone(),
            depth: 1,
            verified: callee_file != "unverified",
        });
        queue.push_back((callee_name, callee_file, 1));
    }

    while let Some((name, file, depth)) = queue.pop_front() {
        let key = (name.clone(), file.clone());
        if visited.contains(&key) {
            continue;
        }
        visited.insert(key);

        if max_depth != 0 && depth >= max_depth {
            continue;
        }
        if file == "unverified" {
            continue;
        }

        for (callee_name, callee_file) in direct_callees(graph, &name, &file, allow_unverified) {
            result_edges.push(DependencyEdge {
                source_name: name.clone(),
                source_file: file.clone(),
                target_name: callee_name.clone(),
                target_file: callee_file.clone(),
                depth: depth + 1,
                verified: callee_file != "unverified",
            });
            queue.push_back((callee_name, callee_file, depth + 1));
        }
    }
}

fn direct_callees(
    graph: &ast::lang::graphs::ArrayGraph,
    name: &str,
    file: &str,
    allow_unverified: bool,
) -> Vec<(String, String)> {
    // Group all call targets by callee name, collecting all unique files
    let mut by_name: HashMap<String, HashSet<String>> = HashMap::new();
    for e in graph.edges.iter().filter(|e| {
        e.edge == EdgeType::Calls
            && e.source.node_data.name == name
            && (file == "unverified" || e.source.node_data.file == file)
    }) {
        let callee_file = e.target.node_data.file.clone();
        if !allow_unverified && callee_file == "unverified" {
            continue;
        }
        by_name
            .entry(e.target.node_data.name.clone())
            .or_default()
            .insert(callee_file);
    }
    // For each callee name: if any verified resolution exists, emit only the verified ones;
    // otherwise emit a single unverified entry.
    let mut result = Vec::new();
    for (callee_name, files) in by_name {
        let verified: Vec<_> = files.iter().filter(|f| *f != "unverified").cloned().collect();
        if !verified.is_empty() {
            for f in verified {
                result.push((callee_name.clone(), f));
            }
        } else if allow_unverified {
            result.push((callee_name, "unverified".to_string()));
        }
    }
    result
}
