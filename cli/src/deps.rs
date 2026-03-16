use std::collections::{HashSet, VecDeque};

use ast::lang::graphs::EdgeType;
use console::style;
use shared::{Error, Result};

use super::args::DepsArgs;
use super::output::Output;
use super::progress::CliSpinner;
use super::utils::{
    build_graph_for_files_with_options, expand_dirs_for_parse, parse_node_types,
};

pub async fn run(args: &DepsArgs, out: &mut Output, show_progress: bool) -> Result<()> {
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

    for seed in seeds {
        let file = &seed.node_data.file;
        let line = seed.node_data.start + 1;
        out.writeln(format!(
            "{}: {}  [{}:{}]",
            style(seed.node_type.to_string()).bold().cyan(),
            style(&args.name).bold().white(),
            style(file).dim(),
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
                    "{}{}{}  [{}]",
                    item.prefix,
                    connector,
                    style(&item.name).white(),
                    style("unverified").dim().yellow()
                )
            } else {
                let display_line = graph
                    .nodes
                    .iter()
                    .find(|n| n.node_data.name == item.name && n.node_data.file == item.file)
                    .map(|n| n.node_data.start + 1)
                    .unwrap_or(0);
                format!(
                    "{}{}{}  [{}:{}]",
                    item.prefix,
                    connector,
                    style(&item.name).white(),
                    style(&item.file).dim(),
                    style(display_line).dim()
                )
            };
            out.writeln(node_line)?;

            let key = (item.name.clone(), item.file.clone());
            let max_depth = args.depth;
            if (max_depth == 0 || item.depth < max_depth) && !visited.contains(&key) {
                visited.insert(key);
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

fn direct_callees(
    graph: &ast::lang::graphs::ArrayGraph,
    name: &str,
    file: &str,
    allow_unverified: bool,
) -> Vec<(String, String)> {
    graph
        .edges
        .iter()
        .filter(|e| {
            e.edge == EdgeType::Calls
                && e.source.node_data.name == name
                && (file == "unverified" || e.source.node_data.file == file)
        })
        .filter(|e| allow_unverified || e.target.node_data.file != "unverified")
        .map(|e| {
            (
                e.target.node_data.name.clone(),
                e.target.node_data.file.clone(),
            )
        })
        .collect()
}
