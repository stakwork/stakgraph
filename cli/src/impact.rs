use std::collections::{HashMap, HashSet, VecDeque};

use ast::lang::graphs::{EdgeType, NodeType};
use console::style;
use serde::Serialize;
use shared::{Error, Result};

use super::args::ImpactArgs;
use super::output::{write_json_success, JsonWarning, Output, OutputMode};
use super::progress::CliSpinner;
use super::utils::{
    build_graph_for_files_with_options, expand_dirs_for_parse, parse_node_types,
    rel_path_from_cwd,
};

const REVERSE_EDGE_TYPES: &[EdgeType] = &[
    EdgeType::Calls,
    EdgeType::Uses,
    EdgeType::Handler,
    EdgeType::Renders,
    EdgeType::Imports,
    EdgeType::Operand,
];

#[derive(Serialize)]
struct ImpactSeed {
    node_type: String,
    name: String,
    file: String,
    line: usize,
}

#[derive(Serialize)]
struct ChainHop {
    edge: String,
    via: String,
}

#[derive(Serialize)]
struct AffectedNode {
    node_type: String,
    name: String,
    file: String,
    line: usize,
    depth: usize,
    edge_chain: Vec<ChainHop>,
}

#[derive(Serialize)]
struct ImpactSummary {
    total: usize,
    by_type: HashMap<String, usize>,
}

#[derive(Serialize)]
struct ImpactData {
    seeds: Vec<ImpactSeed>,
    files: Vec<String>,
    depth: usize,
    summary: ImpactSummary,
    affected: Vec<AffectedNode>,
}

pub async fn run(
    args: &ImpactArgs,
    out: &mut Output,
    show_progress: bool,
    output_mode: OutputMode,
) -> Result<()> {
    if args.name.is_none() && args.file.is_none() {
        return Err(Error::validation(
            "at least one of --name or --file is required",
        ));
    }

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

    let seed_node_types: HashSet<NodeType> = [
        NodeType::Function,
        NodeType::Endpoint,
        NodeType::Class,
        NodeType::Trait,
        NodeType::DataModel,
        NodeType::Page,
        NodeType::Var,
        NodeType::UnitTest,
        NodeType::IntegrationTest,
        NodeType::E2eTest,
    ]
    .into_iter()
    .collect();

    let seeds: Vec<_> = graph
        .nodes
        .iter()
        .filter(|n| {
            if let Some(ref nt) = node_type_filter {
                if n.node_type != *nt {
                    return false;
                }
            } else if !seed_node_types.contains(&n.node_type) {
                return false;
            }
            let name_ok = args
                .name
                .as_ref()
                .map_or(true, |name| n.node_data.name == *name);
            let file_ok = args
                .file
                .as_ref()
                .map_or(true, |f| n.node_data.file.ends_with(f.as_str()));
            name_ok && file_ok
        })
        .collect();

    if seeds.is_empty() {
        let label = match (&args.name, &args.file) {
            (Some(name), Some(file)) => format!("'{}' in {}", name, file),
            (Some(name), None) => format!("'{}'", name),
            (None, Some(file)) => format!("nodes in {}", file),
            _ => unreachable!(),
        };
        return Err(Error::validation(format!(
            "no matching nodes found for {}",
            label
        )));
    }

    let mut all_affected: Vec<AffectedNode> = Vec::new();
    let mut visited: HashSet<(String, String)> = HashSet::new();

    for seed in &seeds {
        visited.insert((seed.node_data.name.clone(), seed.node_data.file.clone()));
    }

    for seed in &seeds {
        collect_impact(
            &graph,
            &seed.node_data.name,
            &seed.node_data.file,
            args.depth,
            &mut visited,
            &mut all_affected,
        );
    }

    all_affected.sort_by(|a, b| a.node_type.cmp(&b.node_type).then(a.name.cmp(&b.name)));

    let mut by_type: HashMap<String, usize> = HashMap::new();
    for a in &all_affected {
        *by_type.entry(a.node_type.clone()).or_default() += 1;
    }

    if output_mode.is_json() {
        let data = ImpactData {
            seeds: seeds
                .iter()
                .map(|s| ImpactSeed {
                    node_type: s.node_type.to_string(),
                    name: s.node_data.name.clone(),
                    file: s.node_data.file.clone(),
                    line: s.node_data.start + 1,
                })
                .collect(),
            files: files.clone(),
            depth: args.depth,
            summary: ImpactSummary {
                total: all_affected.len(),
                by_type: by_type.clone(),
            },
            affected: all_affected,
        };
        let warnings = if data.affected.is_empty() {
            vec![JsonWarning::new(
                "no_impact",
                "No upstream dependents found",
            )]
        } else {
            Vec::new()
        };
        write_json_success(out, "impact", data, warnings)?;
        return Ok(());
    }

    for seed in &seeds {
        out.writeln(format!(
            "Impact: {}  {}  [{}:{}]",
            style(seed.node_type.to_string()).bold().cyan(),
            style(&seed.node_data.name).bold().white(),
            style(rel_path_from_cwd(&seed.node_data.file)).dim(),
            style(seed.node_data.start + 1).dim()
        ))?;
    }

    if all_affected.is_empty() {
        out.writeln(format!(
            "  {}",
            style("No upstream dependents found.").dim()
        ))?;
        return Ok(());
    }

    let summary_parts: Vec<String> = {
        let order = [
            "Endpoint",
            "UnitTest",
            "IntegrationTest",
            "E2eTest",
            "Function",
            "Class",
            "Page",
        ];
        let mut parts = Vec::new();
        for t in &order {
            if let Some(count) = by_type.get(*t) {
                let label = if *count == 1 {
                    t.to_string()
                } else {
                    format!("{}s", t)
                };
                parts.push(format!("{} {}", count, label));
            }
        }
        for (t, count) in &by_type {
            if !order.contains(&t.as_str()) {
                parts.push(format!("{} {}", count, t));
            }
        }
        parts
    };

    out.writeln(format!(
        "  {} affected (depth: {})",
        style(summary_parts.join(", ")).bold(),
        args.depth
    ))?;
    out.newline()?;

    let grouped = group_by_type(&all_affected);
    let display_order = [
        "Endpoint",
        "UnitTest",
        "IntegrationTest",
        "E2eTest",
        "Function",
        "Class",
        "Page",
    ];

    for type_name in &display_order {
        if let Some(nodes) = grouped.get(*type_name) {
            print_group(out, type_name, nodes)?;
        }
    }
    for (type_name, nodes) in &grouped {
        if !display_order.contains(&type_name.as_str()) {
            print_group(out, type_name, nodes)?;
        }
    }

    Ok(())
}

fn style_edge(edge: &str) -> String {
    match edge {
        "CALLS" => style(edge).cyan().to_string(),
        "USES" => style(edge).cyan().dim().to_string(),
        "HANDLER" => style(edge).yellow().bold().to_string(),
        "RENDERS" => style(edge).green().to_string(),
        "IMPORTS" => style(edge).magenta().to_string(),
        "OPERAND" => style(edge).blue().to_string(),
        _ => style(edge).dim().to_string(),
    }
}

fn print_group(out: &mut Output, type_name: &str, nodes: &[&AffectedNode]) -> Result<()> {
    let type_style = match type_name {
        "Endpoint" => style(format!("  {}:", type_name)).yellow().bold(),
        "Function" => style(format!("  {}:", type_name)).green().bold(),
        "Class" => style(format!("  {}:", type_name)).blue().bold(),
        "DataModel" => style(format!("  {}:", type_name)).magenta().bold(),
        "UnitTest" | "IntegrationTest" | "E2eTest" => {
            style(format!("  {}:", type_name)).bright().bold()
        }
        _ => style(format!("  {}:", type_name)).bold(),
    };
    out.writeln(type_style.to_string())?;

    for node in nodes {
        let chain = if node.edge_chain.is_empty() {
            String::new()
        } else {
            let parts: Vec<String> = node
                .edge_chain
                .iter()
                .rev()
                .map(|hop| {
                    format!(
                        "{} {} {}",
                        style("←").dim(),
                        style_edge(&hop.edge),
                        style(&hop.via).dim()
                    )
                })
                .collect();
            format!("  {}", parts.join(" "))
        };
        out.writeln(format!(
            "    {}  [{}:{}]{}",
            style(&node.name).white(),
            style(rel_path_from_cwd(&node.file)).dim(),
            style(node.line).dim(),
            chain,
        ))?;
    }
    out.newline()?;
    Ok(())
}

fn group_by_type<'a>(affected: &'a [AffectedNode]) -> HashMap<String, Vec<&'a AffectedNode>> {
    let mut groups: HashMap<String, Vec<&AffectedNode>> = HashMap::new();
    for node in affected {
        groups.entry(node.node_type.clone()).or_default().push(node);
    }
    groups
}

fn collect_impact(
    graph: &ast::lang::graphs::ArrayGraph,
    seed_name: &str,
    seed_file: &str,
    max_depth: usize,
    visited: &mut HashSet<(String, String)>,
    result: &mut Vec<AffectedNode>,
) {
    struct BfsItem {
        name: String,
        file: String,
        depth: usize,
        // each hop: (edge_type_str, via_node_name) — path from this node back to seed
        edge_chain: Vec<(String, String)>,
    }

    let mut queue: VecDeque<BfsItem> = VecDeque::new();

    for (source_ref, edge_type) in graph.find_dependents(seed_name, seed_file, REVERSE_EDGE_TYPES) {
        let key = (
            source_ref.node_data.name.clone(),
            source_ref.node_data.file.clone(),
        );
        if visited.contains(&key) {
            continue;
        }
        visited.insert(key);
        queue.push_back(BfsItem {
            name: source_ref.node_data.name.clone(),
            file: source_ref.node_data.file.clone(),
            depth: 1,
            edge_chain: vec![(format!("{}", edge_type), seed_name.to_string())],
        });
    }

    while let Some(item) = queue.pop_front() {
        let found = graph
            .nodes
            .iter()
            .find(|n| n.node_data.name == item.name && n.node_data.file == item.file);

        let node_type = found
            .map(|n| n.node_type.to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        let line = found.map(|n| n.node_data.start + 1).unwrap_or(0);

        result.push(AffectedNode {
            node_type,
            name: item.name.clone(),
            file: item.file.clone(),
            line,
            depth: item.depth,
            edge_chain: item
                .edge_chain
                .iter()
                .map(|(e, v)| ChainHop {
                    edge: e.clone(),
                    via: v.clone(),
                })
                .collect(),
        });

        if max_depth == 0 || item.depth < max_depth {
            for (source_ref, edge_type) in
                graph.find_dependents(&item.name, &item.file, REVERSE_EDGE_TYPES)
            {
                let key = (
                    source_ref.node_data.name.clone(),
                    source_ref.node_data.file.clone(),
                );
                if visited.contains(&key) {
                    continue;
                }
                visited.insert(key);
                let mut chain = item.edge_chain.clone();
                chain.push((format!("{}", edge_type), item.name.clone()));
                queue.push_back(BfsItem {
                    name: source_ref.node_data.name.clone(),
                    file: source_ref.node_data.file.clone(),
                    depth: item.depth + 1,
                    edge_chain: chain,
                });
            }
        }
    }
}
