use std::collections::HashMap;
use std::collections::HashSet;
use std::io;

use console::{style, Style};
use serde::Serialize;
use shared::Result;

use ast::lang::graphs::EdgeType;
use ast::lang::graphs::NodeType;
use ast::lang::ArrayGraph;
use ast::lang::Edge;
use ast::lang::Node;

use super::output::Output;
use super::utils::first_lines;

fn format_lines(start: usize, end: usize) -> String {
    if start != end {
        format!("{}-{}", start + 1, end + 1)
    } else {
        format!("{}", start + 1)
    }
}

pub fn get_language_delimiter(file: &str) -> &'static str {
    if file.ends_with(".rs") {
        "::"
    } else {
        "."
    }
}

fn style_for_node_type(node_type: &NodeType) -> Style {
    match node_type {
        NodeType::Function => Style::new().green().bold(),
        NodeType::Endpoint => Style::new().yellow().bold(),
        NodeType::Request => Style::new().cyan().bold(),
        NodeType::DataModel => Style::new().magenta().bold(),
        NodeType::Class => Style::new().blue().bold(),
        NodeType::Import => Style::new().dim(),
        NodeType::UnitTest | NodeType::IntegrationTest | NodeType::E2eTest => {
            Style::new().bright().bold()
        }
        _ => Style::new().bold(),
    }
}

pub fn node_display_name(node: &Node) -> String {
    let nd = &node.node_data;

    if let Some(verb) = nd.meta.get("verb") {
        format!("{} {}", verb, nd.name)
    } else if matches!(node.node_type, NodeType::Import) {
        String::new()
    } else if matches!(node.node_type, NodeType::Function) {
        if let Some(operand) = nd.meta.get("operand") {
            let delimiter = get_language_delimiter(&nd.file);
            format!("{}{}{}", operand, delimiter, nd.name)
        } else {
            nd.name.clone()
        }
    } else {
        nd.name.clone()
    }
}

pub fn node_code_preview(node: &Node) -> Option<String> {
    let nd = &node.node_data;
    if let Some(interface) = nd.meta.get("interface") {
        return Some(interface.clone());
    }
    let body_lines = match node.node_type {
        NodeType::Function | NodeType::Var => 20,
        NodeType::Endpoint | NodeType::Request => 0,
        NodeType::DataModel | NodeType::Import => 100,
        NodeType::UnitTest | NodeType::IntegrationTest | NodeType::E2eTest => 0,
        _ => 0,
    };
    if body_lines == 0 || nd.body.is_empty() {
        return None;
    }
    let body = if matches!(node.node_type, NodeType::Import) {
        nd.body
            .lines()
            .filter(|l| !l.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        nd.body.clone()
    };
    Some(first_lines(&body, body_lines, 200))
}

#[derive(Serialize)]
pub struct CallRef {
    pub name: String,
    pub line: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

pub fn resolve_call_ref(
    edge: &Edge,
    source_file: &str,
    graph: &ArrayGraph,
) -> CallRef {
    let target_name = &edge.target.node_data.name;
    let target_file = &edge.target.node_data.file;

    let operand = edge.operand.as_ref().or_else(|| {
        graph
            .nodes
            .iter()
            .find(|n| {
                ast::utils::create_node_key(n).to_lowercase()
                    == ast::utils::create_node_key_from_ref(&edge.target).to_lowercase()
            })
            .and_then(|n| n.node_data.meta.get("operand"))
    });

    let name = if let Some(op) = operand {
        let file_for_delimiter = if target_file == "unverified" {
            source_file
        } else {
            target_file
        };
        let delimiter = get_language_delimiter(file_for_delimiter);
        format!("{}{}{}", op, delimiter, target_name)
    } else {
        target_name.clone()
    };

    let file = if source_file != target_file && target_file != "unverified" {
        std::path::Path::new(target_file)
            .file_name()
            .and_then(|n| n.to_str())
            .map(|s| s.to_string())
    } else {
        None
    };

    CallRef {
        name,
        line: edge.target.node_data.start + 1,
        file,
    }
}

pub fn build_call_index<'a>(edges: &'a [Edge]) -> HashMap<String, Vec<&'a Edge>> {
    let mut index = HashMap::new();
    for edge in edges {
        if matches!(edge.edge, EdgeType::Calls | EdgeType::Uses) {
            let key = ast::utils::create_node_key_from_ref(&edge.source).to_lowercase();
            index.entry(key).or_insert_with(Vec::new).push(edge);
        }
    }
    index
}

fn build_edge_indices(
    edges: &[Edge],
) -> (HashMap<String, Vec<&Edge>>, HashMap<String, Vec<&Edge>>) {
    let mut edges_by_source = HashMap::new();
    let mut edges_by_target = HashMap::new();

    for edge in edges {
        match edge.edge {
            EdgeType::Calls | EdgeType::Uses => {
                let source_key = ast::utils::create_node_key_from_ref(&edge.source).to_lowercase();
                edges_by_source
                    .entry(source_key)
                    .or_insert_with(Vec::new)
                    .push(edge);
            }
            EdgeType::Operand => {
                let target_key = ast::utils::create_node_key_from_ref(&edge.target).to_lowercase();
                edges_by_target
                    .entry(target_key)
                    .or_insert_with(Vec::new)
                    .push(edge);
            }
            _ => {}
        }
    }

    (edges_by_source, edges_by_target)
}

fn print_function_edges(
    out: &mut Output,
    node: &Node,
    edges_by_source: &HashMap<String, Vec<&Edge>>,
    graph: &ArrayGraph,
) -> io::Result<()> {
    let source_key = ast::utils::create_node_key(node).to_lowercase();

    if let Some(edges) = edges_by_source.get(&source_key) {
        for edge in edges {
            let cr = resolve_call_ref(edge, &node.node_data.file, graph);
            let file_info = cr
                .file
                .map(|f| format!(" [{}]", f))
                .unwrap_or_default();

            let arrow = style("→").dim();
            let line_num = style(format!("L{}", cr.line)).dim();
            let file_info_styled = style(file_info).dim();

            out.writeln(format!(
                "  {} {} ({}){}",
                arrow, cr.name, line_num, file_info_styled
            ))?;
        }
    }
    Ok(())
}

fn print_node_summary(out: &mut Output, node: &ast::lang::graphs::Node) -> io::Result<()> {
    let nd = &node.node_data;
    let name = node_display_name(node);
    let lines = format_lines(nd.start, nd.end);

    let node_type_styled = style_for_node_type(&node.node_type).apply_to(&node.node_type);
    let lines_styled = style(format!("({})", lines)).dim();

    out.writeln(format!("{}: {} {}", node_type_styled, name, lines_styled))?;

    if matches!(node.node_type, NodeType::Endpoint) {
        if let Some(handler) = nd.meta.get("handler") {
            let handler_label = style("Handler:").dim();
            out.writeln(format!("  {} {}", handler_label, style(handler).green()))?;
        }
    }

    if let Some(docs) = &nd.docs {
        let docs_label = style("Docs:").dim();
        out.writeln(format!(
            "{} {}",
            docs_label,
            first_lines(docs.as_str(), 3, 200)
        ))?;
    }

    if let Some(code) = node_code_preview(node) {
        let fence = style("```").dim();
        out.writeln(format!("{}\n{}\n{}", fence, code, fence))?;
    }
    Ok(())
}

fn print_file_nodes_inner(
    out: &mut Output,
    graph: &ArrayGraph,
    file_path: &str,
    allowed_types: Option<&[NodeType]>,
) -> Result<()> {
    let file_path = std::fs::canonicalize(file_path)?
        .to_string_lossy()
        .to_string();

    let file_label = style("File:").bold().cyan();
    out.writeln(format!("{} {}", file_label, style(super::utils::rel_path_from_cwd(&file_path)).cyan()))?;

    let mut nodes: Vec<_> = graph
        .nodes
        .iter()
        .filter(|node| {
            if matches!(node.node_type, NodeType::File | NodeType::Directory) {
                return false;
            }
            if let Some(types) = allowed_types {
                if !types.contains(&node.node_type) {
                    return false;
                }
            }
            let node_file_str = &node.node_data.file;
            // Try canonicalize first (works when node stores an absolute/resolvable path)
            let node_file = std::fs::canonicalize(node_file_str)
                .map(|p| p.to_string_lossy().to_string());
            if let Ok(ref nf) = node_file {
                if *nf == file_path {
                    return true;
                }
            }
            // Fallback: node file may be a strip_tmp-ed relative path (e.g. "owner/repo/src/foo.ts")
            // that the canonical file_path ends with
            file_path.ends_with(node_file_str)
        })
        .collect();

    nodes.sort_by_key(|node| node.node_data.start);

    let endpoint_lines: HashSet<usize> = nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeType::Endpoint))
        .map(|n| n.node_data.start)
        .collect();
    let (edges_by_source, _edges_by_target) = build_edge_indices(&graph.edges);

    for node in nodes {
        if matches!(node.node_type, NodeType::Request)
            && endpoint_lines.contains(&node.node_data.start)
        {
            continue;
        }
        print_node_summary(out, node)?;

        if matches!(node.node_type, NodeType::Function) {
            print_function_edges(out, node, &edges_by_source, graph)?;
        }

        out.newline()?;
    }

    Ok(())
}

pub fn print_named_node(
    out: &mut Output,
    graph: &ArrayGraph,
    file_path: &str,
    node_name: &str,
    type_filter: Option<&[NodeType]>,
) -> Result<()> {
    let file_path = std::fs::canonicalize(file_path)?
        .to_string_lossy()
        .to_string();

    let mut matches: Vec<_> = graph
        .nodes
        .iter()
        .filter(|node| {
            if matches!(node.node_type, NodeType::File | NodeType::Directory) {
                return false;
            }
            if let Some(types) = type_filter {
                if !types.contains(&node.node_type) {
                    return false;
                }
            }
            let node_file_str = &node.node_data.file;
            let node_file = std::fs::canonicalize(node_file_str)
                .map(|p| p.to_string_lossy().to_string());
            let in_file = if let Ok(ref nf) = node_file {
                *nf == file_path
            } else {
                file_path.ends_with(node_file_str)
            };
            if !in_file {
                return false;
            }
            node.node_data.name == node_name
        })
        .collect();

    matches.sort_by_key(|node| node.node_data.start);

    if matches.is_empty() {
        let type_hint = type_filter
            .map(|t| {
                let names: Vec<_> = t.iter().map(|nt| nt.to_string()).collect();
                format!(" (type: {})", names.join(", "))
            })
            .unwrap_or_default();
        out.writeln(format!(
            "{}",
            style(format!(
                "No node named '{}'{} found in {}",
                node_name, type_hint, super::utils::rel_path_from_cwd(&file_path)
            ))
            .yellow()
        ))?;
        return Ok(());
    }

    if matches.len() > 1 {
        out.writeln(format!(
            "{}",
            style(format!(
                "Multiple nodes named '{}' found — use --type to disambiguate:",
                node_name
            ))
            .yellow()
        ))?;
        for node in &matches {
            let lines = format_lines(node.node_data.start, node.node_data.end);
            out.writeln(format!(
                "  {} ({}) at line {}",
                style(&node.node_type).bold(),
                node.node_data.name,
                lines
            ))?;
        }
        return Ok(());
    }

    let node = matches[0];
    let nd = &node.node_data;
    let name = node_display_name(node);
    let lines = format_lines(nd.start, nd.end);

    let node_type_styled = style_for_node_type(&node.node_type).apply_to(&node.node_type);
    let lines_styled = style(format!("({})", lines)).dim();
    out.writeln(format!("{}: {} {}", node_type_styled, name, lines_styled))?;

    if matches!(node.node_type, NodeType::Endpoint) {
        if let Some(handler) = nd.meta.get("handler") {
            let handler_label = style("Handler:").dim();
            out.writeln(format!("  {} {}", handler_label, style(handler).green()))?;
        }
    }

    if let Some(docs) = &nd.docs {
        let docs_label = style("Docs:").dim();
        out.writeln(format!("{} {}", docs_label, docs))?;
    }

    if !nd.body.is_empty() {
        let body = if matches!(node.node_type, NodeType::Import) {
            nd.body
                .lines()
                .filter(|line| !line.trim().is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            nd.body.clone()
        };
        let fence = style("```").dim();
        out.writeln(format!("{}\n{}\n{}", fence, body, fence))?;
    }

    let (edges_by_source, _) = build_edge_indices(&graph.edges);
    if matches!(node.node_type, NodeType::Function) {
        print_function_edges(out, node, &edges_by_source, graph)?;
    }

    Ok(())
}

pub fn print_single_file_nodes(
    out: &mut Output,
    graph: &ArrayGraph,
    file_path: &str,
) -> Result<()> {
    print_file_nodes_inner(out, graph, file_path, None)
}

pub fn print_single_file_nodes_filtered(
    out: &mut Output,
    graph: &ArrayGraph,
    file_path: &str,
    allowed_types: &[NodeType],
) -> Result<()> {
    print_file_nodes_inner(out, graph, file_path, Some(allowed_types))
}

pub fn render_file_nodes_filtered(
    graph: &ArrayGraph,
    file_path: &str,
    allowed_types: &[NodeType],
) -> Result<String> {
    let mut out = Output::new_buffer();
    print_file_nodes_inner(&mut out, graph, file_path, Some(allowed_types))?;
    Ok(out.into_string())
}
