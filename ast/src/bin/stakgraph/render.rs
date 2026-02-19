use std::collections::HashMap;

use anyhow::Result;

use ast::lang::graphs::EdgeType;
use ast::lang::graphs::NodeType;
use ast::lang::ArrayGraph;
use ast::lang::Edge;
use ast::lang::Node;

pub fn common_ancestor(files: &[String]) -> Option<std::path::PathBuf> {
    if files.is_empty() {
        return None;
    }

    let mut ancestors: Vec<Vec<std::path::PathBuf>> = Vec::new();
    for file in files {
        let abs_path = std::fs::canonicalize(file).ok()?;
        let mut path_ancestors = Vec::new();
        let mut current = abs_path.as_path();
        while let Some(parent) = current.parent() {
            path_ancestors.push(parent.to_path_buf());
            current = parent;
        }
        path_ancestors.reverse();
        ancestors.push(path_ancestors);
    }

    if ancestors.is_empty() {
        return None;
    }

    let mut common = std::path::PathBuf::new();
    let min_len = ancestors.iter().map(|a| a.len()).min()?;

    for i in 0..min_len {
        let dir = &ancestors[0][i];
        if ancestors.iter().all(|a| a.get(i) == Some(dir)) {
            common = dir.clone();
        } else {
            break;
        }
    }

    if common.as_os_str().is_empty() {
        None
    } else {
        Some(common)
    }
}

pub fn first_lines(text: &str, n: usize, max_line_len: usize) -> String {
    text.lines()
        .take(n)
        .map(|line| {
            if line.len() > max_line_len {
                &line[..max_line_len]
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn format_lines(start: usize, end: usize) -> String {
    if start != end {
        format!("{}-{}", start + 1, end + 1)
    } else {
        format!("{}", start + 1)
    }
}

fn get_language_delimiter(file: &str) -> &'static str {
    if file.ends_with(".rs") {
        "::"
    } else {
        "."
    }
}

fn format_function_name_with_operand(node: &Node) -> String {
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

fn build_edge_indices(edges: &[Edge]) -> (HashMap<String, Vec<&Edge>>, HashMap<String, Vec<&Edge>>) {
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

fn print_function_edges(node: &Node, edges_by_source: &HashMap<String, Vec<&Edge>>, graph: &ArrayGraph) {
    let source_key = ast::utils::create_node_key(node).to_lowercase();
    let source_file = &node.node_data.file;

    if let Some(edges) = edges_by_source.get(&source_key) {
        for edge in edges {
            let target_name = &edge.target.node_data.name;
            let target_line = edge.target.node_data.start;
            let target_file = &edge.target.node_data.file;

            let target_display = {
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

                if let Some(op) = operand {
                    let file_for_delimiter = if target_file == "unverified" {
                        source_file
                    } else {
                        target_file
                    };
                    let delimiter = get_language_delimiter(file_for_delimiter);
                    format!("{}{}{}", op, delimiter, target_name)
                } else {
                    target_name.clone()
                }
            };

            let file_info = if source_file != target_file && target_file != "unverified" {
                let filename = std::path::Path::new(target_file)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(target_file);
                format!(" [{}]", filename)
            } else {
                String::new()
            };

            println!(
                "  • {:?} → {} (L{}){}",
                edge.edge,
                target_display,
                target_line + 1,
                file_info
            );
        }
    }
}

fn print_node_summary(node: &ast::lang::graphs::Node) {
    let nd = &node.node_data;
    let name = format_function_name_with_operand(node);
    let lines = format_lines(nd.start, nd.end);

    println!("{}: {} ({})", node.node_type, name, lines);

    if let Some(docs) = &nd.docs {
        println!("Docs: {}", first_lines(docs.as_str(), 3, 200));
    }

    if let Some(interface) = nd.meta.get("interface") {
        println!("```\n{}\n```", interface);
    } else {
        let body_lines = match node.node_type {
            NodeType::Function | NodeType::Endpoint | NodeType::Var => 20,
            NodeType::DataModel | NodeType::Import | NodeType::Request => 100,
            NodeType::UnitTest | NodeType::IntegrationTest | NodeType::E2eTest => 0,
            _ => 0,
        };

        if body_lines > 0 && !nd.body.is_empty() {
            let body = if matches!(node.node_type, NodeType::Import) {
                nd.body
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                nd.body.clone()
            };
            let body_preview = first_lines(&body, body_lines, 200);
            println!("```\n{}\n```", body_preview);
        }
    }
}

pub fn print_single_file_nodes(graph: &ArrayGraph, file_path: &str) -> Result<()> {
    let file_path = std::fs::canonicalize(file_path)?
        .to_string_lossy()
        .to_string();

    println!("File: {}", file_path);

    let mut nodes: Vec<_> = graph
        .nodes
        .iter()
        .filter(|node| {
            if matches!(node.node_type, NodeType::File | NodeType::Directory) {
                return false;
            }
            let node_file = std::fs::canonicalize(&node.node_data.file)
                .unwrap_or_else(|_| std::path::PathBuf::from(&node.node_data.file))
                .to_string_lossy()
                .to_string();
            node_file == file_path
        })
        .collect();

    nodes.sort_by_key(|node| node.node_data.start);

    let (edges_by_source, _edges_by_target) = build_edge_indices(&graph.edges);

    for node in nodes {
        print_node_summary(node);

        if matches!(node.node_type, NodeType::Function) {
            print_function_edges(node, &edges_by_source, graph);
        }

        println!();
    }

    Ok(())
}
