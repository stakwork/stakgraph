use std::path::Path;
use std::str::FromStr;

use crate::lang::graphs::{EdgeType, Node, NodeType};
use crate::lang::Graph;

#[derive(Debug, Clone)]
enum Direction {
    Incoming,
    Outgoing,
}

#[derive(Debug, Clone)]
struct EdgeAnnotation {
    edge_type: EdgeType,
    direction: Direction,
    other_type: NodeType,
    other_name: String,
    other_file: String,
}

fn parse_node_type(s: &str) -> Option<NodeType> {
    match s {
        "Repository" => Some(NodeType::Repository),
        "Package" => Some(NodeType::Package),
        "Language" => Some(NodeType::Language),
        "Directory" => Some(NodeType::Directory),
        "File" => Some(NodeType::File),
        "Import" => Some(NodeType::Import),
        "Library" => Some(NodeType::Library),
        "Class" => Some(NodeType::Class),
        "Trait" => Some(NodeType::Trait),
        "Instance" => Some(NodeType::Instance),
        "Function" => Some(NodeType::Function),
        "Endpoint" => Some(NodeType::Endpoint),
        "Request" => Some(NodeType::Request),
        "DataModel" => Some(NodeType::DataModel),
        "Feature" => Some(NodeType::Feature),
        "Page" => Some(NodeType::Page),
        "Var" => Some(NodeType::Var),
        "UnitTest" => Some(NodeType::UnitTest),
        "IntegrationTest" => Some(NodeType::IntegrationTest),
        "E2eTest" => Some(NodeType::E2eTest),
        "Mock" => Some(NodeType::Mock),
        _ => None,
    }
}

fn parse_quoted_tokens(s: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut chars = s.chars().peekable();
    while chars.peek().is_some() {
        while chars.peek().map_or(false, |c| c.is_whitespace()) {
            chars.next();
        }
        match chars.peek() {
            None => break,
            Some('"') => {
                chars.next();
                let mut tok = String::new();
                loop {
                    match chars.next() {
                        None | Some('"') => break,
                        Some(ch) => tok.push(ch),
                    }
                }
                tokens.push(tok);
            }
            _ => {
                let mut tok = String::new();
                while chars.peek().map_or(false, |c| !c.is_whitespace()) {
                    tok.push(chars.next().unwrap());
                }
                if !tok.is_empty() {
                    tokens.push(tok);
                }
            }
        }
    }
    tokens
}

fn parse_file_annotations(source: &str) -> Vec<(NodeType, String, Vec<EdgeAnnotation>)> {
    let mut result = Vec::new();
    let mut current: Option<(NodeType, String, Vec<EdgeAnnotation>)> = None;

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("// @ast ") {
            if let Some(node_rest) = rest.strip_prefix("node: ") {
                if let Some(prev) = current.take() {
                    result.push(prev);
                }
                let toks = parse_quoted_tokens(node_rest);
                if toks.len() >= 2 {
                    if let Some(nt) = parse_node_type(&toks[0]) {
                        current = Some((nt, toks[1].clone(), Vec::new()));
                    }
                }
            } else if let Some(edge_rest) = rest.strip_prefix("edge: ") {
                if let Some((_, _, ref mut edges)) = current {
                    let toks = parse_quoted_tokens(edge_rest);
                    if toks.len() >= 5 {
                        let dir = match toks[1].as_str() {
                            "<-" => Direction::Incoming,
                            "->" => Direction::Outgoing,
                            _ => continue,
                        };
                        if let (Ok(et), Some(nt)) =
                            (EdgeType::from_str(&toks[0]), parse_node_type(&toks[2]))
                        {
                            edges.push(EdgeAnnotation {
                                edge_type: et,
                                direction: dir,
                                other_type: nt,
                                other_name: toks[3].clone(),
                                other_file: toks[4].clone(),
                            });
                        }
                    }
                }
            }
        }
    }
    if let Some(g) = current {
        result.push(g);
    }
    result
}

pub fn verify_file(source: &str, file_suffix: &str, graph: &impl Graph) -> Vec<String> {
    let groups = parse_file_annotations(source);
    let mut failures = Vec::new();

    for (node_type, node_name, edges) in &groups {
        let subject_data = match graph
            .find_node_by_name_and_file_end_with(node_type.clone(), node_name, file_suffix)
        {
            Some(nd) => nd,
            None => {
                failures.push(format!(
                    "FAIL node not found: {:?}(\"{}\") in {}",
                    node_type, node_name, file_suffix
                ));
                continue;
            }
        };
        let subject = Node::new(node_type.clone(), subject_data);

        for ea in edges {
            let other_data = match graph.find_node_by_name_and_file_end_with(
                ea.other_type.clone(),
                &ea.other_name,
                &ea.other_file,
            ) {
                Some(nd) => nd,
                None => {
                    failures.push(format!(
                        "FAIL node not found: {:?}(\"{}\") in {} (edge {:?} from {:?}(\"{}\"))",
                        ea.other_type,
                        ea.other_name,
                        ea.other_file,
                        ea.edge_type,
                        node_type,
                        node_name
                    ));
                    continue;
                }
            };
            let other = Node::new(ea.other_type.clone(), other_data);

            let (src, tgt) = match ea.direction {
                Direction::Incoming => (&other, &subject),
                Direction::Outgoing => (&subject, &other),
            };

            if !graph.has_edge(src, tgt, ea.edge_type.clone()) {
                let arrow = match ea.direction {
                    Direction::Incoming => "<-",
                    Direction::Outgoing => "->",
                };
                failures.push(format!(
                    "FAIL edge: {:?}  {:?}(\"{}\") {} {:?}(\"{}\")",
                    ea.edge_type, node_type, node_name, arrow, ea.other_type, ea.other_name
                ));
            }
        }
    }
    failures
}

pub fn walk_and_verify(fixture_dir: &Path, root: &Path, graph: &impl Graph) -> Vec<String> {
    let mut failures = Vec::new();
    walk_impl(fixture_dir, root, graph, &mut failures);
    failures
}

fn walk_impl(dir: &Path, root: &Path, graph: &impl Graph, failures: &mut Vec<String>) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = read.flatten().collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            if matches!(
                path.file_name().and_then(|n| n.to_str()).unwrap_or(""),
                "node_modules" | "target" | ".next" | ".git"
            ) {
                continue;
            }
            walk_impl(&path, root, graph, failures);
        } else {
            if !matches!(
                path.extension().and_then(|e| e.to_str()).unwrap_or(""),
                "ts" | "tsx" | "js" | "jsx" | "mdx"
            ) {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&path) else {
                continue;
            };
            if !src.contains("// @ast") {
                continue;
            }
            let suffix = path
                .strip_prefix(root)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            failures.extend(verify_file(&src, &suffix, graph));
        }
    }
}
