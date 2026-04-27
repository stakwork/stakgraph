use std::collections::BTreeMap;
use std::path::Path;
use std::str::FromStr;

use crate::lang::graphs::{EdgeType, Node, NodeType};
use crate::lang::{Graph, Lang};
use crate::repo::{Repo, Repos};
use lsp::Language;
use shared::error::Result;

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
    other_meta: BTreeMap<String, String>,
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

// Parses an optional "[key=value key2=value2]" block from the tail of a line.
// Returns the key→value map (empty if no block present).
fn parse_meta_filter(s: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let s = s.trim();
    if let (Some(open), Some(close)) = (s.rfind('['), s.rfind(']')) {
        if open < close {
            let inner = &s[open + 1..close];
            for pair in inner.split_whitespace() {
                if let Some((k, v)) = pair.split_once('=') {
                    map.insert(k.to_string(), v.to_string());
                }
            }
        }
    }
    map
}

#[derive(Debug, Clone)]
struct AbsentAnnotation {
    node_type: NodeType,
    name: String,
    file_suffix: String,
}

fn parse_absent_annotations(source: &str, prefix: &str) -> Vec<AbsentAnnotation> {
    let absent_prefix = format!("{}absent: ", prefix);
    let mut result = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(absent_prefix.as_str()) {
            let toks = parse_quoted_tokens(rest);
            if toks.len() >= 3 {
                if let Some(nt) = parse_node_type(&toks[0]) {
                    result.push(AbsentAnnotation {
                        node_type: nt,
                        name: toks[1].clone(),
                        file_suffix: toks[2].clone(),
                    });
                }
            }
        }
    }
    result
}

fn parse_file_annotations(
    source: &str,
    prefix: &str,
) -> Vec<(NodeType, String, BTreeMap<String, String>, Vec<EdgeAnnotation>)> {
    let mut result = Vec::new();
    let mut current: Option<(NodeType, String, BTreeMap<String, String>, Vec<EdgeAnnotation>)> =
        None;

    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            if let Some(node_rest) = rest.strip_prefix("node: ") {
                if let Some(prev) = current.take() {
                    result.push(prev);
                }
                let toks = parse_quoted_tokens(node_rest);
                if toks.len() >= 2 {
                    if let Some(nt) = parse_node_type(&toks[0]) {
                        let subject_meta = parse_meta_filter(node_rest);
                        current = Some((nt, toks[1].clone(), subject_meta, Vec::new()));
                    }
                }
            } else if let Some(edge_rest) = rest.strip_prefix("edge: ") {
                if let Some((_, _, _, ref mut edges)) = current {
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
                            let other_meta = parse_meta_filter(edge_rest);
                            edges.push(EdgeAnnotation {
                                edge_type: et,
                                direction: dir,
                                other_type: nt,
                                other_name: toks[3].clone(),
                                other_file: toks[4].clone(),
                                other_meta,
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

fn annotation_prefix_for_ext(ext: &str, default: &'static str) -> &'static str {
    match ext {
        "html" => "<!-- @ast ",
        "css" | "scss" | "sass" | "less" => "/* @ast ",
        _ => default,
    }
}

pub fn verify_file(source: &str, file_suffix: &str, graph: &impl Graph, prefix: &str) -> (Vec<String>, BTreeMap<NodeType, usize>) {
    let groups = parse_file_annotations(source, prefix);
    let absent = parse_absent_annotations(source, prefix);
    let mut failures: Vec<String> = Vec::new();
    let mut counts: BTreeMap<NodeType, usize> = BTreeMap::new();

    for (node_type, _, _, _) in &groups {
        *counts.entry(node_type.clone()).or_insert(0) += 1;
    }

    for a in &absent {
        if graph
            .find_node_by_name_and_file_end_with(a.node_type.clone(), &a.name, &a.file_suffix)
            .is_some()
        {
            failures.push(format!(
                "FAIL node should be absent: {:?}(\"{}\") in {} — but it was found",
                a.node_type, a.name, a.file_suffix
            ));
        }
    }

    for (node_type, node_name, subject_meta, edges) in &groups {
        let subject_data = if subject_meta.is_empty() {
            graph.find_node_by_name_and_file_end_with(node_type.clone(), node_name, file_suffix)
        } else {
            graph.find_node_by_name_file_and_meta(
                node_type.clone(),
                node_name,
                file_suffix,
                subject_meta,
            )
        };
        let subject_data = match subject_data {
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
            let other_data = if ea.other_meta.is_empty() {
                graph.find_node_by_name_and_file_end_with(
                    ea.other_type.clone(),
                    &ea.other_name,
                    &ea.other_file,
                )
            } else {
                graph.find_node_by_name_file_and_meta(
                    ea.other_type.clone(),
                    &ea.other_name,
                    &ea.other_file,
                    &ea.other_meta,
                )
            };
            let other_data = match other_data {
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
    (failures, counts)
}

pub fn walk_and_verify(fixture_dir: &Path, root: &Path, graph: &impl Graph, lang: &Language) -> Vec<String> {
    let mut failures = Vec::new();
    let mut counts: BTreeMap<NodeType, usize> = BTreeMap::new();
    let exts: Vec<&str> = lang.exts();
    let skip_dirs: Vec<&str> = lang.skip_dirs();
    walk_impl(fixture_dir, root, graph, &mut failures, &mut counts, &exts, &skip_dirs, lang);
    for (node_type, expected) in &counts {
        let actual = graph
            .find_nodes_by_type(node_type.clone())
            .iter()
            .filter(|n| !n.name.contains('\n'))
            .count();
        if actual != *expected {
            failures.push(format!(
                "FAIL count: {:?} expected {} got {}",
                node_type, expected, actual
            ));
        }
    }
    failures
}

fn walk_impl(
    dir: &Path,
    root: &Path,
    graph: &impl Graph,
    failures: &mut Vec<String>,
    counts: &mut BTreeMap<NodeType, usize>,
    exts: &[&str],
    skip_dirs: &[&str],
    lang: &Language,
) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = read.flatten().collect();
    entries.sort_by_key(|e| e.path());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            let dir_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if skip_dirs.contains(&dir_name) {
                continue;
            }
            walk_impl(&path, root, graph, failures, counts, exts, skip_dirs, lang);
        } else {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !exts.contains(&ext) {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&path) else {
                continue;
            };
            if !src.contains("@ast ") {
                continue;
            }
            let suffix = path
                .strip_prefix(root)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| path.to_string_lossy().to_string());
            let file_prefix = annotation_prefix_for_ext(ext, lang.annotation_prefix());
            let (file_failures, file_counts) = verify_file(&src, &suffix, graph, file_prefix);
            failures.extend(file_failures);
            for (nt, n) in file_counts {
                *counts.entry(nt).or_insert(0) += n;
            }
        }
    }
}

pub async fn run_fixture_test<G: Graph + Sync>(
    subdir: &str,
    lang: &str,
    annotation_lang: Language,
) -> Result<()> {
    let repo = Repo::new(
        subdir,
        Lang::from_str(lang).unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();
    let repos = Repos(vec![repo]);
    let graph = repos.build_graphs_inner::<G>().await?;
    graph.analysis();
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(subdir);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let failures = walk_and_verify(&fixture_dir, root, &graph, &annotation_lang);
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("{}", f);
        }
        panic!("{} annotation verification failure(s)", failures.len());
    }
    Ok(())
}
