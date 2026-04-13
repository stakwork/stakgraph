use std::collections::HashSet;

use ast::lang::graphs::{EdgeType, NodeType};
use ast::lang::ArrayGraph;
use ast::lang::Node;
use console::style;
use serde::Serialize;
use shared::{Error, Result};

use super::args::SearchArgs;
use super::output::{write_json_success, JsonWarning, Output, OutputMode};
use super::progress::CliSpinner;
use super::render::{node_display_name, style_for_node_type};
use super::utils::{
    build_graph_for_files_with_options, expand_dirs_for_parse, parse_node_types, rel_path_from_cwd,
};

const SEARCHABLE_TYPES: &[NodeType] = &[
    NodeType::Function,
    NodeType::Endpoint,
    NodeType::Request,
    NodeType::DataModel,
    NodeType::Class,
    NodeType::Trait,
    NodeType::Instance,
    NodeType::Var,
    NodeType::UnitTest,
    NodeType::IntegrationTest,
    NodeType::E2eTest,
    NodeType::Page,
    NodeType::Feature,
];

const TEST_TYPES: &[NodeType] = &[
    NodeType::UnitTest,
    NodeType::IntegrationTest,
    NodeType::E2eTest,
];

fn format_lines(start: usize, end: usize) -> String {
    if start != end {
        format!("{}-{}", start + 1, end + 1)
    } else {
        format!("{}", start + 1)
    }
}

#[derive(Serialize)]
struct ContextRef {
    name: String,
    node_type: String,
    file: String,
    line: usize,
}

#[derive(Serialize)]
struct RelatedRef {
    name: String,
    node_type: String,
    file: String,
    lines: String,
}

#[derive(Serialize)]
struct SearchResultNode {
    name: String,
    display_name: String,
    node_type: String,
    file: String,
    lines: String,
    score: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tests: Vec<RelatedRef>,
    callers: Vec<ContextRef>,
    callees: Vec<ContextRef>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    related: Vec<RelatedRef>,
}

#[derive(Serialize)]
struct SearchData {
    query: String,
    files: Vec<String>,
    total: usize,
    results: Vec<SearchResultNode>,
}

// Score a node against the query terms.
// Returns 0 if no term matches at all (node should be excluded).
fn score_node(node: &Node, terms: &[&str], search_body: bool) -> u32 {
    if terms.is_empty() {
        return 1;
    }

    let name_lower = node.node_data.name.to_lowercase();
    let file_lower = node.node_data.file.to_lowercase();
    let docs_lower = node.node_data.docs.as_deref().unwrap_or("").to_lowercase();
    let meta_values: Vec<String> = node
        .node_data
        .meta
        .values()
        .map(|v| v.to_lowercase())
        .collect();

    let mut score: u32 = 0;
    for term in terms {
        let t = term.to_lowercase();
        if name_lower == t {
            score += 100;
        } else if name_lower.starts_with(t.as_str()) || name_lower.ends_with(t.as_str()) {
            score += 70;
        } else if name_lower.contains(t.as_str()) {
            score += 50;
        }
        if docs_lower.contains(t.as_str()) {
            score += 25;
        }
        if meta_values.iter().any(|v| v.contains(t.as_str())) {
            score += 15;
        }
        if search_body && node.node_data.body.to_lowercase().contains(t.as_str()) {
            score += 20;
        }
        if file_lower.contains(t.as_str()) {
            score += 10;
        }
    }

    score
}

fn collect_callers(graph: &ArrayGraph, node: &Node) -> Vec<ContextRef> {
    let target_name = &node.node_data.name;
    let target_file = &node.node_data.file;
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut result = Vec::new();

    for edge in &graph.edges {
        if !matches!(edge.edge, EdgeType::Calls | EdgeType::Uses) {
            continue;
        }
        if &edge.target.node_data.name != target_name
            || &edge.target.node_data.file != target_file
        {
            continue;
        }
        let key = (
            edge.source.node_data.name.clone(),
            edge.source.node_data.file.clone(),
        );
        if !seen.insert(key) {
            continue;
        }
        if edge.source.node_data.file == "unverified" {
            continue;
        }
        let caller = graph.nodes.iter().find(|n| {
            n.node_data.name == edge.source.node_data.name
                && n.node_data.file == edge.source.node_data.file
        });
        let (node_type, line) = caller
            .map(|n| (n.node_type.to_string(), n.node_data.start + 1))
            .unwrap_or_default();
        result.push(ContextRef {
            name: edge.source.node_data.name.clone(),
            node_type,
            file: rel_path_from_cwd(&edge.source.node_data.file),
            line,
        });
    }

    result
}

fn collect_callees(graph: &ArrayGraph, node: &Node) -> Vec<ContextRef> {
    let source_name = &node.node_data.name;
    let source_file = &node.node_data.file;
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut result = Vec::new();

    for edge in &graph.edges {
        if !matches!(edge.edge, EdgeType::Calls | EdgeType::Uses) {
            continue;
        }
        if &edge.source.node_data.name != source_name
            || &edge.source.node_data.file != source_file
        {
            continue;
        }
        let key = (
            edge.target.node_data.name.clone(),
            edge.target.node_data.file.clone(),
        );
        if !seen.insert(key) {
            continue;
        }
        if edge.target.node_data.file == "unverified" {
            continue;
        }
        let callee = graph.nodes.iter().find(|n| {
            n.node_data.name == edge.target.node_data.name
                && n.node_data.file == edge.target.node_data.file
        });
        let (node_type, file, line) = callee
            .map(|n| {
                (
                    n.node_type.to_string(),
                    rel_path_from_cwd(&n.node_data.file),
                    n.node_data.start + 1,
                )
            })
            .unwrap_or_else(|| {
                (
                    String::new(),
                    edge.target.node_data.file.clone(),
                    edge.target.node_data.start + 1,
                )
            });
        result.push(ContextRef {
            name: edge.target.node_data.name.clone(),
            node_type,
            file,
            line,
        });
    }

    result
}

fn collect_tests(graph: &ArrayGraph, node: &Node) -> Vec<RelatedRef> {
    let name = &node.node_data.name;
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut result = Vec::new();

    for edge in &graph.edges {
        if !matches!(edge.edge, EdgeType::Calls | EdgeType::Uses) {
            continue;
        }
        if &edge.target.node_data.name != name {
            continue;
        }
        let test_node = graph.nodes.iter().find(|n| {
            TEST_TYPES.contains(&n.node_type)
                && n.node_data.name == edge.source.node_data.name
                && n.node_data.file == edge.source.node_data.file
        });
        if let Some(t) = test_node {
            let key = (t.node_data.name.clone(), t.node_data.file.clone());
            if seen.insert(key) {
                result.push(RelatedRef {
                    name: t.node_data.name.clone(),
                    node_type: t.node_type.to_string(),
                    file: rel_path_from_cwd(&t.node_data.file),
                    lines: format_lines(t.node_data.start, t.node_data.end),
                });
            }
        }
    }

    result
}

fn collect_related(graph: &ArrayGraph, node: &Node, limit: usize) -> Vec<RelatedRef> {
    let file = &node.node_data.file;
    let name = &node.node_data.name;
    let mut result: Vec<RelatedRef> = graph
        .nodes
        .iter()
        .filter(|n| {
            SEARCHABLE_TYPES.contains(&n.node_type)
                && &n.node_data.file == file
                && &n.node_data.name != name
        })
        .take(limit)
        .map(|n| RelatedRef {
            name: node_display_name(n),
            node_type: n.node_type.to_string(),
            file: rel_path_from_cwd(&n.node_data.file),
            lines: format_lines(n.node_data.start, n.node_data.end),
        })
        .collect();
    result.sort_by_key(|r| r.name.clone());
    result
}

pub async fn run(
    args: &SearchArgs,
    out: &mut Output,
    show_progress: bool,
    output_mode: OutputMode,
) -> Result<()> {
    let type_filter = if args.r#type.is_empty() {
        None
    } else {
        Some(parse_node_types(&args.r#type)?)
    };

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

    let graph = build_graph_for_files_with_options(&files, true).await?;

    if let Some(sp) = &spinner {
        sp.finish_and_clear();
    }

    let terms: Vec<&str> = args.query.split_whitespace().collect();

    let mut scored: Vec<(&Node, u32)> = graph
        .nodes
        .iter()
        .filter(|n| {
            let type_ok = if let Some(ref filter) = type_filter {
                filter.contains(&n.node_type)
            } else {
                SEARCHABLE_TYPES.contains(&n.node_type)
            };
            if !type_ok {
                return false;
            }
            if let Some(ref fp) = args.file {
                if !n.node_data.file.contains(fp.as_str()) {
                    return false;
                }
            }
            true
        })
        .filter_map(|n| {
            let s = score_node(n, &terms, args.body);
            if s > 0 {
                Some((n, s))
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.node_data.name.cmp(&b.0.node_data.name)));
    scored.truncate(args.limit);

    if output_mode.is_json() {
        let results: Vec<SearchResultNode> = scored
            .iter()
            .map(|(node, score)| {
                let nd = &node.node_data;
                SearchResultNode {
                    name: nd.name.clone(),
                    display_name: node_display_name(node),
                    node_type: node.node_type.to_string(),
                    file: rel_path_from_cwd(&nd.file),
                    lines: format_lines(nd.start, nd.end),
                    score: *score,
                    body: args.code.then(|| nd.body.clone()),
                    callers: if args.context {
                        collect_callers(&graph, node)
                    } else {
                        Vec::new()
                    },
                    callees: if args.context {
                        collect_callees(&graph, node)
                    } else {
                        Vec::new()
                    },
                    tests: if args.tests {
                        collect_tests(&graph, node)
                    } else {
                        Vec::new()
                    },
                    related: if args.related {
                        collect_related(&graph, node, 10)
                    } else {
                        Vec::new()
                    },
                }
            })
            .collect();

        let warnings = if results.is_empty() {
            vec![JsonWarning::new(
                "no_results",
                format!(
                    "No nodes matching '{}' found in the parsed files",
                    args.query
                ),
            )]
        } else {
            Vec::new()
        };

        write_json_success(
            out,
            "search",
            SearchData {
                query: args.query.clone(),
                files,
                total: results.len(),
                results,
            },
            warnings,
        )?;
        return Ok(());
    }

    // Human output
    if scored.is_empty() {
        out.writeln(format!(
            "{}",
            style(format!(
                "No nodes matching '{}' found in the parsed files.",
                args.query
            ))
            .yellow()
        ))?;
        return Ok(());
    }

    out.writeln(format!(
        "{} {} for '{}'",
        style(scored.len()).bold(),
        if scored.len() == 1 { "result" } else { "results" },
        style(&args.query).bold().white()
    ))?;
    out.newline()?;

    for (node, _score) in &scored {
        let nd = &node.node_data;
        let display_name = node_display_name(node);
        let lines = format_lines(nd.start, nd.end);
        let rel_file = rel_path_from_cwd(&nd.file);

        // Summary line: Type: name  [file:lines]
        let type_styled = style_for_node_type(&node.node_type).apply_to(&node.node_type);
        out.writeln(format!(
            "{}: {}  [{}:{}]",
            type_styled,
            style(&display_name).bold().white(),
            style(&rel_file).dim(),
            style(&lines).dim(),
        ))?;

        // Node-type-aware extras
        match node.node_type {
            NodeType::Endpoint => {
                if let Some(handler) = nd.meta.get("handler") {
                    out.writeln(format!(
                        "  {} {}",
                        style("Handler:").dim(),
                        style(handler).green()
                    ))?;
                }
            }
            NodeType::DataModel => {
                if let Some(interface) = nd.meta.get("interface") {
                    let preview: Vec<&str> = interface.lines().take(3).collect();
                    let fence = style("```").dim();
                    out.writeln(format!("  {}", fence))?;
                    for line in &preview {
                        out.writeln(format!("  {}", line))?;
                    }
                    out.writeln(format!("  {}", fence))?;
                }
            }
            _ => {}
        }

        // --code: full source body
        if args.code && !nd.body.is_empty() {
            let fence = style("```").dim();
            out.writeln(format!("  {}", fence))?;
            for line in nd.body.lines() {
                out.writeln(format!("  {}", line))?;
            }
            out.writeln(format!("  {}", fence))?;
        }

        // --context: immediate callers and callees
        if args.context {
            let callers = collect_callers(&graph, node);
            let callees = collect_callees(&graph, node);
            for cr in &callers {
                out.writeln(format!(
                    "  {} {}  {}  [{}:{}]",
                    style("←").dim(),
                    style(&cr.name).white(),
                    style(format!("[{}]", cr.node_type)).cyan().dim(),
                    style(&cr.file).dim(),
                    style(cr.line).dim(),
                ))?;
            }
            for cr in &callees {
                out.writeln(format!(
                    "  {} {}  {}  [{}:{}]",
                    style("→").dim(),
                    style(&cr.name).white(),
                    style(format!("[{}]", cr.node_type)).cyan().dim(),
                    style(&cr.file).dim(),
                    style(cr.line).dim(),
                ))?;
            }
        }

        // --tests: associated test nodes that call this node
        if args.tests {
            let tests = collect_tests(&graph, node);
            if !tests.is_empty() {
                out.writeln(format!("  {}:", style("Tests").dim()))?;
                for t in &tests {
                    out.writeln(format!(
                        "    {}  [{}:{}]",
                        style(&t.name).white(),
                        style(&t.file).dim(),
                        style(&t.lines).dim(),
                    ))?;
                }
            }
        }

        // --related: sibling nodes in the same file
        if args.related {
            let related = collect_related(&graph, node, 5);
            if !related.is_empty() {
                out.writeln(format!("  {}:", style("Related").dim()))?;
                for r in &related {
                    out.writeln(format!(
                        "    {}  {}  [{}]",
                        style(format!("[{}]", r.node_type)).cyan().dim(),
                        style(&r.name).white(),
                        style(&r.lines).dim(),
                    ))?;
                }
            }
        }

        out.newline()?;
    }

    Ok(())
}
