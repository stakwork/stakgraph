use ast::lang::asg::NodeData;
use ast::lang::NodeType;
use shared::Result;
use std::str::FromStr;

use crate::types::{Node, NodeConcise, NodesResponse, NodesResponseItem};

pub fn parse_node_type(node_type: &str) -> Result<NodeType> {
    let mut chars: Vec<char> = node_type.chars().collect();
    if !chars.is_empty() {
        chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
    }
    let titled_case = chars.into_iter().collect::<String>();
    NodeType::from_str(&titled_case)
}

pub fn extract_ref_id(node_data: &NodeData) -> String {
    node_data
        .meta
        .get("ref_id")
        .cloned()
        .unwrap_or_else(|| "placeholder".to_string())
}

pub fn create_uncovered_response_items(
    nodes: Vec<(NodeData, usize)>,
    node_type: &NodeType,
    concise: bool,
) -> Vec<NodesResponseItem> {
    let nodes_with_coverage: Vec<(NodeData, usize, bool, usize)> = nodes
        .into_iter()
        .map(|(node_data, weight)| (node_data, weight, false, 0))
        .collect();
    create_nodes_response_items(nodes_with_coverage, node_type, concise)
}

pub fn format_uncovered_response_as_snippet(response: &NodesResponse) -> String {
    format_nodes_response_as_snippet(response)
}

pub fn create_nodes_response_items(
    nodes: Vec<(NodeData, usize, bool, usize)>,
    node_type: &NodeType,
    concise: bool,
) -> Vec<NodesResponseItem> {
    nodes
        .into_iter()
        .map(|(node_data, weight, covered, test_count)| {
            if concise {
                NodesResponseItem::Concise(NodeConcise {
                    name: node_data.clone().name,
                    file: node_data.clone().file,
                    ref_id: extract_ref_id(&node_data),
                    weight,
                    test_count,
                    covered,
                })
            } else {
                let ref_id = extract_ref_id(&node_data);
                NodesResponseItem::Full(Node {
                    node_type: node_type.to_string(),
                    ref_id,
                    weight,
                    test_count,
                    covered,
                    properties: node_data,
                })
            }
        })
        .collect()
}

pub fn format_nodes_response_as_snippet(response: &NodesResponse) -> String {
    let mut text = String::new();

    if let Some(ref functions) = response.functions {
        for item in functions {
            match item {
                NodesResponseItem::Full(node) => {
                    let coverage_indicator = if node.covered {
                        "[COVERED]"
                    } else {
                        "[UNCOVERED]"
                    };
                    text.push_str(&format!(
                        "<snippet>\nname: {}: {} {}\nref_id: {}\nweight: {}\nfile: {}\nstart: {}, end: {}\n\n{}\n</snippet>\n\n",
                        &node.node_type,
                        &node.properties.name,
                        coverage_indicator,
                        &node.ref_id,
                        node.weight,
                        &node.properties.file,
                        node.properties.start,
                        node.properties.end,
                        &node.properties.body,
                    ));
                }
                NodesResponseItem::Concise(node) => {
                    let coverage_indicator = if node.covered {
                        "[COVERED]"
                    } else {
                        "[UNCOVERED]"
                    };
                    text.push_str(&format!(
                        "Function: {} {} (weight: {})\nFile: {}\n\n",
                        &node.name, coverage_indicator, node.weight, &node.file,
                    ));
                }
            }
        }
    }

    if let Some(ref endpoints) = response.endpoints {
        for item in endpoints {
            match item {
                NodesResponseItem::Full(node) => {
                    let coverage_indicator = if node.covered {
                        "[COVERED]"
                    } else {
                        "[UNCOVERED]"
                    };
                    text.push_str(&format!(
                        "<snippet>\nname: {}: {} {}\nref_id: {}\nweight: {}\nfile: {}\nstart: {}, end: {}\n\n{}\n</snippet>\n\n",
                        &node.node_type,
                        &node.properties.name,
                        coverage_indicator,
                        &node.ref_id,
                        node.weight,
                        &node.properties.file,
                        node.properties.start,
                        node.properties.end,
                        &node.properties.body,
                    ));
                }
                NodesResponseItem::Concise(node) => {
                    let coverage_indicator = if node.covered {
                        "[COVERED]"
                    } else {
                        "[UNCOVERED]"
                    };
                    text.push_str(&format!(
                        "Endpoint: {} {} (weight: {})\nFile: {}\n\n",
                        &node.name, coverage_indicator, node.weight, &node.file,
                    ));
                }
            }
        }
    }

    text
}
