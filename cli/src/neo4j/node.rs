use ast::lang::graphs::neo4j::{execute_node_query, find_node_by_name_file_query, find_nodes_by_name_query};
use ast::lang::graphs::NodeType;
use shared::Result;
use std::str::FromStr;

use crate::output::Output;

use super::connection::connect_graph_ops;
use super::ALL_NODE_TYPES;

pub(super) async fn run_node(
    name: &str,
    node_type: Option<&str>,
    file: Option<&str>,
    out: &mut Output,
) -> Result<()> {
    let ops = connect_graph_ops().await?;
    let connection = ops.graph.ensure_connected().await?;

    let types_to_query: Vec<NodeType> = if let Some(nt_str) = node_type {
        match NodeType::from_str(nt_str) {
            Ok(nt) => vec![nt],
            Err(_) => {
                out.writeln(format!("Unknown node type: {:?}", nt_str))?;
                return Ok(());
            }
        }
    } else {
        ALL_NODE_TYPES.to_vec()
    };

    let mut found: Vec<(NodeType, ast::lang::NodeData)> = Vec::new();
    for nt in &types_to_query {
        let nodes = if let Some(f) = file {
            let (q, params) = find_node_by_name_file_query(nt, name, f);
            execute_node_query(&connection, q, params).await
        } else {
            let (q, params) = find_nodes_by_name_query(nt, name, "");
            execute_node_query(&connection, q, params).await
        };
        for node in nodes {
            found.push((nt.clone(), node));
        }
    }

    if found.is_empty() {
        out.writeln(format!("No node found with name {:?}", name))?;
        return Ok(());
    }

    for (node_type, node_data) in found {
        let graph_node = ast::lang::graphs::Node { node_type, node_data };
        crate::render::print_node_summary(out, &graph_node)
            .map_err(|e| shared::Error::Io(e))?;
        out.newline()?;
    }
    Ok(())
}
