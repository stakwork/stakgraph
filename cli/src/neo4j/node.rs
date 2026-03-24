use ast::lang::graphs::neo4j::execute_node_query;
use ast::lang::graphs::NodeType;
use shared::Result;

use crate::output::Output;

use super::connection::connect_graph_ops;
use super::ALL_NODE_TYPES;

pub(super) async fn run_node(name: &str, out: &mut Output) -> Result<()> {
    let ops = connect_graph_ops().await?;
    let connection = ops.graph.ensure_connected().await?;

    let mut found: Vec<(NodeType, ast::lang::NodeData)> = Vec::new();
    for nt in ALL_NODE_TYPES {
        use ast::lang::graphs::neo4j::find_nodes_by_name_query;
        let (q, params) = find_nodes_by_name_query(nt, name, "");
        let nodes = execute_node_query(&connection, q, params).await;
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
