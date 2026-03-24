use ast::lang::graphs::graph_ops::GraphOps;
use ast::lang::graphs::neo4j::{boltmap_insert_str, execute_count_query};
use ast::lang::graphs::NodeType;
use neo4rs::BoltMap;

use crate::output::Output;

pub async fn print_caller_counts(
    out: &mut Output,
    graph: &ast::lang::graphs::ArrayGraph,
    files_to_print: &[String],
) {
    let mut ops = GraphOps::new();
    if ops.connect().await.is_err() {
        eprintln!("warning: neo4j unavailable, skipping caller count annotations");
        return;
    }
    let connection = match ops.graph.ensure_connected().await {
        Ok(c) => c,
        Err(_) => {
            eprintln!("warning: neo4j unavailable, skipping caller count annotations");
            return;
        }
    };

    let functions: Vec<&ast::lang::NodeData> = graph
        .nodes
        .iter()
        .filter(|n| {
            n.node_type == NodeType::Function
                && files_to_print.iter().any(|f| *f == n.node_data.file)
        })
        .map(|n| &n.node_data)
        .collect();

    if functions.is_empty() {
        return;
    }

    out.writeln(
        console::style("\n--- Caller counts (from graph) ---")
            .bold()
            .to_string(),
    )
    .ok();

    for func in functions {
        let query =
            "MATCH (caller:Function)-[:CALLS]->(target:Function {name: $name, file: $file}) RETURN count(caller) AS cnt"
                .to_string();
        let mut params = BoltMap::new();
        boltmap_insert_str(&mut params, "name", &func.name);
        boltmap_insert_str(&mut params, "file", &func.file);

        let count = execute_count_query(&connection, query, params).await;
        if count > 0 {
            out.writeln(format!(
                "  {:>4} caller(s)  {}  [{}]",
                count,
                console::style(&func.name).bold(),
                func.file
            ))
            .ok();
        }
    }
}
