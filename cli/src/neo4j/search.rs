use std::str::FromStr;

use ast::lang::graphs::{Node, NodeType};
use ast::lang::NodeData;
use console::style;
use neo4rs::{query as nq, BoltType};
use shared::Result;

use crate::output::Output;

use super::connection::connect_graph_ops;

pub(super) fn node_data_from_neo_node(neo_node: &neo4rs::Node) -> Option<Node> {
    let node_data = NodeData::try_from(neo_node).ok()?;
    let labels = neo_node.labels();
    let label = labels.iter().copied().find(|l| *l != "Data_Bank")?;
    let node_type = NodeType::from_str(label).ok()?;
    Some(Node { node_type, node_data })
}

pub(super) fn escape_fulltext_query(query: &str) -> String {
    if query.contains(' ') {
        let escaped = query.replace('"', "\\\"");
        return format!("\"{}\"", escaped);
    }
    let special = [
        '+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '?', ':',
        '\\', '/',
    ];
    let mut result = query.to_string();
    for ch in special {
        result = result.replace(ch, &format!("\\{}", ch));
    }
    format!("{}*", result)
}

pub(super) async fn run_search(
    query: &str,
    node_type_strs: &[String],
    limit: usize,
    out: &mut Output,
) -> Result<()> {
    let ops = connect_graph_ops().await?;
    let connection = ops.graph.ensure_connected().await?;

    let escaped = escape_fulltext_query(query);

    // Skip test nodes and imports (same default as MCP)
    let skip_types = vec!["UnitTest", "IntegrationTest", "E2etest", "Import"];

    let node_type_labels: Vec<String> = if node_type_strs.is_empty() {
        vec![]
    } else {
        node_type_strs.to_vec()
    };

    let cypher = r#"
        CALL db.index.fulltext.queryNodes('nameBodyFileIndex', $query) YIELD node, score
        WITH node, score
        WHERE
        CASE
            WHEN size($node_types) = 0 THEN true
            ELSE ANY(label IN labels(node) WHERE label IN $node_types)
        END
        AND NOT ANY(label IN labels(node) WHERE label IN $skip_types)
        RETURN node, score
        ORDER BY score DESC
        LIMIT toInteger($limit)
"#;

    let node_types_bolt: Vec<BoltType> = node_type_labels
        .iter()
        .map(|s: &String| BoltType::String(s.clone().into()))
        .collect();
    let skip_bolt: Vec<BoltType> = skip_types
        .iter()
        .map(|s: &&str| BoltType::String((*s).into()))
        .collect();

    let q = nq(cypher)
        .param("query", escaped)
        .param("node_types", node_types_bolt.as_slice())
        .param("skip_types", skip_bolt.as_slice())
        .param("limit", limit as i64);

    let mut result = connection.execute(q).await?;
    let mut count = 0;

    while let Some(row) = result.next().await? {
        let neo_node: neo4rs::Node = match row.get("node") {
            Ok(n) => n,
            Err(_) => continue,
        };
        let score: f64 = row.get("score").unwrap_or(0.0);

        if let Some(graph_node) = node_data_from_neo_node(&neo_node) {
            let score_label = style(format!("{:.3}", score)).dim();
            out.writeln(score_label.to_string())?;
            crate::render::print_node_summary(out, &graph_node)
                .map_err(|e| shared::Error::Io(e))?;
            out.newline()?;
        }
        count += 1;
    }

    if count == 0 {
        out.writeln(format!("No results for {:?}", query))?;
    } else {
        out.writeln(format!("\n{} result(s)", count))?;
    }
    Ok(())
}
