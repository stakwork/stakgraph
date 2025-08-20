use super::neo4j_utils::*;
use super::*;
use crate::lang::graphs::graph_ops::GraphOps;
use neo4rs::BoltMap;
use shared::error::Result;

pub struct GraphUploader {
    pub graph_ops: GraphOps,
}

impl GraphUploader {
    pub async fn new() -> Result<Self> {
        let mut graph_ops = GraphOps::new();
        graph_ops.connect().await?;
        Ok(Self { graph_ops })
    }

    pub async fn upload_nodes_by_type(
        &mut self,
        btree_graph: &BTreeMapGraph,
        node_type: NodeType,
    ) -> Result<()> {
        let nodes = btree_graph.find_nodes_by_type(node_type.clone());
        if nodes.is_empty() {
            return Ok(());
        }

        let queries: Vec<(String, BoltMap)> = nodes
            .iter()
            .map(|node| add_node_query(&node_type, node))
            .collect();

        self.graph_ops.graph.execute_batch(queries).await
    }

    pub async fn upload_edges_between_types(
        &mut self,
        btree_graph: &BTreeMapGraph,
        source_type: NodeType,
        target_type: NodeType,
        edge_type: EdgeType,
    ) -> Result<()> {
        let edges = btree_graph.find_nodes_with_edge_type(
            source_type.clone(),
            target_type.clone(),
            edge_type.clone(),
        );
        if edges.is_empty() {
            return Ok(());
        }

        let queries: Vec<(String, BoltMap)> = edges
            .iter()
            .map(|(source, target)| {
                let edge = Edge::new(
                    edge_type.clone(),
                    NodeRef::from(source.into(), source_type.clone()),
                    NodeRef::from(target.into(), target_type.clone()),
                );
                add_edge_query(&edge)
            })
            .collect();

        self.graph_ops.graph.execute_simple(queries).await
    }
}

pub fn tests_sources(tests_filter: Option<&str>) -> Vec<NodeType> {
    let unit = tests_filter
        .map(|s| s.eq_ignore_ascii_case("unit"))
        .unwrap_or(false);
    let e2e = tests_filter
        .map(|s| s.eq_ignore_ascii_case("e2e"))
        .unwrap_or(false);
    let both = tests_filter.is_none()
        || tests_filter
            .map(|s| s.eq_ignore_ascii_case("both"))
            .unwrap_or(false);
    if both || (!unit && !e2e) {
        return vec![NodeType::Test, NodeType::E2eTest];
    }
    let mut sources = Vec::new();
    if unit {
        sources.push(NodeType::Test);
    }
    if e2e {
        sources.push(NodeType::E2eTest);
    }
    sources
}
