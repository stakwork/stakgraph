#![cfg(feature = "neo4j")]
use std::collections::BTreeSet;

use crate::lang::graphs::{neo4j::*, Neo4jGraph};
use crate::lang::{EdgeType, NodeData, NodeType};
use neo4rs::BoltMap;
use shared::Result;
use tracing::{debug, info, instrument};
use uuid::Uuid;

pub struct GraphStreamingUploader {}

impl GraphStreamingUploader {
    pub fn new() -> Self {
        Self {}
    }
#[instrument(skip(self, neo, delta_node_queries), fields(stage, node_count = delta_node_queries.len()))]
    pub async fn flush_stage(
        &mut self,
        neo: &Neo4jGraph,
        stage: &str,
        delta_node_queries: &[(String, BoltMap)],
    ) -> Result<()> {
        let node_cnt = delta_node_queries.len();
        if node_cnt > 0 {
            debug!(stage = stage, count = node_cnt, "stream_upload_nodes");
            neo.execute_batch(delta_node_queries.to_vec()).await?;
            info!(stage = stage, nodes = node_cnt, "stream_stage_flush");
        }
        Ok(())
    }

    pub async fn flush_edges_stage(
        &mut self,
        neo: &Neo4jGraph,
        stage: &str,
        edges: &BTreeSet<(String, String, EdgeType)>,
    ) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }

        debug!(stage = stage, count = edges.len(), "stream_upload_edges");

        let edge_queries = build_batch_edge_queries_stream(
            edges.iter().map(|e| {
                (
                    e.0.clone(),
                    e.1.clone(),
                    e.2.clone(),
                    Uuid::new_v4().to_string(),
                )
            }),
            256,
        );

        neo.execute_simple(edge_queries).await?;

        info!(stage = stage, edges = edges.len(), "stream_edges_flushed");
        Ok(())
    }
}

pub struct StreamingUploadContext {
    pub neo: Neo4jGraph,
    pub uploader: GraphStreamingUploader,
}

impl StreamingUploadContext {
    pub fn new(neo: Neo4jGraph) -> Self {
        Self {
            neo,
            uploader: GraphStreamingUploader::new(),
        }
    }
}

pub fn nodes_to_bolt_format(nodes: Vec<(NodeType, NodeData)>) -> Vec<(String, BoltMap)> {
    nodes
        .iter()
        .map(|(nt, nd)| add_node_query_stream(nt, nd))
        .collect()
}
