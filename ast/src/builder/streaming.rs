#![cfg(feature = "neo4j")]
use std::collections::BTreeSet;

use crate::lang::graphs::{neo4j::*, Graph, Neo4jGraph};
use crate::lang::{EdgeType, NodeData, NodeType};
use neo4rs::BoltMap;
use shared::Result;
use tracing::{debug, info};
use uuid::Uuid;

pub struct GraphStreamingUploader {}

impl GraphStreamingUploader {
    pub fn new() -> Self {
        Self {}
    }

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

pub fn nodes_to_bolt_format<'a>(
    nodes: impl Iterator<Item = (&'a NodeType, &'a NodeData)>,
) -> Vec<(String, BoltMap)> {
    nodes
        .map(|(nt, nd)| add_node_query_stream(nt, nd))
        .collect()
}

pub async fn flush_stage_nodes<G: Graph>(
    ctx: &mut StreamingUploadContext,
    graph: &G,
    stage: &str,
) -> Result<()> {
    let bolt_nodes = nodes_to_bolt_format(graph.iter_all_nodes());
    ctx.uploader
        .flush_stage(&ctx.neo, stage, &bolt_nodes)
        .await?;
    Ok(())
}

pub async fn flush_stage_nodes_and_edges<G: Graph>(
    ctx: &mut StreamingUploadContext,
    graph: &G,
    stage: &str,
) -> Result<()> {
    let bolt_nodes = nodes_to_bolt_format(graph.iter_all_nodes());
    ctx.uploader
        .flush_stage(&ctx.neo, stage, &bolt_nodes)
        .await?;
    let edges = graph.get_edge_keys();
    ctx.uploader
        .flush_edges_stage(&ctx.neo, stage, &edges)
        .await?;
    Ok(())
}
