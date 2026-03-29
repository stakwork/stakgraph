#![cfg(feature = "neo4j")]
use std::collections::BTreeSet;
use std::time::Instant;

use crate::builder::utils::log_stage_timing;
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
            let start = Instant::now();
            neo.execute_batch(delta_node_queries.to_vec()).await?;
            log_stage_timing("neo4j_flush_nodes", start, Some(&format!("stage={} nodes={}", stage, node_cnt)));
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

        let start = Instant::now();
        neo.execute_simple(edge_queries).await?;
        log_stage_timing("neo4j_flush_edges", start, Some(&format!("stage={} edges={}", stage, edges.len())));

        info!(stage = stage, edges = edges.len(), "stream_edges_flushed");
        Ok(())
    }
}

pub struct StreamingUploadContext {
    pub neo: Neo4jGraph,
    pub uploader: GraphStreamingUploader,
    pub flushed_node_count: usize,
    pub flushed_edge_count: usize,
}

impl StreamingUploadContext {
    pub fn new(neo: Neo4jGraph) -> Self {
        Self {
            neo,
            uploader: GraphStreamingUploader::new(),
            flushed_node_count: 0,
            flushed_edge_count: 0,
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
    let all_nodes: Vec<_> = graph.iter_all_nodes().collect();
    let new_nodes = &all_nodes[ctx.flushed_node_count..];
    if !new_nodes.is_empty() {
        let bolt_nodes = nodes_to_bolt_format(new_nodes.iter().copied());
        ctx.uploader.flush_stage(&ctx.neo, stage, &bolt_nodes).await?;
        ctx.flushed_node_count = all_nodes.len();
    }
    Ok(())
}

pub async fn flush_stage_nodes_and_edges<G: Graph>(
    ctx: &mut StreamingUploadContext,
    graph: &G,
    stage: &str,
) -> Result<()> {
    let all_nodes: Vec<_> = graph.iter_all_nodes().collect();
    let new_nodes = &all_nodes[ctx.flushed_node_count..];
    if !new_nodes.is_empty() {
        let bolt_nodes = nodes_to_bolt_format(new_nodes.iter().copied());
        ctx.uploader.flush_stage(&ctx.neo, stage, &bolt_nodes).await?;
        ctx.flushed_node_count = all_nodes.len();
    }
    let all_edges = graph.get_edge_keys();
    let new_edges: BTreeSet<_> = all_edges.iter().skip(ctx.flushed_edge_count).cloned().collect();
    if !new_edges.is_empty() {
        ctx.uploader.flush_edges_stage(&ctx.neo, stage, &new_edges).await?;
        ctx.flushed_edge_count = all_edges.len();
    }
    Ok(())
}
