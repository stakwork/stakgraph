#![cfg(feature = "neo4j")]
use crate::lang::graphs::{neo4j_utils::*, Neo4jGraph};
use crate::lang::{Edge, NodeData, NodeType};
use neo4rs::BoltMap;
use shared::Result;
use tracing::{debug, info};

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
            info!(
                stage = stage,
                nodes = node_cnt,
                "stream_stage_flush"
            );
        }
        Ok(())
    }

    pub async fn flush_edges(
        &mut self,
        neo: &Neo4jGraph,
        edges: &[Edge],
    ) -> Result<()> {
        if edges.is_empty() {
            return Ok(());
        }
        
        let edge_queries: Vec<(String, BoltMap)> = edges
            .iter()
            .map(|e| add_edge_query(e))
            .collect();
        
        info!(count = edges.len(), "bulk_upload_edges");
        neo.execute_batch(edge_queries).await?;
        info!(edges = edges.len(), "bulk_edges_uploaded");
        
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

use lazy_static::lazy_static;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

lazy_static! {
    static ref DELTA_NODES: Mutex<Vec<(String, BoltMap)>> = Mutex::new(Vec::new());
}

static STREAM_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enable_streaming() {
    STREAM_ENABLED.store(true, Ordering::Relaxed);
}

pub fn disable_streaming() {
    STREAM_ENABLED.store(false, Ordering::Relaxed);
}

pub fn is_streaming_enabled() -> bool {
    STREAM_ENABLED.load(Ordering::Relaxed)
}

pub fn record_node(nt: &NodeType, nd: &NodeData) {
    if !is_streaming_enabled() {
        return;
    }
    if let Ok(mut g) = DELTA_NODES.lock() {
        g.push(add_node_query(nt, nd));
    }
}

pub fn drain_deltas() -> Vec<(String, BoltMap)> {
    let mut n = DELTA_NODES.lock().unwrap();
    std::mem::take(&mut *n)
}
