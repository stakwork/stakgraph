#![cfg(feature = "neo4j")]
use crate::lang::graphs::EdgeType;
use crate::lang::graphs::{neo4j_utils::*, Neo4jGraph};
use crate::lang::{Edge, NodeData, NodeType};
use neo4rs::BoltMap;
use shared::Result;
use std::collections::HashSet;
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
        delta_edges: &[Edge],
    ) -> Result<()> {
        let node_cnt = delta_node_queries.len();
        if node_cnt > 0 {
            debug!(stage = stage, count = node_cnt, "stream_upload_nodes");
            neo.execute_batch(delta_node_queries.to_vec()).await?;
        }

        const EDGE_BATCH_SIZE: usize = 1024;
        let mut edge_specs: Vec<(String, String, EdgeType)> = Vec::with_capacity(delta_edges.len());
        if !delta_edges.is_empty() {
            let mut seen: HashSet<(String, String, EdgeType)> =
                HashSet::with_capacity(delta_edges.len() * 2);
            for e in delta_edges.iter() {
                let s = crate::utils::create_node_key_from_ref(&e.source);
                let t = crate::utils::create_node_key_from_ref(&e.target);
                let key = (s, t, e.edge.clone());
                if seen.insert(key.clone()) {
                    edge_specs.push(key);
                }
            }
            let edge_cnt = edge_specs.len();
            if edge_cnt > 0 {
                let edge_queries =
                    build_batch_edge_queries(edge_specs.into_iter(), EDGE_BATCH_SIZE);
                debug!(stage = stage, count = edge_cnt, "stream_upload_edges");
                neo.execute_simple(edge_queries).await?;
            }
            if node_cnt > 0 || edge_cnt > 0 {
                info!(
                    stage = stage,
                    nodes = node_cnt,
                    edges = edge_cnt,
                    "stream_stage_flush"
                );
            }
            return Ok(());
        }
        if node_cnt > 0 {
            info!(
                stage = stage,
                nodes = node_cnt,
                edges = 0,
                "stream_stage_flush"
            );
        }
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
use std::sync::Mutex;

lazy_static! {
    static ref STREAM_ENABLED: bool = std::env::var("STREAM_UPLOAD").is_ok();
    static ref DELTA_NODES: Mutex<Vec<(String, BoltMap)>> = Mutex::new(Vec::new());
    static ref DELTA_EDGES: Mutex<Vec<Edge>> = Mutex::new(Vec::new());
}

pub fn record_node(nt: &NodeType, nd: &NodeData) {
    if !*STREAM_ENABLED {
        return;
    }
    if let Ok(mut g) = DELTA_NODES.lock() {
        g.push(add_node_query(nt, nd));
    }
}
pub fn record_edge(e: &Edge) {
    if !*STREAM_ENABLED {
        return;
    }
    if let Ok(mut g) = DELTA_EDGES.lock() {
        g.push(e.clone());
    }
}
pub fn drain_deltas() -> (Vec<(String, BoltMap)>, Vec<Edge>) {
    let mut n = DELTA_NODES.lock().unwrap();
    let mut e = DELTA_EDGES.lock().unwrap();
    (std::mem::take(&mut *n), std::mem::take(&mut *e))
}
