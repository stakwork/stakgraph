#![cfg(feature = "neo4j")]
use crate::lang::graphs::{neo4j_utils::*, Neo4jGraph, BTreeMapGraph, ArrayGraph};
use crate::lang::graphs::{Graph, EdgeType};
use crate::utils::{create_node_key, create_node_key_from_ref};
use neo4rs::BoltMap;
use std::collections::HashSet;
use shared::Result;
use tracing::{info, debug};

pub struct GraphStreamingUploader {
    uploaded_nodes: HashSet<String>,
    uploaded_edges: HashSet<(String, String, EdgeType)>,
}

impl GraphStreamingUploader {
    pub fn new() -> Self {
        Self { uploaded_nodes: HashSet::new(), uploaded_edges: HashSet::new() }
    }

    pub async fn flush_stage<G: Graph + std::fmt::Debug + 'static>(
        &mut self,
        graph: &G,
        neo: &Neo4jGraph,
        stage: &str,
    ) -> Result<()> {
        let mut node_queries: Vec<(String, BoltMap)> = Vec::new();
        let mut new_nodes_cnt = 0usize;
        if let Some(bt) = (graph as &dyn std::any::Any).downcast_ref::<BTreeMapGraph>() {
            for (k, n) in bt.nodes.iter() {
                if self.uploaded_nodes.insert(k.clone()) {
                    node_queries.push(add_node_query(&n.node_type, &n.node_data));
                    new_nodes_cnt += 1;
                }
            }
        } else if let Some(arr) = (graph as &dyn std::any::Any).downcast_ref::<ArrayGraph>() {
            for n in arr.nodes.iter() {
                let k = create_node_key(n);
                if self.uploaded_nodes.insert(k) {
                    node_queries.push(add_node_query(&n.node_type, &n.node_data));
                    new_nodes_cnt += 1;
                }
            }
        }

        if new_nodes_cnt > 0 {
            debug!(stage = stage, count = new_nodes_cnt, "stream_upload_nodes");
            neo.execute_batch(node_queries).await?;
        }

        let mut edge_specs: Vec<(String, String, EdgeType)> = Vec::new();
        let mut new_edges_cnt = 0usize;
        if let Some(bt) = (graph as &dyn std::any::Any).downcast_ref::<BTreeMapGraph>() {
            for (s, t, et) in bt.edges.iter() {
                let triple = (s.clone(), t.clone(), et.clone());
                if self.uploaded_edges.insert(triple.clone()) {
                    edge_specs.push(triple);
                    new_edges_cnt += 1;
                }
            }
        } else if let Some(arr) = (graph as &dyn std::any::Any).downcast_ref::<ArrayGraph>() {
            for e in arr.edges.iter() {
                let s = create_node_key_from_ref(&e.source);
                let t = create_node_key_from_ref(&e.target);
                let triple = (s, t, e.edge.clone());
                if self.uploaded_edges.insert(triple.clone()) {
                    edge_specs.push(triple);
                    new_edges_cnt += 1;
                }
            }
        }

        if new_edges_cnt > 0 {
            let edge_queries = build_batch_edge_queries(edge_specs.into_iter(), 256);
            debug!(stage = stage, count = new_edges_cnt, "stream_upload_edges");
            neo.execute_simple(edge_queries).await?;
        }

        if new_nodes_cnt > 0 || new_edges_cnt > 0 {
            info!(stage = stage, nodes = new_nodes_cnt, edges = new_edges_cnt, "stream_stage_flush");
        }
        Ok(())
    }
}

pub struct StreamingUploadContext {
    pub neo: Neo4jGraph,
    pub uploader: GraphStreamingUploader,
}

impl StreamingUploadContext {
    pub fn new(neo: Neo4jGraph) -> Self { Self { neo, uploader: GraphStreamingUploader::new() } }
}
