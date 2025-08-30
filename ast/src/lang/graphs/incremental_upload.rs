use std::collections::{HashMap, HashSet};
use tokio::sync::mpsc;
use crate::lang::{NodeType, NodeData};
use super::{neo4j_graph::Neo4jGraph, neo4j_utils::{add_node_query, build_batch_edge_queries}};
use crate::utils::{create_node_key, create_node_key_from_ref};
use crate::lang::Edge;
use tracing::{error};

const NODE_THRESHOLD: usize = 256;
const EDGE_BATCH_INTERNAL: usize = 512;
const EDGE_TX_CHUNK: usize = 128;

pub enum UploadEvent {
    Node(NodeType, NodeData),
    Edge(Edge),
    Stop,
}

#[derive(Debug, Clone)]
pub struct IncrementalUploadHandle(pub mpsc::UnboundedSender<UploadEvent>);

impl PartialEq for IncrementalUploadHandle { fn eq(&self, _other:&Self)->bool { true } }
impl Eq for IncrementalUploadHandle {}

impl IncrementalUploadHandle { pub fn new(graph: Neo4jGraph) -> Self {
        let (tx, mut rx) = mpsc::unbounded_channel::<UploadEvent>();
        tokio::spawn(async move {
            let mut neo = graph;
            let mut node_buf: Vec<(NodeType, NodeData, String)> = Vec::with_capacity(NODE_THRESHOLD);
            let mut uploaded_nodes: HashSet<String> = HashSet::new();
            let mut waiting: HashMap<String, Vec<Edge>> = HashMap::new();
            let mut edge_buf: Vec<Edge> = Vec::new();
            while let Some(ev) = rx.recv().await {
                match ev {
                    UploadEvent::Node(t,d) => {
                        let key = create_node_key(&crate::lang::Node::new(t.clone(), d.clone()));
                        node_buf.push((t,d,key));
                        if node_buf.len() >= NODE_THRESHOLD { if let Err(e)=flush_nodes(&mut neo, &mut node_buf, &mut uploaded_nodes, &mut waiting, &mut edge_buf).await { error!("flush_nodes error: {:?}", e); } }
                    }
                    UploadEvent::Edge(e) => {
                        let s = create_node_key_from_ref(&e.source);
                        let t = create_node_key_from_ref(&e.target);
                        if uploaded_nodes.contains(&s) && uploaded_nodes.contains(&t) { edge_buf.push(e); } else {
                            let missing = if !uploaded_nodes.contains(&s) { s } else { t };
                            waiting.entry(missing).or_default().push(e);
                        }
                        if edge_buf.len() >= EDGE_BATCH_INTERNAL { if let Err(e)=flush_edges(&mut neo, &mut edge_buf).await { error!("flush_edges error: {:?}", e); } }
                    }
                    UploadEvent::Stop => {
                        let _ = flush_nodes(&mut neo, &mut node_buf, &mut uploaded_nodes, &mut waiting, &mut edge_buf).await;
                        let _ = flush_edges(&mut neo, &mut edge_buf).await;
                        break;
                    }
                }
                promote_waiting(&uploaded_nodes, &mut waiting, &mut edge_buf);
            }
        });
        IncrementalUploadHandle(tx)
    } pub fn stop(&self) { let _ = self.0.send(UploadEvent::Stop); } }

async fn flush_nodes(neo: &mut Neo4jGraph, buf: &mut Vec<(NodeType,NodeData,String)>, uploaded: &mut HashSet<String>, waiting: &mut HashMap<String,Vec<Edge>>, edge_buf: &mut Vec<Edge>) -> shared::error::Result<()> {
    if buf.is_empty() { return Ok(()); }
    neo.ensure_connected().await?;
    let queries = buf.iter().map(|(t,d,_)| add_node_query(t,d)).collect();
    neo.execute_batch(queries).await?;
    for (_,_,k) in buf.drain(..) { uploaded.insert(k); }
    promote_waiting(uploaded, waiting, edge_buf);
    Ok(())
}

fn promote_waiting(uploaded: &HashSet<String>, waiting: &mut HashMap<String,Vec<Edge>>, edge_buf: &mut Vec<Edge>) {
    let ready: Vec<String> = waiting.keys().filter(|k| uploaded.contains(*k)).cloned().collect();
    for k in ready { if let Some(edges)=waiting.remove(&k) { for e in edges { let s=create_node_key_from_ref(&e.source); let t=create_node_key_from_ref(&e.target); if uploaded.contains(&s) && uploaded.contains(&t) { edge_buf.push(e); } else { let missing = if !uploaded.contains(&s) { s } else { t }; waiting.entry(missing).or_default().push(e); } } } }
}

async fn flush_edges(neo: &mut Neo4jGraph, buf: &mut Vec<Edge>) -> shared::error::Result<()> {
    if buf.is_empty() { return Ok(()); }
    neo.ensure_connected().await?;
    let tuples = buf.drain(..).map(|e| {
        let s = create_node_key_from_ref(&e.source);
        let t = create_node_key_from_ref(&e.target);
        (s,t,e.edge)
    });
    let batches = build_batch_edge_queries(tuples, EDGE_BATCH_INTERNAL);
    for chunk in batches.chunks(EDGE_TX_CHUNK) { neo.execute_simple(chunk.to_vec()).await?; }
    Ok(())
}