use crate::types::{Result, VectorSearchParams, VectorSearchResult};
use axum::{Json, extract::Query};
use ast::lang::graphs::graph_ops::GraphOps;


pub async fn vector_search_handler(
    Query(params): Query<VectorSearchParams>,
) -> Result<Json<Vec<VectorSearchResult>>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    //comma-separated node types
    let node_types: Vec<String> = params
        .node_types
        .as_ref()
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let results = graph_ops
        .vector_search(
            &params.query,
            params.limit.unwrap_or(10),
            node_types,
            params.similarity_threshold.unwrap_or(0.7),
            params.language.as_deref(),
        )
        .await?;

    let is_test = std::env::var("TEST_REF_ID")
        .ok()
        .filter(|v| !v.is_empty())
        .is_some();
    let response: Vec<VectorSearchResult> = results
        .into_iter()
        .map(|(mut node, score)| {
            if is_test {
                node.meta.remove("date_added_to_graph");
            }
            VectorSearchResult { node, score }
        })
        .collect();

    Ok(Json(response))
}
