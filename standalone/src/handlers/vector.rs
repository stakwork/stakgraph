use crate::types::{EmbedCodeParams, Result};
use ast::lang::graphs::graph_ops::GraphOps;
use axum::{extract::Query, Json};

pub async fn embed_code_handler(
    Query(params): Query<EmbedCodeParams>,
) -> Result<Json<serde_json::Value>> {
    let do_files = params.files.unwrap_or(false);
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;
    graph_ops.embed_data_bank_bodies(do_files).await?;
    Ok(Json(serde_json::json!({ "status": "completed" })))
}
