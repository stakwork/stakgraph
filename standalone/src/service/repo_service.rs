use crate::types::{FetchRepoBody, FetchRepoResponse, Result};
use ast::lang::graphs::graph_ops::GraphOps;
use axum::Json;

pub async fn fetch_repo(body: Json<FetchRepoBody>) -> Result<Json<FetchRepoResponse>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;
    let repo_node = graph_ops.fetch_repo(&body.repo_name).await?;
    Ok(Json(FetchRepoResponse {
        status: "success".to_string(),
        repo_name: repo_node.name,
        hash: repo_node.hash.unwrap_or_default(),
    }))
}

pub async fn fetch_repos() -> Result<Json<Vec<FetchRepoResponse>>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;
    let repo_nodes = graph_ops.fetch_repos().await;
    let repos = repo_nodes
        .into_iter()
        .map(|node| FetchRepoResponse {
            status: "success".to_string(),
            repo_name: node.name,
            hash: node.hash.unwrap_or_default(),
        })
        .collect();
    Ok(Json(repos))
}
