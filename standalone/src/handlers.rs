use crate::types::ProcessBody;
use crate::types::{AppError, ProcessResponse, Result};
#[cfg(feature = "neo4j")]
use ast::lang::graphs::graph_ops::GraphOps;
use ast::lang::Graph;
use ast::repo::Repo;
use axum::Json;
use lsp::git::{get_commit_hash, git_pull_or_clone};
use tracing::info;
pub async fn process(Json(body): Json<ProcessBody>) -> Result<Json<ProcessResponse>> {
    let repo_url = body.repo_url.clone();
    let username = body.username.clone();
    let pat = body.pat.clone();

    let result = tokio::task::spawn_blocking(move || {
        #[cfg(feature = "neo4j")]
        {
            use futures::executor::block_on;

            info!("Processing repository: {}", repo_url);
            let mut graph_ops = GraphOps::new();
            graph_ops.connect()?;

            let repo_path = Repo::get_path_from_url(&repo_url)?;
            if let Err(e) = block_on(git_pull_or_clone(&repo_url, &repo_path, username, pat)) {
                return Err(AppError::Anyhow(anyhow::anyhow!(
                    "Git pull or clone failed : {}",
                    e
                )));
            }

            let current_hash = match block_on(get_commit_hash(&repo_path)) {
                Ok(hash) => hash,
                Err(e) => {
                    return Err(AppError::Anyhow(anyhow::anyhow!(
                        "Could not get current hash: {}",
                        e
                    )))
                }
            };

            let stored_hash = match graph_ops.graph.get_repository_hash(&repo_url) {
                Ok(hash) => Some(hash),
                Err(_) => None,
            };

            info!(
                "Current hash: {} | Stored hash: {:?}",
                current_hash, stored_hash
            );

            if let Some(hash) = &stored_hash {
                if hash == &current_hash {
                    let (nodes, edges) = graph_ops.graph.get_graph_size();
                    return Ok(ProcessResponse {
                        status: "success".to_string(),
                        message: "Repository already processed".to_string(),
                        nodes: nodes as usize,
                        edges: edges as usize,
                    });
                }
            }

            let (nodes, edges) = if let Some(hash) = stored_hash {
                info!("Updating repository hash from {} to {}", hash, current_hash);
                graph_ops.update_incremental(&repo_url, &repo_path, &current_hash, &hash)?
            } else {
                info!("Adding new repository hash: {}", current_hash);
                graph_ops.update_full(&repo_url, &repo_path, &current_hash)?
            };

            Ok(ProcessResponse {
                status: "success".to_string(),
                message: "Repository processed successfully".to_string(),
                nodes: nodes as usize,
                edges: edges as usize,
            })
        }
        #[cfg(not(feature = "neo4j"))]
        {
            Err(AppError::Anyhow(anyhow::anyhow!(
                "Neo4j feature is not enabled"
            )))
        }
    })
    .await
    .map_err(|e| AppError::Anyhow(anyhow::anyhow!("An error occured: {}", e)))?;

    Ok(Json(result?))
}

pub async fn clear_graph() -> Result<Json<ProcessResponse>> {
    #[cfg(feature = "neo4j")]
    {
        let mut graph_ops = GraphOps::new();
        graph_ops.connect()?;
        let (nodes, edges) = graph_ops.clear()?;
        Ok(Json(ProcessResponse {
            status: "success".to_string(),
            message: "Graph cleared".to_string(),
            nodes: nodes as usize,
            edges: edges as usize,
        }))
    }
    #[cfg(not(feature = "neo4j"))]
    {
        Err(AppError::Anyhow(anyhow::anyhow!(
            "Neo4j feature is not enabled"
        )))
    }
}
