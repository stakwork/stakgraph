use std::vec;

use anyhow::Context;
use ast::builder::filter_by_revs; // Import the function
#[cfg(feature = "neo4j")]
use ast::lang::graphs::Neo4jGraph;
use ast::lang::Graph;
use ast::repo::{check_revs_files, Repo};
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use lsp::git::{get_commit_hash, git_pull_or_clone};
use serde::{Deserialize, Serialize};
use tracing::info;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::EnvFilter;
pub type Result<T> = std::result::Result<T, AppError>;

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(filter)
        .init();
    start().await
}

async fn start() -> Result<()> {
    let cors_layer = tower_http::cors::CorsLayer::permissive();
    let app = Router::new()
        .route("/process", post(process))
        .layer(cors_layer);
    let port = std::env::var("PORT").unwrap_or_else(|_| "7777".to_string());
    let bind = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(bind).await.unwrap();
    println!("=> listening on http://{}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
    Ok(())
}

#[derive(Serialize, Deserialize)]
pub struct ProcessBody {
    pub repo_url: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub pat: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ProcessResponse {
    pub status: String,
    pub message: String,
    pub nodes: usize,
    pub edges: usize,
}

#[axum::debug_handler]
pub async fn process(Json(body): Json<ProcessBody>) -> Result<Json<ProcessResponse>> {
    #[cfg(feature = "neo4j")]
    {
        let repo_url = body.repo_url;
        info!("Processing repository: {}", repo_url);
        let mut graph = Neo4jGraph::default();

        graph.connect().await?;

        let repo_path = Repo::get_path_from_url(&repo_url)?;

        git_pull_or_clone(
            &repo_url,
            &repo_path,
            body.username.clone(),
            body.pat.clone(),
        )
        .await
        .context("Failed to clone repository")?;

        let current_hash = get_commit_hash(&repo_path).await?;

        let stored_hash = match graph.get_repository_hash(&repo_url) {
            Ok(hash) => Some(hash),
            Err(_) => None,
        };
        info!(
            "Current hash: {} | Stored hash: {:?}",
            current_hash, stored_hash
        );

        if let Some(hash) = &stored_hash {
            if hash == &current_hash {
                let (nodes, edges) = graph.get_graph_size();
                return Ok(Json(ProcessResponse {
                    status: "success".to_string(),
                    message: "Repository already processed".to_string(),
                    nodes: nodes as usize,
                    edges: edges as usize,
                }));
            }
        }

        if let Some(hash) = stored_hash {
            let revs = vec![hash.clone(), current_hash.clone()];

            if let Some(changed_files) = check_revs_files(&repo_path, revs.clone()) {
                let mut deleted_files = Vec::new();
                for file_path in &changed_files {
                    let full_path = std::path::Path::new(&repo_path).join(file_path);
                    if !full_path.exists() {
                        info!("File deleted: {}", file_path);
                        deleted_files.push(file_path.clone());
                        graph.remove_nodes_by_file(file_path)?;
                    }
                }

                //modified and added files
                let mut subgraph = Neo4jGraph::default();
                subgraph.connect().await?;

                let filtered_graph = filter_by_revs(&repo_path, revs.clone(), subgraph);

                let (nodes, edges) = filtered_graph.get_graph_size();
                info!("Filtered graph: \n {} nodes \n{} edges", nodes, edges);

                graph.extend_graph(filtered_graph);
            }

            graph.update_repository_hash(&repo_url, &current_hash)?;

            let (nodes, edges) = graph.get_graph_size();
            info!("Updated graph - Nodes: {}, Edges: {}", nodes, edges);

            return Ok(Json(ProcessResponse {
                status: "success".to_string(),
                message: "Repository processed successfully".to_string(),
                nodes: nodes as usize,
                edges: edges as usize,
            }));
        }
    }

    return Ok(Json(ProcessResponse {
        status: "Failed".to_string(),
        message: "Failed to process repository".to_string(),
        nodes: 0,
        edges: 0,
    }));
}

#[derive(Debug)]
pub enum AppError {
    Anyhow(anyhow::Error),
}

// Tell axum how to convert `AppError` into a response.
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self {
            AppError::Anyhow(err) => (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
        }
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self::Anyhow(err.into())
    }
}
