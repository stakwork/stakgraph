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
    let repo_url = body.repo_url.clone();
    let username = body.username.clone();
    let pat = body.pat.clone();

    let result = tokio::task::spawn_blocking(move || {
        #[cfg(feature = "neo4j")]
        {
            use futures::executor::block_on;

            info!("Processing repository: {}", repo_url);
            let mut graph = Neo4jGraph::default();

            if let Err(e) = block_on(graph.connect()) {
                return Err(AppError::Anyhow(anyhow::anyhow!(
                    "Neo4j connection error: {}",
                    e
                )));
            }

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

            let (old_nodes, old_edges) = graph.get_graph_size();
            info!("Old graph size: {} nodes, {} edges", old_nodes, old_edges);

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
                    return Ok(ProcessResponse {
                        status: "success".to_string(),
                        message: "Repository already processed".to_string(),
                        nodes: nodes as usize,
                        edges: edges as usize,
                    });
                }
            }

            if let Some(hash) = stored_hash {
                info!("Updating repository hash from {} to {}", hash, current_hash);
                let revs = vec![hash.clone(), current_hash.clone()];

                if let Some(changed_files) = check_revs_files(&repo_path, revs.clone()) {
                    info!(
                        "Processing {} changed files between commits",
                        changed_files.len()
                    );

                    let mut deleted_files = Vec::new();
                    for file_path in &changed_files {
                        let full_path = std::path::Path::new(&repo_path).join(file_path);
                        if !full_path.exists() {
                            info!("File deleted: {}", file_path);
                            deleted_files.push(file_path.clone());

                            let (nodes_before, edges_before) = graph.get_graph_size();
                            let deleted_count = graph.remove_nodes_by_file(file_path)?;
                            let (nodes_after, edges_after) = graph.get_graph_size();

                            if deleted_count > 0 {
                                info!(
                                    "Removed {} nodes and {} edges for deleted file {}",
                                    nodes_before - nodes_after,
                                    edges_before - edges_after,
                                    file_path
                                );
                            } else {
                                info!("No nodes removed for deleted file {}", file_path);
                                // return Err(AppError::Anyhow(anyhow::anyhow!(
                                //     "No nodes removed for deleted file {}",
                                //     file_path
                                // )));
                            }
                        }
                    }

                    let modified_files: Vec<String> = changed_files
                        .iter()
                        .filter(|f| !deleted_files.contains(f))
                        .cloned()
                        .collect();

                    if !modified_files.is_empty() {
                        for file in &modified_files {
                            graph.remove_nodes_by_file(file)?;

                            let file_repos = block_on(Repo::new_multi_detect(
                                &repo_path,
                                Some(repo_url.clone()),
                                vec![file.clone()],
                                Vec::new(),
                            ))?;

                            for repo in &file_repos.0 {
                                let file_graph = block_on(repo.build_graph_inner::<Neo4jGraph>())?;

                                let (nodes_before, edges_before) = graph.get_graph_size();
                                graph.extend_graph(file_graph);
                                let (nodes_after, edges_after) = graph.get_graph_size();

                                info!(
                                    "Updated file {}: added {} nodes and {} edges",
                                    file,
                                    nodes_after - nodes_before,
                                    edges_after - edges_before
                                );
                            }
                        }
                    }
                }

                graph.update_repository_hash(&repo_url, &current_hash)?;
            } else {
                info!("Adding new repository hash: {}", current_hash);

                let repos = match block_on(Repo::new_multi_detect(
                    &repo_path,
                    Some(repo_url.clone()),
                    Vec::new(),
                    Vec::new(),
                )) {
                    Ok(r) => r,
                    Err(e) => return Err(AppError::Anyhow(e.into())),
                };

                let temp_graph = match block_on(repos.build_graphs_inner::<Neo4jGraph>()) {
                    Ok(g) => g,
                    Err(e) => return Err(AppError::Anyhow(e.into())),
                };

                graph.extend_graph(temp_graph);

                graph.update_repository_hash(&repo_url, &current_hash)?;
            }

            let (nodes, edges) = graph.get_graph_size();
            info!("Updated graph - Nodes: {}, Edges: {}", nodes, edges);
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

#[derive(Debug)]
pub enum AppError {
    Anyhow(anyhow::Error),
}

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
