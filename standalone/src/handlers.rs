use crate::types::{
    AsyncRequestStatus, AsyncStatus, FetchRepoBody, FetchRepoResponse, ProcessBody,
    ProcessResponse, Result, WebError,
};
use crate::AppState;
use ast::lang::graphs::graph_ops::GraphOps;
use ast::lang::Graph;
use ast::repo::{clone_repo, Repo};
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::{extract::State, Json};
use broadcast::error::RecvError;
use futures::stream;
use lsp::{git::get_commit_hash, strip_tmp};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use tokio::sync::broadcast;
use tracing::info;

pub async fn sse_handler(State(app_state): State<Arc<AppState>>) -> impl IntoResponse {
    let rx = app_state.tx.subscribe();

    let stream = stream::unfold(rx, move |mut rx| async move {
        loop {
            match rx.recv().await {
                Ok(msg) => {
                    let data = msg.as_json_str();
                    let millis = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis();
                    let event = Event::default().data(data).id(format!("{}", millis));
                    return Some((Ok::<Event, Infallible>(event), rx));
                }
                Err(RecvError::Lagged(skipped)) => {
                    println!("SSE receiver lagged, skipped {} messages", skipped);
                    continue;
                }
                Err(RecvError::Closed) => {
                    return None;
                }
            }
        }
    });

    let headers = [
        ("Cache-Control", "no-cache, no-store, must-revalidate"),
        ("Connection", "keep-alive"),
        ("Content-Type", "text/event-stream"),
        ("X-Accel-Buffering", "no"), // nginx
        ("X-Proxy-Buffering", "no"), // other proxies
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Headers", "Cache-Control"),
    ];
    (
        headers,
        Sse::new(stream).keep_alive(
            KeepAlive::new()
                .interval(Duration::from_millis(500))
                .text("ping"),
        ),
    )
}

#[axum::debug_handler]
pub async fn process(body: Json<ProcessBody>) -> Result<Json<ProcessResponse>> {
    if body.repo_url.clone().unwrap_or_default().contains(",") {
        return Err(WebError(shared::Error::Custom(
            "Multiple repositories are not supported in a single request".into(),
        )));
    }
    let (final_repo_path, final_repo_url, username, pat, _) = resolve_repo(&body)?;
    let use_lsp = body.use_lsp;

    let total_start = Instant::now();

    let repo_path = &final_repo_path;
    let repo_url = &final_repo_url;

    clone_repo(&repo_url, &repo_path, username.clone(), pat.clone(), None).await?;

    let current_hash = match get_commit_hash(&repo_path).await {
        Ok(hash) => hash,
        Err(e) => {
            return Err(WebError(shared::Error::Custom(format!(
                "Could not get current hash: {e}"
            ))));
        }
    };

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    let stored_hash = match graph_ops.graph.get_repository_hash(&repo_url).await {
        Ok(hash) => Some(hash),
        Err(_) => None,
    };

    info!(
        "Current hash: {} | Stored hash: {:?}",
        current_hash, stored_hash
    );

    if let Some(hash) = &stored_hash {
        if hash == &current_hash {
            info!(
                "Repository already processed with hash: {}\n\n",
                current_hash
            );
            let (nodes, edges) = graph_ops.graph.get_graph_size();
            return Ok(Json(ProcessResponse { nodes, edges }));
        }
    }

    let (prev_nodes, prev_edges) = graph_ops.graph.get_graph_size();

    let (nodes, edges) = if let Some(hash) = stored_hash {
        info!("Updating repository hash from {} to {}", hash, current_hash);
        graph_ops
            .update_incremental(
                &repo_url,
                username.clone(),
                pat.clone(),
                &current_hash,
                &hash,
                None,
                use_lsp,
            )
            .await?
    } else {
        info!("Adding new repository hash: {}", current_hash);
        graph_ops
            .update_full(
                &repo_url,
                username.clone(),
                pat.clone(),
                &current_hash,
                None,
                use_lsp,
            )
            .await?
    };
    info!(
        "\n\n ==>> Total processing time: {:.2?} \n\n",
        total_start.elapsed()
    );

    let delta_nodes = nodes - prev_nodes;
    let delta_edges = edges - prev_edges;

    Ok(Json(ProcessResponse {
        nodes: delta_nodes,
        edges: delta_edges,
    }))
}

pub async fn clear_graph() -> Result<Json<ProcessResponse>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;
    let (nodes, edges) = graph_ops.clear().await?;
    Ok(Json(ProcessResponse { nodes, edges }))
}

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

#[axum::debug_handler]
pub async fn ingest(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> Result<Json<ProcessResponse>> {
    let start_total = Instant::now();
    let (_, final_repo_url, username, pat, commit) = resolve_repo(&body)?;
    let use_lsp = body.use_lsp;

    let repo_url = final_repo_url.clone();

    let start_build = Instant::now();

    let mut repos = Repo::new_clone_multi_detect(
        &repo_url,
        username.clone(),
        pat.clone(),
        Vec::new(),
        Vec::new(),
        commit.as_deref(),
        use_lsp,
    )
    .await
    .map_err(|e| {
        WebError(shared::Error::Custom(format!(
            "Repo detection Failed: {}",
            e
        )))
    })?;

    repos.set_status_tx(state.tx.clone()).await;

    let btree_graph = repos
        .build_graphs_inner::<ast::lang::graphs::BTreeMapGraph>()
        .await
        .map_err(|e| {
            WebError(shared::Error::Custom(format!(
                "Failed to build graphs: {}",
                e
            )))
        })?;
    info!(
        "\n\n ==>>Building BTreeMapGraph took {:.2?} \n\n",
        start_build.elapsed()
    );
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    for repo in &repos.0 {
        let stripped_root = strip_tmp(&repo.root).display().to_string();
        info!("Clearing old data for {}...", stripped_root);
        graph_ops.clear_existing_graph(&stripped_root).await?;
    }

    let start_upload = Instant::now();

    info!("Uploading to Neo4j...");
    let (nodes, edges) = graph_ops
        .upload_btreemap_to_neo4j(&btree_graph, Some(state.tx.clone()))
        .await?;
    graph_ops.graph.create_indexes().await?;

    info!(
        "\n\n ==>> Uploading to Neo4j took {:.2?} \n\n",
        start_upload.elapsed()
    );

    info!(
        "\n\n ==>> Total ingest time: {:.2?} \n\n",
        start_total.elapsed()
    );

    if let Ok(diry) = std::env::var("PRINT_ROOT") {
        // add timestamp to the filename
        let timestamp = Instant::now().elapsed().as_millis();
        let filename = format!("{}/standalone-{}", diry, timestamp);
        info!("Printing nodes and edges to files... {}", filename);
        if let Err(e) = ast::utils::print_json(&btree_graph, &filename) {
            tracing::warn!("Error printing nodes and edges to files: {}", e);
        }
    }

    Ok(Json(ProcessResponse { nodes, edges }))
}

#[axum::debug_handler]
pub async fn ingest_async(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> impl IntoResponse {
    let request_id = uuid::Uuid::new_v4().to_string();
    let status_map = state.async_status.clone();
    let mut rx = state.tx.subscribe();

    {
        let mut map = status_map.lock().await;
        map.insert(
            request_id.clone(),
            AsyncRequestStatus {
                status: AsyncStatus::InProgress,
                result: None,
                progress: 0,
            },
        );
    }

    let state_clone = state.clone();
    let status_map_clone = status_map.clone();
    let body_clone = body.clone();
    let request_id_clone = request_id.clone();

    tokio::spawn(async move {
        while let Ok(update) = rx.recv().await {
            let mut map = status_map_clone.lock().await;
            if let Some(status) = map.get_mut(&request_id_clone) {
                let total_steps = update.total_steps.max(1) as f64;
                let step = update.step.max(1) as f64;
                let step_progress = update.progress.min(100) as f64;

                let overall_progress = (((step - 1.0) + (step_progress / 100.0)) / total_steps
                    * 100.0)
                    .min(100.0) as u32;
                status.progress = overall_progress;
            }
        }
    });

    let request_id_clone = request_id.clone();

    //run ingest as a background task
    tokio::spawn(async move {
        let result = ingest(State(state_clone), body_clone).await;
        let mut map = status_map.lock().await;

        match result {
            Ok(Json(resp)) => map.insert(
                request_id_clone,
                AsyncRequestStatus {
                    status: AsyncStatus::Complete,
                    result: Some(resp),
                    progress: 100,
                },
            ),
            Err(e) => map.insert(
                request_id_clone,
                AsyncRequestStatus {
                    status: AsyncStatus::Failed(format!("{:?}", e)),
                    result: None,
                    progress: 0,
                },
            ),
        }
    });

    Json(serde_json::json!({ "request_id": request_id }))
}

#[axum::debug_handler]
pub async fn sync_async(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> impl IntoResponse {
    let request_id = uuid::Uuid::new_v4().to_string();
    let status_map = state.async_status.clone();

    {
        let mut map = status_map.lock().await;
        map.insert(
            request_id.clone(),
            AsyncRequestStatus {
                status: AsyncStatus::InProgress,
                result: None,
                progress: 0,
            },
        );
    }
    let body_clone = body.clone();
    let request_id_clone = request_id.clone();

    //run /sync as a background task
    tokio::spawn(async move {
        let result = process(body_clone).await;
        let mut map = status_map.lock().await;

        match result {
            Ok(Json(resp)) => map.insert(
                request_id_clone,
                AsyncRequestStatus {
                    status: AsyncStatus::Complete,
                    result: Some(resp),
                    progress: 100,
                },
            ),
            Err(e) => map.insert(
                request_id_clone,
                AsyncRequestStatus {
                    status: AsyncStatus::Failed(format!("{:?}", e)),
                    result: None,
                    progress: 0,
                },
            ),
        }
    });

    Json(serde_json::json!({ "request_id": request_id }))
}

pub async fn get_status(
    State(state): State<Arc<AppState>>,
    Path(request_id): Path<String>,
) -> impl IntoResponse {
    let status_map = state.async_status.clone();
    let map = status_map.lock().await;

    if let Some(status) = map.get(&request_id) {
        Json(status).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            format!("Request ID {} not found", request_id),
        )
            .into_response()
    }
}

fn env_not_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

fn resolve_repo(
    body: &ProcessBody,
) -> Result<(
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
)> {
    let repo_path = body
        .repo_path
        .clone()
        .or_else(|| env_not_empty("REPO_PATH"));
    let repo_url = body.repo_url.clone().or_else(|| env_not_empty("REPO_URL"));
    let username = body.username.clone().or_else(|| env_not_empty("USERNAME"));
    let pat = body.pat.clone().or_else(|| env_not_empty("PAT"));
    let commit = body.commit.clone();

    if repo_path.is_none() && repo_url.is_none() {
        return Err(WebError(shared::Error::Custom(
            "Neither REPO_PATH nor REPO_URL is set in the body or environment".into(),
        )));
    }

    if let Some(path) = repo_path {
        Ok((path, repo_url.unwrap_or_default(), username, pat, commit))
    } else {
        let url = repo_url.unwrap();
        let tmp_path = Repo::get_path_from_url(&url)?;
        Ok((tmp_path, url, username, pat, commit))
    }
}
