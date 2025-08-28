use crate::types::{
    AsyncRequestStatus, AsyncStatus, CoverageParams, CoverageStat, Coverage, EmbedCodeParams,
    FetchRepoBody, FetchRepoResponse, HasParams, HasResponse, ProcessBody, ProcessResponse, Result,
    UncoveredParams, UncoveredResponse, VectorSearchParams, VectorSearchResult, WebError,
    WebhookPayload,
};
use crate::utils::{
    create_uncovered_response_items, format_uncovered_response_as_snippet, parse_node_type,
};
use crate::webhook::{send_with_retries, validate_callback_url_async};
use crate::AppState;
use ast::lang::graphs::graph_ops::GraphOps;
use ast::lang::{Graph, NodeType};
use ast::repo::{clone_repo, Repo};
use axum::extract::{Path, Query};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use axum::{extract::State, Json};
use broadcast::error::RecvError;
use chrono::Utc;
use futures::stream;
use lsp::{git::get_commit_hash, git::validate_git_credentials, strip_tmp};
use reqwest::Client;
use shared::Error;
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

    if let Err(e) = validate_git_credentials(&final_repo_url, username.clone(), pat.clone()).await {
        return Err(WebError(e));
    }

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

    let hash = stored_hash.as_deref().unwrap_or_default();

    let (prev_nodes, prev_edges) = graph_ops.graph.get_graph_size();

    info!("Updating repository hash from {} to {}", hash, current_hash);
    let (nodes, edges) = graph_ops
        .update_incremental(
            &repo_url,
            username.clone(),
            pat.clone(),
            &current_hash,
            hash,
            None,
            use_lsp,
        )
        .await?;

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

    let _ = state.tx.send(ast::repo::StatusUpdate {
        status: "Complete".to_string(),
        message: "Graph building completed successfully".to_string(),
        step: 16,
        total_steps: 16,
        progress: 100,
        stats: Some(std::collections::HashMap::from([
            ("total_nodes".to_string(), nodes as usize),
            ("total_edges".to_string(), edges as usize),
        ])),
        step_description: Some("Graph building completed".to_string()),
    });

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
    let (_, repo_url, username, pat, _) = match resolve_repo(&body) {
        Ok(config) => config,
        Err(e) => {
            return Json(serde_json::json!({
                "error": format!("Invalid repository configuration: {:?}", e)
            }))
            .into_response();
        }
    };

    if let Err(e) = validate_git_credentials(&repo_url, username.clone(), pat.clone()).await {
        return Json(serde_json::json!({
            "error": format!("Git authentication failed: {:?}", e)
        }))
        .into_response();
    }

    let request_id = uuid::Uuid::new_v4().to_string();
    let status_map = state.async_status.clone();
    let mut rx = state.tx.subscribe();

    let callback_url = body.callback_url.clone();
    let started_at = Utc::now();

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
            Ok(Json(resp)) => {
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Complete,
                    result: Some(resp.clone()),
                    progress: 100,
                };
                map.insert(request_id_clone.clone(), entry);
                if let Some(url) = callback_url {
                    if let Ok(valid) = validate_callback_url_async(&url).await {
                        let payload = WebhookPayload {
                            request_id: request_id_clone.clone(),
                            status: "Complete".to_string(),
                            progress: 100,
                            result: Some(ProcessResponse {
                                nodes: resp.nodes,
                                edges: resp.edges,
                            }),
                            error: None,
                            started_at: started_at.to_rfc3339(),
                            completed_at: Utc::now().to_rfc3339(),
                            duration_ms: (Utc::now() - started_at).num_milliseconds().max(0) as u64,
                        };
                        let client = Client::new();
                        let _ = send_with_retries(&client, &request_id_clone, &valid, &payload)
                            .await
                            .map_err(|e| {
                                tracing::error!("Error sending webhook: {:?}", e);
                                WebError(shared::Error::Custom(format!(
                                    "Error sending webhook: {:?}",
                                    e
                                )))
                            });
                    }
                }
            }
            Err(e) => {
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Failed(format!("{:?}", e)),
                    result: None,
                    progress: 0,
                };
                map.insert(request_id_clone.clone(), entry);
                if let Some(url) = callback_url {
                    if let Ok(valid) = validate_callback_url_async(&url).await {
                        let payload = WebhookPayload {
                            request_id: request_id_clone.clone(),
                            status: "Failed".to_string(),
                            progress: 0,
                            result: None,
                            error: Some(format!("{:?}", e)),
                            started_at: started_at.to_rfc3339(),
                            completed_at: Utc::now().to_rfc3339(),
                            duration_ms: (Utc::now() - started_at).num_milliseconds().max(0) as u64,
                        };
                        let client = Client::new();
                        let _ = send_with_retries(&client, &request_id_clone, &valid, &payload)
                            .await
                            .map_err(|e| {
                                tracing::error!("Error sending webhook: {:?}", e);
                                WebError(shared::Error::Custom(format!(
                                    "Error sending webhook: {:?}",
                                    e
                                )))
                            });
                    }
                }
            }
        }
    });

    Json(serde_json::json!({ "request_id": request_id })).into_response()
}

#[axum::debug_handler]
pub async fn sync_async(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> impl IntoResponse {
    let (_, repo_url, username, pat, _) = match resolve_repo(&body) {
        Ok(config) => config,
        Err(e) => {
            return Json(serde_json::json!({
                "error": format!("Invalid repository configuration: {:?}", e)
            }))
            .into_response();
        }
    };

    if let Err(e) = validate_git_credentials(&repo_url, username.clone(), pat.clone()).await {
        return Json(serde_json::json!({
            "error": format!("Git authentication failed: {:?}", e)
        }))
        .into_response();
    }

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
    let callback_url = body.callback_url.clone();
    let started_at = Utc::now();

    //run /sync as a background task
    tokio::spawn(async move {
        let result = process(body_clone).await;
        let mut map = status_map.lock().await;

        match result {
            Ok(Json(resp)) => {
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Complete,
                    result: Some(resp.clone()),
                    progress: 100,
                };
                map.insert(request_id_clone.clone(), entry);
                if let Some(url) = callback_url.clone() {
                    if let Ok(valid) = crate::webhook::validate_callback_url_async(&url).await {
                        let payload = WebhookPayload {
                            request_id: request_id_clone.clone(),
                            status: "Complete".to_string(),
                            progress: 100,
                            result: Some(ProcessResponse {
                                nodes: resp.nodes,
                                edges: resp.edges,
                            }),
                            error: None,
                            started_at: started_at.to_rfc3339(),
                            completed_at: Utc::now().to_rfc3339(),
                            duration_ms: (Utc::now() - started_at).num_milliseconds().max(0) as u64,
                        };
                        let client = Client::new();
                        let _ = crate::webhook::send_with_retries(
                            &client,
                            &request_id_clone,
                            &valid,
                            &payload,
                        )
                        .await;
                    }
                }
            }
            Err(e) => {
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Failed(format!("{:?}", e)),
                    result: None,
                    progress: 0,
                };
                map.insert(request_id_clone.clone(), entry);
                if let Some(url) = callback_url.clone() {
                    if let Ok(valid) = crate::webhook::validate_callback_url_async(&url).await {
                        let payload = WebhookPayload {
                            request_id: request_id_clone.clone(),
                            status: "Failed".to_string(),
                            progress: 0,
                            result: None,
                            error: Some(format!("{:?}", e)),
                            started_at: started_at.to_rfc3339(),
                            completed_at: Utc::now().to_rfc3339(),
                            duration_ms: (Utc::now() - started_at).num_milliseconds().max(0) as u64,
                        };
                        let client = Client::new();
                        let _ = crate::webhook::send_with_retries(
                            &client,
                            &request_id_clone,
                            &valid,
                            &payload,
                        )
                        .await;
                    }
                }
            }
        }
    });

    Json(serde_json::json!({ "request_id": request_id })).into_response()
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

pub async fn embed_code_handler(
    Query(params): Query<EmbedCodeParams>,
) -> Result<Json<serde_json::Value>> {
    let do_files = params.files.unwrap_or(false);
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;
    graph_ops.embed_data_bank_bodies(do_files).await?;
    Ok(Json(serde_json::json!({ "status": "completed" })))
}

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

    let response: Vec<VectorSearchResult> = results
        .into_iter()
        .map(|(node, score)| VectorSearchResult { node, score })
        .collect();

    Ok(Json(response))
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

#[axum::debug_handler]
pub async fn coverage_handler(
    Query(params): Query<CoverageParams>,
) -> Result<Json<Coverage>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    let totals = graph_ops
        .get_coverage(
            params.repo.as_deref(),
        )
        .await?;

    Ok(Json(Coverage {
    unit_tests: totals.unit_tests.map(|s| CoverageStat { total: s.total, covered: s.covered, percent: s.percent }),
    integration_tests: totals.integration_tests.map(|s| CoverageStat { total: s.total, covered: s.covered, percent: s.percent }),
    e2e_tests: totals.e2e_tests.map(|s| CoverageStat { total: s.total, covered: s.covered, percent: s.percent }),
    }))
}

#[axum::debug_handler]
pub async fn uncovered_handler(Query(params): Query<UncoveredParams>) -> Result<impl IntoResponse> {
    let with_usage = params
        .sort
        .as_deref()
        .unwrap_or("usage")
        .eq_ignore_ascii_case("usage");
    let limit = params.limit.unwrap_or(50);
    let output = params.output.as_deref().unwrap_or("json");
    let concise = params.concise.unwrap_or(false);

    let node_type = parse_node_type(&params.node_type).map_err(|e| WebError(e))?;

    let is_function = matches!(node_type, NodeType::Function);
    let is_endpoint = matches!(node_type, NodeType::Endpoint);

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    let (funcs, endpoints) = graph_ops
        .list_uncovered(
            node_type,
            with_usage,
            limit,
            params.root.as_deref(),
            params.tests.as_deref(),
        )
        .await?;

    let functions = if is_function {
        Some(create_uncovered_response_items(
            funcs,
            &NodeType::Function,
            concise,
        ))
    } else {
        None
    };
    let endpoints = if is_endpoint {
        Some(create_uncovered_response_items(
            endpoints,
            &NodeType::Endpoint,
            concise,
        ))
    } else {
        None
    };

    let response = UncoveredResponse {
        functions,
        endpoints,
    };
    match output {
        "snippet" => {
            let text = format_uncovered_response_as_snippet(&response);
            Ok(text.into_response())
        }
        _ => Ok(Json(response).into_response()),
    }
}

#[axum::debug_handler]
pub async fn has_handler(Query(params): Query<HasParams>) -> Result<Json<HasResponse>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;
    let node_type = match params.node_type.to_lowercase().as_str() {
        "function" => NodeType::Function,
        "endpoint" => NodeType::Endpoint,
        _ => return Err(WebError(Error::Custom("invalid node_type".into()))),
    };
    let covered = graph_ops
        .has_coverage(
            node_type,
            &params.name,
            &params.file,
            params.start,
            params.root.as_deref(),
            params.tests.as_deref(),
        )
        .await?;
    Ok(Json(HasResponse { covered }))
}
