use crate::types::{
    AsyncRequestStatus, AsyncStatus, CodecovBody, CodecovRequestStatus, Coverage, CoverageParams,
    CoverageStat, EmbedCodeParams, FetchRepoBody, FetchRepoResponse, HasParams, HasResponse, Node,
    NodeConcise, NodesResponseItem, ProcessBody, ProcessResponse, QueryNodesParams,
    QueryNodesResponse, Result, VectorSearchParams, VectorSearchResult, WebError, WebhookPayload,
};
use crate::utils::parse_node_type;
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
    let (final_repo_path, final_repo_url, username, pat, _, branch) = resolve_repo(&body)?;

    if let Err(e) = validate_git_credentials(&final_repo_url, username.clone(), pat.clone()).await {
        return Err(WebError(e));
    }

    let use_lsp = body.use_lsp;

    let total_start = Instant::now();

    let repo_path = &final_repo_path;
    let repo_url = &final_repo_url;

    clone_repo(
        &repo_url,
        &repo_path,
        username.clone(),
        pat.clone(),
        None,
        branch.as_deref(),
    )
    .await?;

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
            branch.as_deref(),
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
    let (final_repo_path, final_repo_url, username, pat, commit, branch) = resolve_repo(&body)?;
    let use_lsp = body.use_lsp;
    let repo_url = final_repo_url.clone();

    let start_clone = Instant::now();
    let mut repos = if body.repo_path.is_some() || std::env::var("REPO_PATH").is_ok() {
        info!("Using local repository at: {}", final_repo_path);
        Repo::new_multi_detect(
            &final_repo_path,
            Some(final_repo_url.clone()),
            Vec::new(),
            Vec::new(),
            use_lsp,
        )
        .await
        .map_err(|e| {
            WebError(shared::Error::Custom(format!(
                "Repo detection Failed: {}",
                e
            )))
        })?
    } else {
        Repo::new_clone_multi_detect(
            &repo_url,
            username.clone(),
            pat.clone(),
            Vec::new(),
            Vec::new(),
            commit.as_deref(),
            branch.as_deref(),
            use_lsp,
        )
        .await
        .map_err(|e| {
            WebError(shared::Error::Custom(format!(
                "Repo detection Failed: {}",
                e
            )))
        })?
    };
    let clone_s = start_clone.elapsed().as_secs_f64();
    info!(
        "[perf][ingest] phase=clone_detect repo={} s={:.2}",
        final_repo_url.clone(), clone_s
    );

    repos.set_status_tx(state.tx.clone()).await;
    let streaming = std::env::var("STREAM_UPLOAD").is_ok();
    if streaming {
        let mut graph_ops = GraphOps::new();
        graph_ops.connect().await?;
        for repo in &repos.0 {
            let stripped_root = strip_tmp(&repo.root).display().to_string();
            info!("[Stream] Pre-clearing old data for {}...", stripped_root);
            graph_ops.clear_existing_graph(&stripped_root).await?;
        }
    }

    let start_build = Instant::now();
    let btree_graph = repos
        .build_graphs_inner::<ast::lang::graphs::BTreeMapGraph>()
        .await
        .map_err(|e| {
            WebError(shared::Error::Custom(format!(
                "Failed to build graphs: {}",
                e
            )))
        })?;
    let build_s = start_build.elapsed().as_secs_f64();
    info!(
        "[perf][ingest] phase=build repo={} streaming={} s={:.2}",
        final_repo_url.clone(), streaming, build_s
    );
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    if !streaming {
        for repo in &repos.0 {
            let stripped_root = strip_tmp(&repo.root).display().to_string();
            info!("Clearing old data for {}...", stripped_root);
            graph_ops.clear_existing_graph(&stripped_root).await?;
        }
    }

    let start_upload = Instant::now();

    let (nodes, edges) = if streaming {
        graph_ops.graph.get_graph_size()
    } else {
        info!("Uploading to Neo4j...");
        let res = graph_ops
            .upload_btreemap_to_neo4j(&btree_graph, Some(state.tx.clone()))
            .await?;
        graph_ops.graph.create_indexes().await?;
        res
    };

    // Set Data_Bank property for nodes that don't have it
    info!("Setting Data_Bank property for nodes missing it...");
    if let Err(e) = graph_ops.set_missing_data_bank().await {
        tracing::warn!("Error setting Data_Bank property: {:?}", e);
    }

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

    let upload_s = start_upload.elapsed().as_secs_f64();
    info!(
        "[perf][ingest] phase=upload repo={} streaming={} s={:.2}",
        final_repo_url, streaming, upload_s
    );

    let build_upload_s = build_s + upload_s;
    let total_s = start_total.elapsed().as_secs_f64();
    info!(
        "[perf][ingest][results] repo={} streaming={} clone_s={:.2} build_s={:.2} upload_s={:.2} build_upload_s={:.2} total_s={:.2} nodes={} edges={}",
        final_repo_url,
        streaming,
        clone_s,
        build_s,
        upload_s,
        build_upload_s,
        total_s,
        nodes,
        edges
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
    let (_, repo_url, username, pat, _, _branch) = match resolve_repo(&body) {
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
    let (_, repo_url, username, pat, _, _branch) = match resolve_repo(&body) {
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

    info!(
        "/sync with Request ID: {} and callback_url:  {:?}",
        request_id_clone.clone(),
        callback_url.clone()
    );

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
    let branch = body.branch.clone();

    if repo_path.is_none() && repo_url.is_none() {
        return Err(WebError(shared::Error::Custom(
            "Neither REPO_PATH nor REPO_URL is set in the body or environment".into(),
        )));
    }

    if let Some(path) = repo_path {
        Ok((
            path,
            repo_url.unwrap_or_default(),
            username,
            pat,
            commit,
            branch,
        ))
    } else {
        let url = repo_url.unwrap();
        let tmp_path = Repo::get_path_from_url(&url)?;
        Ok((tmp_path, url, username, pat, commit, branch))
    }
}

#[axum::debug_handler]
pub async fn coverage_handler(Query(params): Query<CoverageParams>) -> Result<Json<Coverage>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    let ignore_dirs: Vec<String> = params
        .ignore_dirs
        .as_ref()
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let totals = graph_ops
        .get_coverage(params.repo.as_deref(), ignore_dirs)
        .await?;

    Ok(Json(Coverage {
        unit_tests: totals.unit_tests.map(|s| CoverageStat {
            total: s.total,
            total_tests: s.total_tests,
            covered: s.covered,
            percent: s.percent,
            total_lines: s.total_lines,
            covered_lines: s.covered_lines,
            line_percent: s.line_percent,
        }),
        integration_tests: totals.integration_tests.map(|s| CoverageStat {
            total: s.total,
            total_tests: s.total_tests,
            covered: s.covered,
            percent: s.percent,
            total_lines: s.total_lines,
            covered_lines: s.covered_lines,
            line_percent: s.line_percent,
        }),
        e2e_tests: totals.e2e_tests.map(|s| CoverageStat {
            total: s.total,
            total_tests: s.total_tests,
            covered: s.covered,
            percent: s.percent,
            total_lines: s.total_lines,
            covered_lines: s.covered_lines,
            line_percent: s.line_percent,
        }),
    }))
}

#[axum::debug_handler]
pub async fn nodes_handler(
    Query(params): Query<QueryNodesParams>,
) -> Result<Json<QueryNodesResponse>> {
    let node_type = parse_node_type(&params.node_type).map_err(|e| WebError(e))?;
    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(10).min(100);
    let sort_by_test_count = params.sort.as_deref().unwrap_or("test_count") == "test_count";
    let coverage_filter = params.coverage.as_deref();
    let concise = params.concise.unwrap_or(true);
    let body_length = params.body_length.unwrap_or(false);
    let line_count = params.line_count.unwrap_or(false);

    if let Some(coverage) = coverage_filter {
        if !matches!(coverage, "tested" | "untested" | "all") {
            return Err(WebError(shared::Error::Custom(
                "Invalid coverage parameter. Must be 'tested', 'untested', or 'all'".into(),
            )));
        }
    }

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    let ignore_dirs: Vec<String> = params
        .ignore_dirs
        .as_ref()
        .map(|s| s.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let (total_count, results) = graph_ops
        .query_nodes_with_count(
            node_type.clone(),
            offset,
            limit,
            sort_by_test_count,
            coverage_filter,
            body_length,
            line_count,
            ignore_dirs,
        )
        .await?;

    let items: Vec<NodesResponseItem> = results
        .into_iter()
        .map(
            |(node_data, usage_count, covered, test_count, ref_id, body_length, line_count)| {
                if concise {
                    NodesResponseItem::Concise(NodeConcise {
                        name: node_data.name,
                        file: node_data.file,
                        ref_id,
                        weight: usage_count,
                        test_count,
                        covered,
                        body_length,
                        line_count,
                    })
                } else {
                    NodesResponseItem::Full(Node {
                        node_type: node_type.to_string(),
                        ref_id,
                        weight: usage_count,
                        test_count,
                        covered,
                        properties: node_data,
                        body_length,
                        line_count,
                    })
                }
            },
        )
        .collect();

    let total_returned = items.len();
    let total_pages = if limit > 0 {
        (total_count + limit - 1) / limit
    } else {
        0
    };
    let current_page = if limit > 0 { (offset / limit) + 1 } else { 0 };

    Ok(Json(QueryNodesResponse {
        items,
        total_returned,
        total_count,
        total_pages,
        current_page,
    }))
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
    info!(
        "[/tests/has] node_type={:?} name={:?} file={:?} start={:?} root={:?} tests={:?}",
        node_type, params.name, params.file, params.start, params.root, params.tests
    );
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

#[axum::debug_handler]
pub async fn codecov_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<CodecovBody>,
) -> impl IntoResponse {
    let request_id = uuid::Uuid::new_v4().to_string();
    let codecov_status_map = state.codecov_status.clone();

    {
        let mut map = codecov_status_map.lock().await;
        map.insert(
            request_id.clone(),
            CodecovRequestStatus {
                status: AsyncStatus::InProgress,
                result: None,
                error: None,
            },
        );
    }

    let request_id_clone = request_id.clone();
    let codecov_status_map_clone = codecov_status_map.clone();

    tokio::spawn(async move {
        match crate::codecov::run(body).await {
            Ok(report) => {
                let mut map = codecov_status_map_clone.lock().await;
                if let Some(status) = map.get_mut(&request_id_clone) {
                    status.status = AsyncStatus::Complete;
                    status.result = Some(report);
                }
            }
            Err(e) => {
                let mut map = codecov_status_map_clone.lock().await;
                if let Some(status) = map.get_mut(&request_id_clone) {
                    status.status = AsyncStatus::Failed(e.to_string());
                    status.error = Some(e.to_string());
                }
            }
        }
    });

    Json(serde_json::json!({ "request_id": request_id })).into_response()
}

#[axum::debug_handler]
pub async fn codecov_status_handler(
    State(state): State<Arc<AppState>>,
    Path(request_id): Path<String>,
) -> impl IntoResponse {
    let codecov_status_map = state.codecov_status.clone();
    let map = codecov_status_map.lock().await;

    if let Some(status) = map.get(&request_id) {
        Json(status).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "Request ID not found" })),
        )
            .into_response()
    }
}
