use axum::http::StatusCode;
use axum::response::Response;
use axum::{extract::State, response::IntoResponse, Json};
use chrono::Utc;
use lsp::git::validate_git_credentials;
use reqwest::Client;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast::error::RecvError;
use tracing::info;
use uuid::Uuid;

use crate::busy::BusyGuard;
use crate::types::{
    AppState, AsyncRequestStatus, AsyncStatus, ProcessBody, ProcessResponse, WebhookPayload,
};
use crate::utils::resolve_repo;
use crate::webhook::{send_with_retries, validate_callback_url_async};

use crate::service::graph_service::{ingest, sync};

async fn validate_webhook_config(body: &ProcessBody) -> Result<(), String> {
    if let Some(url) = body.callback_url.as_deref() {
        validate_callback_url_async(url)
            .await
            .map_err(|e| format!("{}", e))?;
        if std::env::var("WEBHOOK_SECRET").is_err() {
            return Err("callback_url provided but WEBHOOK_SECRET is not configured".to_string());
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum JobKind {
    Sync,
    Ingest,
}

impl JobKind {
    fn label(&self) -> &'static str {
        match self {
            JobKind::Sync => "sync_async",
            JobKind::Ingest => "ingest_async",
        }
    }

    fn timeout_secs(&self) -> u64 {
        match self {
            JobKind::Sync => std::env::var("SYNC_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(30 * 60),
            JobKind::Ingest => std::env::var("INGEST_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(60 * 60),
        }
    }

    fn success_message(&self) -> &'static str {
        match self {
            JobKind::Sync => "Repository sync completed successfully",
            JobKind::Ingest => "Graph building completed successfully",
        }
    }
}

fn error_response(status: StatusCode, msg: String) -> Response {
    (status, Json(serde_json::json!({ "error": msg }))).into_response()
}

fn failed_update(message: String) -> ast::repo::StatusUpdate {
    ast::repo::StatusUpdate {
        status: "Failed".to_string(),
        message,
        step: 0,
        total_steps: ast::repo::TOTAL_STEPS,
        progress: 0,
        stats: None,
        step_description: Some("Failed".to_string()),
    }
}

async fn send_webhook(request_id: &str, url: &str, payload: &WebhookPayload) {
    match validate_callback_url_async(url).await {
        Ok(valid) => {
            let client = Client::new();
            if let Err(e) = send_with_retries(&client, request_id, &valid, payload).await {
                tracing::error!(
                    "Failed to send {} webhook for request {}: {:?}",
                    payload.status,
                    request_id,
                    e
                );
            }
        }
        Err(_) => {
            tracing::warn!(
                "Invalid callback URL for {} webhook, request {}",
                payload.status,
                request_id
            );
        }
    }
}

async fn start_async_job(
    state: Arc<AppState>,
    body: Json<ProcessBody>,
    kind: JobKind,
) -> Response {
    if let Err(e) = validate_webhook_config(&body).await {
        return error_response(StatusCode::BAD_REQUEST, e);
    }

    // Try to acquire the busy lock before creating request_id
    let guard = match BusyGuard::try_new(state.clone()) {
        Some(g) => g,
        None => {
            tracing::warn!("[{}] System busy, rejecting request", kind.label());
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "System is busy processing another request. Please try again later.".to_string(),
            );
        }
    };

    let request_id = Uuid::new_v4().to_string();
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
                update: Some(ast::repo::StatusUpdate {
                    status: "Starting".to_string(),
                    message: "Cloning repository and installing dependencies".to_string(),
                    step: 0,
                    total_steps: ast::repo::TOTAL_STEPS,
                    progress: 0,
                    stats: None,
                    step_description: Some("Initializing".to_string()),
                }),
            },
        );
    }

    let status_map_listener = status_map.clone();
    let request_id_listener = request_id.clone();

    let listener = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(update) => {
                    let mut map = status_map_listener.lock().await;
                    if let Some(status) = map.get_mut(&request_id_listener) {
                        let total_steps = update.total_steps.max(1) as f64;
                        let step = update.step.max(1) as f64;
                        let step_progress = update.progress.min(100) as f64;

                        let overall_progress = (((step - 1.0) + (step_progress / 100.0))
                            / total_steps
                            * 100.0)
                            .min(100.0) as u32;
                        status.progress = overall_progress;

                        if let Some(existing) = &status.update {
                            status.update = Some(ast::repo::StatusUpdate {
                                status: if !update.status.is_empty() {
                                    update.status
                                } else {
                                    existing.status.clone()
                                },
                                message: if !update.message.is_empty() {
                                    update.message
                                } else {
                                    existing.message.clone()
                                },
                                step: update.step,
                                total_steps: update.total_steps,
                                progress: update.progress,
                                stats: update.stats.or_else(|| existing.stats.clone()),
                                step_description: update
                                    .step_description
                                    .or_else(|| existing.step_description.clone()),
                            });
                        } else {
                            status.update = Some(update);
                        }
                    }
                }
                Err(RecvError::Lagged(skipped)) => {
                    tracing::warn!(
                        "Status listener for request {} lagged, skipped {} updates",
                        request_id_listener,
                        skipped
                    );
                }
                Err(RecvError::Closed) => break,
            }
        }
    });

    info!(
        "/{} with Request ID: {} callback_url_present={}",
        kind.label(),
        request_id,
        callback_url.is_some()
    );

    let state_for_task = state.clone();
    let body_clone = body.clone();
    let request_id_task = request_id.clone();

    tokio::spawn(async move {
        // Move guard into task - it will automatically clear busy flag on drop
        let _guard = guard;
        // Hard upper bound so a stuck Neo4j/LSP/IO call surfaces as Err instead of
        // hanging indefinitely waiting on TCP keepalives.
        let timeout_secs = kind.timeout_secs();
        let work = async {
            match kind {
                JobKind::Sync => sync(State(state_for_task.clone()), body_clone).await,
                JobKind::Ingest => ingest(State(state_for_task.clone()), body_clone).await,
            }
        };
        let result = match tokio::time::timeout(Duration::from_secs(timeout_secs), work).await {
            Ok(r) => r,
            Err(_) => Err(crate::types::WebError(shared::Error::internal(format!(
                "{}() timed out after {}s",
                kind.label(),
                timeout_secs
            )))),
        };

        // Stop the listener before writing the final status so it can't
        // overwrite the terminal entry with a stale broadcast.
        listener.abort();

        match result {
            Ok(Json(resp)) => {
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Complete,
                    result: Some(resp.clone()),
                    progress: 100,
                    update: Some(ast::repo::StatusUpdate {
                        status: "Complete".to_string(),
                        message: kind.success_message().to_string(),
                        step: ast::repo::TOTAL_STEPS,
                        total_steps: ast::repo::TOTAL_STEPS,
                        progress: 100,
                        stats: Some(HashMap::from([
                            ("total_nodes".to_string(), resp.nodes as usize),
                            ("total_edges".to_string(), resp.edges as usize),
                        ])),
                        step_description: Some("Complete".to_string()),
                    }),
                };
                status_map
                    .lock()
                    .await
                    .insert(request_id_task.clone(), entry);
                if let Some(url) = &callback_url {
                    let payload = WebhookPayload {
                        request_id: request_id_task.clone(),
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
                    send_webhook(&request_id_task, url, &payload).await;
                }
            }
            Err(e) => {
                let err_msg = format!("{}", e);
                tracing::error!(
                    "[{}] job failed for request {}: {}",
                    kind.label(),
                    request_id_task,
                    err_msg
                );
                let update = failed_update(err_msg.clone());
                let _ = state_for_task.tx.send(update.clone());
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Failed,
                    result: None,
                    progress: 0,
                    update: Some(update),
                };
                status_map
                    .lock()
                    .await
                    .insert(request_id_task.clone(), entry);
                if let Some(url) = &callback_url {
                    let payload = WebhookPayload {
                        request_id: request_id_task.clone(),
                        status: "Failed".to_string(),
                        progress: 0,
                        result: None,
                        error: Some(err_msg),
                        started_at: started_at.to_rfc3339(),
                        completed_at: Utc::now().to_rfc3339(),
                        duration_ms: (Utc::now() - started_at).num_milliseconds().max(0) as u64,
                    };
                    send_webhook(&request_id_task, url, &payload).await;
                }
            }
        }

        // Evict the completed entry after a TTL so the map doesn't grow forever.
        let ttl_secs: u64 = std::env::var("ASYNC_STATUS_TTL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3600);
        tokio::time::sleep(Duration::from_secs(ttl_secs)).await;
        status_map.lock().await.remove(&request_id_task);
    });

    Json(serde_json::json!({ "request_id": request_id })).into_response()
}

#[axum::debug_handler]
pub async fn sync_async(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> impl IntoResponse {
    let (_repo_paths, repo_urls, username, pat, _, _branch) = match resolve_repo(&body) {
        Ok(config) => config,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid repository configuration: {}", e),
            );
        }
    };

    // Enforce single-repo constraint for sync
    if repo_urls.len() > 1 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "sync_async only supports a single repository. Use ingest_async for multiple repositories."
                .to_string(),
        );
    }

    if let Err(e) = validate_git_credentials(&repo_urls[0], username, pat).await {
        return error_response(StatusCode::BAD_REQUEST, format!("{}", e));
    }

    start_async_job(state, body, JobKind::Sync).await
}

#[axum::debug_handler]
pub async fn ingest_async(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> impl IntoResponse {
    let (_repo_paths, repo_urls, username, pat, _, _branch) = match resolve_repo(&body) {
        Ok(config) => config,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid repository configuration: {}", e),
            );
        }
    };

    if let Err(e) = lsp::git::validate_git_credentials_multi(&repo_urls, username, pat).await {
        let err_msg = format!("{}", e);
        let _ = state.tx.send(failed_update(err_msg.clone()));
        return error_response(StatusCode::BAD_REQUEST, err_msg);
    }

    start_async_job(state, body, JobKind::Ingest).await
}
