use axum::{extract::State, response::IntoResponse, Json};
use chrono::Utc;
use lsp::git::validate_git_credentials;
use reqwest::Client;
use std::sync::Arc;
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

#[axum::debug_handler]
pub async fn sync_async(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> impl IntoResponse {
    let (_repo_paths, repo_urls, username, pat, _, _branch) = match resolve_repo(&body) {
        Ok(config) => config,
        Err(e) => {
            return Json(serde_json::json!({
                "error": format!("Invalid repository configuration: {}", e)
            }))
            .into_response();
        }
    };

    // Enforce single-repo constraint for sync
    if repo_urls.len() > 1 {
        return Json(serde_json::json!({
            "error": "sync_async only supports a single repository. Use ingest_async for multiple repositories."
        }))
        .into_response();
    }

    let repo_url = repo_urls[0].clone();

    if let Err(e) = validate_git_credentials(&repo_url, username.clone(), pat.clone()).await {
        return Json(serde_json::json!({
            "error": format!("{}", e)
        }))
        .into_response();
    }

    if let Err(e) = validate_webhook_config(&body).await {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e })),
        )
            .into_response();
    }

    // Try to acquire the busy lock before creating request_id
    let guard = match BusyGuard::try_new(state.clone()) {
        Some(g) => g,
        None => {
            tracing::warn!(
                "[sync_async] System busy, rejecting request for repo_url={}",
                repo_url
            );
            return Json(serde_json::json!({
                "error": "System is busy processing another request. Please try again later."
            }))
            .into_response();
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
                    total_steps: 16,
                    progress: 0,
                    stats: None,
                    step_description: Some("Initializing".to_string()),
                }),
            },
        );
    }

    let status_map_clone = status_map.clone();
    let body_clone = body.clone();
    let request_id_for_listener = request_id.clone();
    let request_id_for_work = request_id.clone();

    tokio::spawn(async move {
        while let Ok(update) = rx.recv().await {
            let mut map = status_map_clone.lock().await;
            if let Some(status) = map.get_mut(&request_id_for_listener) {
                let total_steps = update.total_steps.max(1) as f64;
                let step = update.step.max(1) as f64;
                let step_progress = update.progress.min(100) as f64;

                let overall_progress = (((step - 1.0) + (step_progress / 100.0)) / total_steps
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
    });

    info!(
        "/sync with Request ID: {} callback_url_present={}",
        request_id_for_work,
        callback_url.is_some()
    );

    let state_for_process = state.clone();

    tokio::spawn(async move {
        // Move guard into task - it will automatically clear busy flag on drop
        let _guard = guard;
        // Hard upper bound so a stuck Neo4j/LSP/IO call surfaces as Err instead of
        // hanging indefinitely waiting on TCP keepalives.
        let timeout_secs: u64 = std::env::var("SYNC_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(30 * 60); // 30 minutes default
        let result = match tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            sync(State(state_for_process.clone()), body_clone),
        )
        .await
        {
            Ok(r) => r,
            Err(_) => Err(crate::types::WebError(shared::Error::internal(
                format!("sync() timed out after {}s", timeout_secs),
            ))),
        };

        let mut map = status_map.lock().await;

        match result {
            Ok(Json(resp)) => {
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Complete,
                    result: Some(resp.clone()),
                    progress: 100,
                    update: Some(ast::repo::StatusUpdate {
                        status: "Complete".to_string(),
                        message: "Repository sync completed successfully".to_string(),
                        step: 16,
                        total_steps: 16,
                        progress: 100,
                        stats: Some(std::collections::HashMap::from([
                            ("total_nodes".to_string(), resp.nodes as usize),
                            ("total_edges".to_string(), resp.edges as usize),
                        ])),
                        step_description: Some("Complete".to_string()),
                    }),
                };
                map.insert(request_id_for_work.clone(), entry);

                if let Some(url) = callback_url.clone() {
                    if let Ok(valid) = crate::webhook::validate_callback_url_async(&url).await {
                        let payload = WebhookPayload {
                            request_id: request_id_for_work.to_string(),
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

                        if let Err(webhook_err) = crate::webhook::send_with_retries(
                            &client,
                            &request_id_for_work.to_string(),
                            &valid,
                            &payload,
                        )
                        .await
                        {
                            tracing::error!(
                                "Failed to send sync_async completion webhook for request {}: {:?}",
                                request_id_for_work,
                                webhook_err
                            );
                        }
                    } else {
                        tracing::warn!(
                            "Invalid callback URL for sync_async completion webhook, request {}",
                            request_id_for_work
                        );
                    }
                }
            }
            Err(e) => {
                let err_msg = format!("{}", e);
                tracing::error!(
                    "[sync_async] sync() failed for request {}: {}",
                    request_id_for_work,
                    err_msg
                );
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Failed,
                    result: None,
                    progress: 0,
                    update: Some(ast::repo::StatusUpdate {
                        status: "Failed".to_string(),
                        message: err_msg.clone(),
                        step: 0,
                        total_steps: 16,
                        progress: 0,
                        stats: None,
                        step_description: Some("Failed".to_string()),
                    }),
                };
                map.insert(request_id_for_work.clone(), entry);
                if let Some(url) = callback_url.clone() {
                    if let Ok(valid) = crate::webhook::validate_callback_url_async(&url).await {
                        let payload = WebhookPayload {
                            request_id: request_id_for_work.to_string(),
                            status: "Failed".to_string(),
                            progress: 0,
                            result: None,
                            error: Some(err_msg.clone()),
                            started_at: started_at.to_rfc3339(),
                            completed_at: Utc::now().to_rfc3339(),
                            duration_ms: (Utc::now() - started_at).num_milliseconds().max(0) as u64,
                        };
                        let client = Client::new();
                        if let Err(webhook_err) = crate::webhook::send_with_retries(
                            &client,
                            &request_id_for_work.to_string(),
                            &valid,
                            &payload,
                        )
                        .await
                        {
                            tracing::error!(
                                "Failed to send sync_async failure webhook for request {}: {:?}",
                                request_id_for_work,
                                webhook_err
                            );
                        }
                    } else {
                        tracing::warn!(
                            "Invalid callback URL for sync_async failure webhook, request {}",
                            request_id_for_work
                        );
                    }
                }
            }
        }
    });

    Json(serde_json::json!({ "request_id": request_id })).into_response()
}

#[axum::debug_handler]
pub async fn ingest_async(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> impl IntoResponse {
    let (_repo_paths, repo_urls, username, pat, _, _branch) = match resolve_repo(&body) {
        Ok(config) => config,
        Err(e) => {
            return Json(serde_json::json!({
                "error": format!("Invalid repository configuration: {}", e)
            }))
            .into_response();
        }
    };

    if let Err(e) =
        lsp::git::validate_git_credentials_multi(&repo_urls, username.clone(), pat.clone()).await
    {
        let err_msg = format!("{}", e);
        let _ = state.tx.send(ast::repo::StatusUpdate {
            status: "Failed".to_string(),
            message: err_msg.clone(),
            step: 0,
            total_steps: 16,
            progress: 0,
            stats: None,
            step_description: Some("Failed".to_string()),
        });
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": err_msg })),
        )
            .into_response();
    }

    if let Err(e) = validate_webhook_config(&body).await {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e })),
        )
            .into_response();
    }

    // Try to acquire the busy lock before creating request_id
    let guard = match crate::busy::BusyGuard::try_new(state.clone()) {
        Some(g) => g,
        None => {
            tracing::warn!(
                "[ingest_async] System busy, rejecting request for repo_urls={:?}",
                repo_urls
            );
            return Json(serde_json::json!({
                "error": "System is busy processing another request. Please try again later."
            }))
            .into_response();
        }
    };

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
                update: Some(ast::repo::StatusUpdate {
                    status: "Starting".to_string(),
                    message: "Cloning repository and installing dependencies".to_string(),
                    step: 0,
                    total_steps: 16,
                    progress: 0,
                    stats: None,
                    step_description: Some("Initializing".to_string()),
                }),
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
    });

    let request_id_clone = request_id.clone();

    //run ingest as a background task
    tokio::spawn(async move {
        // Move guard into task - it will automatically clear busy flag on drop
        let _guard = guard;
        // Hard upper bound so a stuck Neo4j/LSP/IO call surfaces as Err instead of
        // hanging indefinitely waiting on TCP keepalives.
        let timeout_secs: u64 = std::env::var("INGEST_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60 * 60); // 60 minutes default
        let result = match tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            ingest(State(state_clone.clone()), body_clone),
        )
        .await
        {
            Ok(r) => r,
            Err(_) => Err(crate::types::WebError(shared::Error::internal(format!(
                "ingest() timed out after {}s",
                timeout_secs
            )))),
        };

        let mut map = status_map.lock().await;

        match result {
            Ok(Json(resp)) => {
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Complete,
                    result: Some(resp.clone()),
                    progress: 100,
                    update: Some(ast::repo::StatusUpdate {
                        status: "Complete".to_string(),
                        message: "Graph building completed successfully".to_string(),
                        step: 16,
                        total_steps: 16,
                        progress: 100,
                        stats: Some(std::collections::HashMap::from([
                            ("total_nodes".to_string(), resp.nodes as usize),
                            ("total_edges".to_string(), resp.edges as usize),
                        ])),
                        step_description: Some("Complete".to_string()),
                    }),
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
                        if let Err(webhook_err) =
                            send_with_retries(&client, &request_id_clone, &valid, &payload).await
                        {
                            tracing::error!(
                                "Failed to send ingest_async completion webhook for request {}: {:?}",
                                request_id_clone,
                                webhook_err
                            );
                        }
                    } else {
                        tracing::warn!(
                            "Invalid callback URL for ingest_async completion webhook, request {}",
                            request_id_clone
                        );
                    }
                }
            }
            Err(e) => {
                let err_msg = format!("{}", e);
                tracing::error!(
                    "[ingest_async] ingest() failed for request {}: {}",
                    request_id_clone,
                    err_msg
                );
                let _ = state_clone.tx.send(ast::repo::StatusUpdate {
                    status: "Failed".to_string(),
                    message: err_msg.clone(),
                    step: 0,
                    total_steps: 16,
                    progress: 0,
                    stats: None,
                    step_description: Some("Failed".to_string()),
                });
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Failed,
                    result: None,
                    progress: 0,
                    update: Some(ast::repo::StatusUpdate {
                        status: "Failed".to_string(),
                        message: err_msg.clone(),
                        step: 0,
                        total_steps: 16,
                        progress: 0,
                        stats: None,
                        step_description: Some("Failed".to_string()),
                    }),
                };
                map.insert(request_id_clone.clone(), entry);
                if let Some(url) = callback_url {
                    if let Ok(valid) = validate_callback_url_async(&url).await {
                        let payload = WebhookPayload {
                            request_id: request_id_clone.clone(),
                            status: "Failed".to_string(),
                            progress: 0,
                            result: None,
                            error: Some(err_msg.clone()),
                            started_at: started_at.to_rfc3339(),
                            completed_at: Utc::now().to_rfc3339(),
                            duration_ms: (Utc::now() - started_at).num_milliseconds().max(0) as u64,
                        };
                        let client = Client::new();
                        if let Err(webhook_err) =
                            send_with_retries(&client, &request_id_clone, &valid, &payload).await
                        {
                            tracing::error!(
                                "Failed to send ingest_async failure webhook for request {}: {:?}",
                                request_id_clone,
                                webhook_err
                            );
                        }
                    } else {
                        tracing::warn!(
                            "Invalid callback URL for ingest_async failure webhook, request {}",
                            request_id_clone
                        );
                    }
                }
            }
        }
    });

    Json(serde_json::json!({ "request_id": request_id })).into_response()
}
