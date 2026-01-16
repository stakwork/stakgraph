use axum::{extract::State, response::IntoResponse, Json};
use chrono::Utc;
use lsp::git::validate_git_credentials;
use reqwest::Client;
use std::sync::Arc;
use tracing::info;

use crate::types::{
    AppState, AsyncRequestStatus, AsyncStatus, ProcessBody, ProcessResponse, WebError,
    WebhookPayload,
};
use crate::utils::{call_mcp_docs, call_mcp_mocks, resolve_repo};
use crate::webhook::{send_with_retries, validate_callback_url_async};

use crate::service::graph_service::{ingest, sync};

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

    // Try to acquire the busy lock before creating request_id
    let guard = match crate::busy::BusyGuard::try_new(state.clone()) {
        Some(g) => g,
        None => {
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
        "/sync with Request ID: {} and callback_url:  {:?}",
        request_id_for_work, callback_url
    );

    let state_for_process = state.clone();

    tokio::spawn(async move {
        // Move guard into task - it will automatically clear busy flag on drop
        let _guard = guard;
        let result = sync(State(state_for_process.clone()), body_clone).await;

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

                // Call mocks discovery in sync mode after process completes
                call_mcp_docs(&repo_url, true).await;
                call_mcp_mocks(&repo_url, username.as_deref(), pat.as_deref(), true).await;

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

                        let _ = crate::webhook::send_with_retries(
                            &client,
                            &request_id_for_work.to_string(),
                            &valid,
                            &payload,
                        )
                        .await;
                    }
                }
            }
            Err(e) => {
                let entry = AsyncRequestStatus {
                    status: AsyncStatus::Failed,
                    result: None,
                    progress: 0,
                    update: Some(ast::repo::StatusUpdate {
                        status: "Failed".to_string(),
                        message: format!("Error: {:?}", e),
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
                            error: Some(format!("{:?}", e)),
                            started_at: started_at.to_rfc3339(),
                            completed_at: Utc::now().to_rfc3339(),
                            duration_ms: (Utc::now() - started_at).num_milliseconds().max(0) as u64,
                        };
                        let client = Client::new();
                        let _ = crate::webhook::send_with_retries(
                            &client,
                            &request_id_for_work.to_string(),
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

    // Try to acquire the busy lock before creating request_id
    let guard = match crate::busy::BusyGuard::try_new(state.clone()) {
        Some(g) => g,
        None => {
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
        let result = ingest(State(state_clone.clone()), body_clone).await;

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
                    status: AsyncStatus::Failed,
                    result: None,
                    progress: 0,
                    update: Some(ast::repo::StatusUpdate {
                        status: "Failed".to_string(),
                        message: format!("Error: {:?}", e),
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
