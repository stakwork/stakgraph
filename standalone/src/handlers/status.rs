use crate::types::AppState;
use axum::{extract::{Path, State}, response::{IntoResponse, sse::{Event, KeepAlive, Sse}}, Json, http::StatusCode};
use std::{sync::Arc, convert::Infallible, time::Duration};
use futures::stream;
use tokio::sync::broadcast::error::RecvError;


#[axum::debug_handler]
pub async fn busy_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    use std::sync::atomic::Ordering;
    let busy = state.busy.load(Ordering::SeqCst);
    tracing::debug!("[busy_handler] Returning busy={}", busy);
    Json(serde_json::json!({ "busy": busy }))
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
            Json(serde_json::json!({
                "error": format!("Request ID {} not found", request_id)
            })),
        )
            .into_response()
    }
}

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
