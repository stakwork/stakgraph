use crate::types::AppState;
use axum::response::IntoResponse;
use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use std::sync::{atomic::Ordering, Arc};

pub struct BusyGuard {
    state: Arc<AppState>,
}

impl BusyGuard {
    // pub fn new(state: Arc<AppState>) -> Self {
    //     state.busy.store(true, Ordering::SeqCst);
    //     Self { state }
    // }

    /// Attempts to acquire the busy lock atomically.
    /// Returns Some(BusyGuard) if successful, None if already busy.
    pub fn try_new(state: Arc<AppState>) -> Option<Self> {
        match state
            .busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => Some(Self { state }),
            Err(_) => None,
        }
    }
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.state.busy.store(false, Ordering::SeqCst);
    }
}

pub async fn busy_middleware(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let _guard = match BusyGuard::try_new(state) {
        Some(guard) => {
            tracing::info!("[busy_middleware] Acquired busy lock");
            guard
        }
        None => {
            tracing::warn!("[busy_middleware] System already busy, rejecting request");
            return (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                axum::Json(serde_json::json!({
                    "error": "System is busy processing another request. Please try again later."
                })),
            )
                .into_response();
        }
    };
    let response = next.run(request).await;
    tracing::info!("[busy_middleware] Request completed, guard will drop and set busy=false");
    response
}
