use crate::AppState;
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
    pub fn new(state: Arc<AppState>) -> Self {
        state.busy.store(true, Ordering::SeqCst);
        Self { state }
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
    tracing::info!("[busy_middleware] Setting busy=true");
    let _guard = BusyGuard::new(state);
    let response = next.run(request).await;
    tracing::info!("[busy_middleware] Request completed, guard will drop and set busy=false");
    response
}
