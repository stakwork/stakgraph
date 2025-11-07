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
    let _guard = BusyGuard::new(state);
    next.run(request).await
}
