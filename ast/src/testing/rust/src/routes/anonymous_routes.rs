use axum::{
    routing::{get, post},
    Router,
};

/// Anonymous closure handlers for testing
pub fn create_anonymous_router() -> Router {
    Router::new()
        // Simple closure
        .route("/anon-get", get(|_| async { "Anonymous GET" }))
        // Closure with args
        .route(
            "/anon-post",
            post(|body: String| async move { format!("Received: {}", body) }),
        )
        // Move closure
        .route("/anon-move", get(move |_| async move { "Move closure" }))
}
