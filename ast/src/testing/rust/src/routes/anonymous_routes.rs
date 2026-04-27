// @ast node: Endpoint "/anon-get" [verb=GET]
// @ast node: Endpoint "/anon-post" [verb=POST]
// @ast node: Endpoint "/anon-move" [verb=GET]
// @ast node: Function "create_anonymous_router"
// @ast node: Function "GET_anon-get_closure_L16"
// @ast node: Function "POST_anon-post_closure_L20"
// @ast node: Function "GET_anon-move_closure_L23"
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
