use anyhow::Result;
use axum::{
    extract::Path,
    http::StatusCode,
    response::Json,
    routing::{delete, get},
    Router,
};
use serde_json::json;

pub fn admin_router() -> Router {
    Router::new()
        .route("/users", get(list_users))
        .route("/users/:id", delete(delete_user))
}

async fn list_users() -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(json!({"users": []})))
}

async fn delete_user(Path(id): Path<u32>) -> (StatusCode, Json<serde_json::Value>) {
    (StatusCode::OK, Json(json!({"deleted": id})))
}
