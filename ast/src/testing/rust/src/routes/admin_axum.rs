// @ast node: Endpoint "/admin/users" [verb=GET]
// @ast edge: Handler -> Function "list_users" "admin_axum.rs"
// @ast node: Endpoint "/admin/users/:id" [verb=DELETE]
// @ast edge: Handler -> Function "delete_user" "admin_axum.rs"
// @ast node: Function "admin_router"
// @ast node: Function "list_users"
// @ast node: Function "delete_user"
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
