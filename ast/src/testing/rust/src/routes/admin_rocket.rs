use rocket::{delete, get, serde::json::Json};
use serde_json::json;

#[get("/users")]
pub async fn list_users() -> Json<serde_json::Value> {
    Json(json!({"users": []}))
}

#[delete("/users/<id>")]
pub async fn delete_user(id: u32) -> Json<serde_json::Value> {
    Json(json!({"deleted": id}))
}
