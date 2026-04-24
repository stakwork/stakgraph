// @ast node: Endpoint "/admin/users" [verb=GET]
// @ast edge: Handler -> Function "list_users" "admin_rocket.rs"
// @ast node: Endpoint "/admin/users/<id>" [verb=DELETE]
// @ast edge: Handler -> Function "delete_user" "admin_rocket.rs"
// @ast node: Function "list_users"
// @ast node: Function "delete_user"
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
