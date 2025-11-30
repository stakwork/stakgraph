use actix_web::{delete, get, web, HttpResponse, Responder};
use serde_json::json;

#[get("/users")]
async fn list_users() -> impl Responder {
    HttpResponse::Ok().json(json!({"users": []}))
}

#[delete("/users/{id}")]
async fn delete_user(path: web::Path<u32>) -> impl Responder {
    let id = path.into_inner();
    HttpResponse::Ok().json(json!({"deleted": id}))
}

pub fn admin_config(cfg: &mut web::ServiceConfig) {
    cfg.service(list_users).service(delete_user);
}
