use crate::db::{Database, Person};
use crate::routes::admin_actix::admin_config;
use actix_web::{get, post, web, HttpResponse, Responder};
use serde_json::json;

#[get("/person/{id}")]
async fn get_person(path: web::Path<u32>) -> impl Responder {
    let id = path.into_inner();

    match Database::get_person_by_id(id).await {
        Ok(person) => HttpResponse::Ok().json(person),
        Err(err) => {
            let error_message = err.to_string();
            HttpResponse::InternalServerError().json(json!({ "error": error_message}))
        }
    }
}

#[post("/person")]
async fn create_person(person: web::Json<Person>) -> impl Responder {
    match Database::new_person(person.into_inner()).await {
        Ok(created_person) => HttpResponse::Created().json(created_person),
        Err(err) => {
            let error_message = err.to_string();
            HttpResponse::InternalServerError().json(json!({ "error": error_message}))
        }
    }
}

#[get("/profile")]
async fn get_profile() -> impl Responder {
    HttpResponse::Ok().json(json!({"profile": "data"}))
}

#[post("/profile/update")]
async fn update_profile(data: web::Json<serde_json::Value>) -> impl Responder {
    HttpResponse::Ok().json(json!({"updated": true}))
}

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/api")
            .service(get_person)
            .service(create_person),
    )
    .service(
        web::scope("/user")
            .service(get_profile)
            .service(update_profile),
    )
    .service(web::scope("/admin").configure(admin_config));
}
