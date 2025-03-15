use actix_web::{web, HttpResponse, Responder};
use serde_json::json;

use crate::{db::get_db, model::Person};

pub fn config(cfg: &mut web::ServiceConfig) {
    cfg.service(web::resource("/person/{id}").route(web::get().to(get_person)))
        .service(web::resource("/person").route(web::post().to(create_person)));
}

async fn get_person(path: web::Path<u32>) -> impl Responder {
    let id = path.into_inner();
    let db = get_db();

    match db.await.get_person_by_id(id).await {
        Ok(person) => HttpResponse::Ok().json(person),
        Err(err) => {
            let error_message = err.to_string();
            HttpResponse::InternalServerError().json(json!({ "error": error_message}))
        }
    }
}

async fn create_person(person: web::Json<Person>) -> impl Responder {
    let db = get_db();

    match db.await.new_person(person.into_inner()).await {
        Ok(created_person) => HttpResponse::Created().json(created_person),
        Err(err) => {
            let error_message = err.to_string();
            HttpResponse::InternalServerError().json(json!({ "error": error_message}))
        }
    }
}
