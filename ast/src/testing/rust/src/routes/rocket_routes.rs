use rocket::http::Status;
use rocket::response::status;
use rocket::{get, post, routes, serde::json::Json};
use serde_json::json;

use crate::{db::get_db, model::Person};

#[get("/person/<id>")]
pub async fn get_person(id: u32) -> Result<Json<Person>, status::Custom<Json<serde_json::Value>>> {
    let db = get_db().await;

    match db.get_person_by_id(id).await {
        Ok(person) => Ok(Json(person)),
        Err(err) => {
            let error_message = err.to_string();
            Err(status::Custom(
                Status::InternalServerError,
                Json(json!({ "error": error_message })),
            ))
        }
    }
}

#[post("/person", data = "<person>")]
pub async fn create_person(
    person: Json<Person>,
) -> Result<Json<Person>, status::Custom<Json<serde_json::Value>>> {
    let db = get_db().await;

    match db.new_person(person.into_inner()).await {
        Ok(person) => Ok(Json(person)),
        Err(err) => {
            let error_message = err.to_string();
            Err(status::Custom(
                Status::InternalServerError,
                Json(json!({ "error": error_message })),
            ))
        }
    }
}

pub fn create_rocket() -> rocket::Rocket<rocket::Build> {
    rocket::build().mount("/", routes![get_person, create_person])
}
