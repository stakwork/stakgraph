use rocket::http::Status;
use rocket::response::status;
use rocket::{get, post, routes, serde::json::Json};
use serde_json::json;

use crate::db::{Database, Person};
use crate::routes::admin_rocket::{delete_user, list_users};

// Example for rocket_routes.rs
#[get("/person/<id>")]
pub async fn get_person(id: u32) -> Result<Json<Person>, status::Custom<Json<serde_json::Value>>> {
    match Database::get_person_by_id(id).await {
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
    match Database::new_person(person.into_inner()).await {
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

#[get("/profile")]
pub async fn get_profile() -> Json<serde_json::Value> {
    Json(json!({"profile": "user"}))
}

#[post("/profile/update")]
pub async fn update_profile(data: Json<serde_json::Value>) -> Json<serde_json::Value> {
    Json(json!({"updated": true}))
}

pub fn create_rocket() -> rocket::Rocket<rocket::Build> {
    rocket::build()
        .mount("/", routes![get_person, create_person])
        .mount("/user", routes![get_profile, update_profile])
        .mount("/admin", routes![list_users, delete_user])
}
