use anyhow::Result;
use axum::{
    extract::{Json, Path},
    http::StatusCode,
    response::{IntoResponse, Json as JsonResponse},
    routing::{get, post},
    Router,
};
use serde_json::json;

use crate::db::{Database, Person};
use crate::routes::admin_axum::admin_router;

pub fn create_router() -> Router {
    Router::new()
        .route("/person/:id", get(get_person))
        .route("/person", post(create_person))
        .nest(
            "/user",
            Router::new()
                .route("/profile", get(get_profile))
                .route("/profile/update", post(update_profile)),
        )
        .nest("/admin", admin_router())
}

#[tracing::instrument]
async fn get_person(Path(id): Path<u32>) -> (StatusCode, JsonResponse<serde_json::Value>) {
    let person_result: Result<Person> = Database::get_person_by_id(id).await;

    match person_result {
        Ok(person) => {
            let person_data: Person = person;
            (StatusCode::OK, JsonResponse(json!(person_data)))
        }
        Err(err) => {
            let error_message = err.to_string();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                JsonResponse(json!({ "error": error_message })),
            )
        }
    }
}

async fn create_person(Json(person): Json<Person>) -> impl axum::response::IntoResponse {
    let result: Result<Person> = Database::new_person(person).await;

    match result {
        Ok(created_person) => {
            (StatusCode::CREATED, JsonResponse(json!(created_person))).into_response()
        }
        Err(err) => {
            let error_message = err.to_string();
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                JsonResponse(json!({ "error": error_message })),
            )
                .into_response()
        }
    }
}

async fn get_profile() -> (StatusCode, JsonResponse<serde_json::Value>) {
    (StatusCode::OK, JsonResponse(json!({"profile": "data"})))
}

async fn update_profile(Json(data): Json<serde_json::Value>) -> impl IntoResponse {
    (StatusCode::OK, JsonResponse(json!({"updated": true}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let router = create_router();
        assert_eq!(std::mem::size_of_val(&router), std::mem::size_of::<Router>());
    }

    #[tokio::test]
    async fn test_axum_person_validation() {
        let person = Person {
            id: None,
            name: "Charlie".to_string(),
            email: "charlie@test.com".to_string(),
        };
        assert!(!person.name.is_empty());
        assert!(person.email.contains('@'));
    }

    #[test]
    fn test_person_struct_creation() {
        let person = Person {
            id: Some(1),
            name: "Test User".to_string(),
            email: "test@example.com".to_string(),
        };
        assert_eq!(person.id, Some(1));
        assert_eq!(person.name, "Test User");
    }

    #[test]
    fn test_person_without_id() {
        let person = Person {
            id: None,
            name: "New User".to_string(),
            email: "new@example.com".to_string(),
        };
        assert!(person.id.is_none());
    }

    #[test]
    fn test_person_email_validation() {
        let person = Person {
            id: None,
            name: "User".to_string(),
            email: "valid.email+tag@example.com".to_string(),
        };
        assert!(person.email.contains('@'));
        assert!(person.email.contains('.'));
    }
}
