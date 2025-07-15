use super::AppState;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::Response;
use axum::Json;
use base64::prelude::*;
use serde_json::json;
use std::sync::Arc;

// Bearer token authentication middleware for JSON API routes
pub async fn bearer_auth(
    State(app_state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> std::result::Result<Response, StatusCode> {
    // If no API token is configured, allow all requests
    let Some(expected_token) = &app_state.api_token else {
        return Ok(next.run(request).await);
    };

    let auth_header = headers
        .get("authorization")
        .and_then(|header| header.to_str().ok());

    if let Some(auth_header) = auth_header {
        if auth_header.starts_with("Bearer ") {
            let token = &auth_header[7..]; // Remove "Bearer " prefix
            if token == expected_token {
                return Ok(next.run(request).await);
            }
        }
    }

    let api_token_header = headers
        .get("x-api-token")
        .and_then(|header| header.to_str().ok());

    if let Some(token) = api_token_header {
        if token == expected_token {
            return Ok(next.run(request).await);
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}

// Basic auth middleware for static files
pub async fn basic_auth(
    State(app_state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> std::result::Result<Response, (StatusCode, HeaderMap)> {
    // If no API token is configured, allow all requests
    let Some(expected_token) = &app_state.api_token else {
        return Ok(next.run(request).await);
    };

    let auth_header = headers
        .get("authorization")
        .and_then(|header| header.to_str().ok());

    if let Some(auth_header) = auth_header {
        if auth_header.starts_with("Basic ") {
            let encoded = &auth_header[6..]; // Remove "Basic " prefix
            if let Ok(decoded) = BASE64_STANDARD.decode(encoded) {
                if let Ok(credentials) = String::from_utf8(decoded) {
                    // Expected format: "username:password" or just ":token"
                    let parts: Vec<&str> = credentials.splitn(2, ':').collect();
                    if parts.len() == 2 {
                        // Check if either username or password matches the token
                        if parts[0] == expected_token || parts[1] == expected_token {
                            return Ok(next.run(request).await);
                        }
                    }
                }
            }
        }
    }

    let mut headers = HeaderMap::new();
    headers.insert(
        "WWW-Authenticate",
        "Basic realm=\"Restricted\"".parse().unwrap(),
    );
    Err((StatusCode::UNAUTHORIZED, headers))
}

pub async fn token_exchange(
    State(app_state): State<Arc<AppState>>,
) -> std::result::Result<Json<serde_json::Value>, StatusCode> {
    // If no API token is configured, return an error
    let Some(expected_token) = &app_state.api_token else {
        return Ok(Json(json!({
            "message": "No API token configured"
        })));
    };
    // Since they already authenticated with basic auth to get the HTML,
    // we can provide the token without additional auth
    Ok(Json(json!({
        "message": "Token exchange successful",
        "token": expected_token,
        "type": "bearer"
    })))
}
