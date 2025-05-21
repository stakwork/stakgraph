use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, AppError>;

#[derive(Serialize, Deserialize)]
pub struct ProcessBody {
    pub repo_url: String,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub pat: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ProcessResponse {
    pub status: String,
    pub message: String,
    pub nodes: usize,
    pub edges: usize,
}

#[derive(Debug)]
pub enum AppError {
    Anyhow(anyhow::Error),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        match self {
            AppError::Anyhow(err) => (StatusCode::BAD_REQUEST, err.to_string()).into_response(),
        }
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self::Anyhow(err.into())
    }
}
