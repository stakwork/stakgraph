use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug)]
pub struct WebError(pub shared::Error);

pub type AppError = WebError;
pub type Result<T> = std::result::Result<T, AppError>;

#[derive(Serialize)]
struct ErrorResponse {
    message: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessBody {
    pub repo_url: Option<String>,
    pub repo_path: Option<String>,
    pub username: Option<String>,
    pub pat: Option<String>,
    pub use_lsp: Option<bool>,
    pub commit: Option<String>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProcessResponse {
    pub nodes: u32,
    pub edges: u32,
}
#[derive(Serialize, Deserialize)]
pub struct FetchRepoBody {
    pub repo_name: String,
}
#[derive(Serialize, Deserialize)]
pub struct FetchRepoResponse {
    pub status: String,
    pub repo_name: String,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AsyncStatus {
    InProgress,
    Complete,
    Failed(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncRequestStatus {
    pub status: AsyncStatus,
    pub result: Option<ProcessResponse>,
    pub progress: u32,
}

pub type AsyncStatusMap = Arc<Mutex<HashMap<String, AsyncRequestStatus>>>;

impl IntoResponse for WebError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            shared::Error::Io(_)
            | shared::Error::SerdeJson(_)
            | shared::Error::Env(_)
            | shared::Error::Neo4j(_)
            | shared::Error::Recv(_)
            | shared::Error::Lsp(_)
            | shared::Error::Utf8(_)
            | shared::Error::GitUrlParse(_)
            | shared::Error::Git2(_)
            | shared::Error::Walkdir(_)
            | shared::Error::TreeSitterLanguage(_) => StatusCode::INTERNAL_SERVER_ERROR,

            shared::Error::Regex(_) => StatusCode::BAD_REQUEST,
            shared::Error::Custom(msg) => {
                if msg.contains("not found") {
                    StatusCode::NOT_FOUND
                } else {
                    StatusCode::BAD_REQUEST
                }
            }
        };
        tracing::error!("Handler error: {:?}", self.0);
        let resp = ErrorResponse {
            message: self.0.to_string(),
        };
        (status, Json(resp)).into_response()
    }
}

impl From<shared::Error> for WebError {
    fn from(e: shared::Error) -> Self {
        WebError(e)
    }
}
