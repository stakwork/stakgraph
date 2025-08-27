use ast::lang::asg::NodeData;
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
    pub callback_url: Option<String>,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProcessResponse {
    pub nodes: u32,
    pub edges: u32,
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WebhookPayload {
    pub request_id: String,
    pub status: String,
    pub progress: u32,
    pub result: Option<ProcessResponse>,
    pub error: Option<String>,
    pub started_at: String,
    pub completed_at: String,
    pub duration_ms: u64,
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

#[derive(Deserialize)]
pub struct EmbedCodeParams {
    pub files: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorSearchResult {
    pub node: NodeData,
    pub score: f64,
}

#[derive(Deserialize)]
pub struct VectorSearchParams {
    pub query: String,
    pub limit: Option<usize>,
    pub node_types: Option<String>,
    pub similarity_threshold: Option<f32>,
    pub language: Option<String>,
}

#[derive(Deserialize)]
pub struct CoverageParams {
    pub root: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoverageStat {
    pub total: usize,
    pub covered: usize,
    pub percent: f64,
}

/// Coverage report per test category.
/// unit_tests: unit test nodes that call at least one function.
/// integration_tests: integration test nodes that call any function/resource.
/// e2e_tests: e2e/system tests that exercise endpoints/pages/requests.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Coverage {
    pub unit_tests: Option<CoverageStat>,
    pub integration_tests: Option<CoverageStat>,
    pub e2e_tests: Option<CoverageStat>,
}

#[derive(Deserialize)]
pub struct UncoveredParams {
    pub node_type: String,
    pub limit: Option<usize>,
    pub sort: Option<String>,
    pub root: Option<String>,
    pub concise: Option<bool>,
    pub tests: Option<String>,
    pub output: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UncoveredNode {
    pub node_type: String,
    pub ref_id: String,
    pub weight: usize,
    pub properties: NodeData,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UncoveredNodeConcise {
    pub name: String,
    pub file: String,
    pub weight: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum UncoveredResponseItem {
    Full(UncoveredNode),
    Concise(UncoveredNodeConcise),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UncoveredResponse {
    pub functions: Option<Vec<UncoveredResponseItem>>,
    pub endpoints: Option<Vec<UncoveredResponseItem>>,
}

#[derive(Deserialize)]
pub struct HasParams {
    pub node_type: String,
    pub name: String,
    pub file: String,
    pub start: Option<usize>,
    pub root: Option<String>,
    pub tests: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HasResponse {
    pub covered: bool,
}

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
            | shared::Error::Other(_)
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
