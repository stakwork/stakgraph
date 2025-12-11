use ast::lang::asg::NodeData;
use ast::repo::StatusUpdate;
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
    pub branch: Option<String>,
    pub callback_url: Option<String>,
    pub realtime: Option<bool>,
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
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncRequestStatus {
    pub status: AsyncStatus,
    pub result: Option<ProcessResponse>,
    pub progress: u32,
    pub update: Option<StatusUpdate>,
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
    pub repo: Option<String>,
    pub ignore_dirs: Option<String>,
    pub regex: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoverageStat {
    pub total: usize,
    pub total_tests: usize,
    pub covered: usize,
    pub percent: f64,
    pub total_lines: usize,
    pub covered_lines: usize,
    pub line_percent: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MockStat {
    pub total: usize,
    pub mocked: usize,
    pub percent: f64,
}

/// Coverage report per test category.
/// unit_tests: unit test nodes that call at least one function.
/// integration_tests: integration test nodes that call any function/resource.
/// e2e_tests: e2e/system tests that exercise endpoints/pages/requests.
/// mocks: 3rd party mock coverage (mocked=true / total).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Coverage {
    pub unit_tests: Option<CoverageStat>,
    pub integration_tests: Option<CoverageStat>,
    pub e2e_tests: Option<CoverageStat>,
    pub mocks: Option<MockStat>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Node {
    pub node_type: String,
    pub ref_id: String,
    pub weight: usize,
    pub test_count: usize,
    pub covered: bool,
    pub properties: NodeData,
    pub body_length: Option<i64>,
    pub line_count: Option<i64>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeConcise {
    pub node_type: String,
    pub name: String,
    pub file: String,
    pub ref_id: String,
    pub weight: usize,
    pub test_count: usize,
    pub covered: bool,
    pub body_length: Option<i64>,
    pub line_count: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verb: Option<String>,
    pub start: usize,
    pub end: usize,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    #[serde(default)]
    pub meta: std::collections::BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum NodesResponseItem {
    Full(Node),
    Concise(NodeConcise),
}

#[derive(Deserialize)]
pub struct QueryNodesParams {
    pub node_type: String,
    pub offset: Option<usize>,
    pub limit: Option<usize>,
    pub sort: Option<String>,
    pub coverage: Option<String>,
    pub concise: Option<bool>,
    pub body_length: Option<bool>,
    pub line_count: Option<bool>,
    pub ignore_dirs: Option<String>,
    pub repo: Option<String>,
    pub regex: Option<String>,
    pub unit_regexes: Option<String>,
    pub integration_regexes: Option<String>,
    pub e2e_regexes: Option<String>,
    pub search: Option<String>,
}

#[derive(Serialize)]
pub struct QueryNodesResponse {
    pub items: Vec<NodesResponseItem>,
    pub total_returned: usize,
    pub total_count: usize,
    pub total_pages: usize,
    pub current_page: usize,
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
