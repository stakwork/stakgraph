use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProcessResponse {
    pub nodes: u32,
    pub edges: u32,
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
