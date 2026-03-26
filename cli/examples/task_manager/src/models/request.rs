use crate::models::task::Priority;

pub struct CreateTaskRequest {
    pub title: String,
    pub description: String,
    pub priority: Priority,
}

pub struct UpdateTaskRequest {
    pub title: Option<String>,
    pub description: Option<String>,
    pub priority: Option<Priority>,
}
