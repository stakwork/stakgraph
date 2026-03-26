use crate::models::request::CreateTaskRequest;
use crate::models::task::{Task, TaskStatus};
use crate::validation::validator::validate_task;
use crate::db::write::db_write;

pub fn create_task(req: CreateTaskRequest) -> String {
    if !validate_task(&req, true) {
        return String::from("validation failed");
    }
    let task = Task {
        id: 0,
        title: req.title.clone(),
        description: req.description.clone(),
        priority: req.priority,
        status: TaskStatus::Pending,
    };
    db_write(task);
    format!("created: {}", req.title)
}
