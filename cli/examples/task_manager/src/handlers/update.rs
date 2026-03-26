use crate::models::request::CreateTaskRequest;
use crate::db::query::db_read;
use crate::validation::validator::validate_task;

pub fn update_task(id: u64, req: CreateTaskRequest) -> String {
    if db_read(id).is_none() {
        return String::from("not found");
    }
    if !validate_task(&req, false) {
        return String::from("validation failed");
    }
    format!("updated task {}", id)
}
