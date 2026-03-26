use crate::models::request::CreateTaskRequest;
use crate::handlers::{create, read, update, delete};

pub fn route_request(req: CreateTaskRequest) -> String {
    match req.priority {
        crate::models::task::Priority::High => create::create_task(req),
        crate::models::task::Priority::Medium => {
            let task = read::get_task(1);
            task.map(|t| t.title).unwrap_or_default()
        }
        crate::models::task::Priority::Low => {
            if req.title == "delete" {
                delete::delete_task(1)
            } else {
                update::update_task(1, req)
            }
        }
    }
}
