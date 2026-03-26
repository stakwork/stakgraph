mod handlers;
mod models;
mod router;
mod db;
mod validation;
mod auth;

use models::request::CreateTaskRequest;

fn main() {
    let req = CreateTaskRequest {
        title: String::from("Write docs"),
        description: String::from("Document the API"),
        priority: models::task::Priority::High,
    };
    let result = handle_request(req);
    println!("{}", result);
}

fn handle_request(req: CreateTaskRequest) -> String {
    if !auth::authenticate("Bearer demo-token") {
        return String::from("unauthorized");
    }
    router::route_request(req)
}
