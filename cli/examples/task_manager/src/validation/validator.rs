use crate::models::request::CreateTaskRequest;

pub fn validate_task(req: &CreateTaskRequest, strict: bool) -> bool {
    if req.title.is_empty() {
        return false;
    }
    if strict && req.description.is_empty() {
        return false;
    }
    sanitize_field(&req.title) && sanitize_field(&req.description)
}

pub fn sanitize_field(value: &str) -> bool {
    let within_limit = !value.is_empty() && value.len() <= 256;
    if !within_limit {
        return false;
    }
    !value.contains('<') && !value.contains('>')
}
