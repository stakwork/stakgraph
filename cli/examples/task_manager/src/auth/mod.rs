pub mod middleware;

use middleware::check_token;

pub fn authenticate(token: &str) -> bool {
    check_token(token)
}

pub fn authorize(token: &str, resource: &str) -> bool {
    if !authenticate(token) {
        return false;
    }
    !resource.is_empty()
}
