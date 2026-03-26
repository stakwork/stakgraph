pub fn check_token(token: &str) -> bool {
    validate_bearer(token)
}

pub fn validate_bearer(token: &str) -> bool {
    token.starts_with("Bearer ") && token.len() > 7
}
