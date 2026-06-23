use std::collections::HashMap;

pub type Scope = Vec<HashMap<String, String>>;

pub fn scope_push(s: &mut Scope) {
    s.push(HashMap::new());
}

pub fn scope_pop(s: &mut Scope) {
    s.pop();
}

pub fn scope_bind(s: &mut Scope, name: &str, type_name: &str) {
    if let Some(frame) = s.last_mut() {
        frame.insert(name.to_string(), type_name.to_string());
    }
}

pub fn scope_lookup<'a>(s: &'a Scope, name: &str) -> Option<&'a str> {
    s.iter().rev().find_map(|f| f.get(name).map(|s| s.as_str()))
}
