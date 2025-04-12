use crate::lang::graphs::{Graph, NodeType};
use crate::lang::{CallsMeta, Edge, Language, NodeData};
use anyhow::{Context, Result};
use lsp::language::PROGRAMMING_LANGUAGES;
use regex::Regex;
use std::path::PathBuf;
use tracing::info;
pub fn link_e2e_tests<G: Graph>(graph: &mut G) -> Result<()> {
    let mut e2e_tests = Vec::new();
    let mut frontend_functions = Vec::new();

    let e2e_test_nodes = graph.find_nodes_by_type(NodeType::E2eTest);
    let function_nodes = graph.find_nodes_by_type(NodeType::Function);

    for node_data in e2e_test_nodes {
        if let Ok(lang) = infer_lang(&node_data) {
            if let Ok(test_ids) = extract_test_ids(&node_data.body, &lang) {
                e2e_tests.push((node_data.clone(), test_ids));
            }
        }
    }

    for node_data in function_nodes {
        if let Ok(lang) = infer_lang(&node_data) {
            if lang.is_frontend() {
                if let Ok(test_ids) = extract_test_ids(&node_data.body, &lang) {
                    frontend_functions.push((node_data.clone(), test_ids));
                }
            }
        }
    }

    let mut i = 0;
    for (t, test_ids) in &e2e_tests {
        for (f, frontend_test_ids) in &frontend_functions {
            for ftestid in frontend_test_ids {
                if test_ids.contains(ftestid) {
                    let edge = Edge::linked_e2e_test_call(t, f);
                    graph.add_edge(edge);
                    i += 1;
                }
            }
        }
    }
    info!("linked {} e2e tests", i);
    Ok(())
}

fn infer_lang(nd: &NodeData) -> Result<Language> {
    for lang in PROGRAMMING_LANGUAGES {
        let pathy = &PathBuf::from(&nd.file);
        let ext = pathy
            .extension()
            .context("no extension")?
            .to_str()
            .context("bad extension")?;
        if lang.exts().contains(&ext) {
            return Ok(lang);
        }
    }
    Err(anyhow::anyhow!("Could not infer language"))
}

fn extract_test_ids(content: &str, lang: &Language) -> Result<Vec<String>> {
    if let None = lang.test_id_regex() {
        return Ok(Vec::new());
    }
    let re = Regex::new(&lang.test_id_regex().unwrap())?;
    let mut test_ids = Vec::new();
    for capture in re.captures_iter(&content) {
        if let Some(test_id) = capture.get(1) {
            test_ids.push(test_id.as_str().to_string());
        }
    }
    Ok(test_ids)
}

pub fn link_api_nodes<G: Graph>(graph: &mut G) -> Result<()> {
    // Collect requests and endpoints in a single pass
    let mut frontend_requests = Vec::new();
    let mut backend_endpoints = Vec::new();

    let request_nodes = graph.find_nodes_by_type(NodeType::Request);
    let endpoint_nodes = graph.find_nodes_by_type(NodeType::Endpoint);

    for node_data in request_nodes {
        if let Some(normalized_path) = normalize_frontend_path(&node_data.name) {
            frontend_requests.push((node_data, normalized_path));
        }
    }

    for node_data in endpoint_nodes {
        if let Some(normalized_path) = normalize_backend_path(&node_data.name) {
            backend_endpoints.push((node_data, normalized_path));
        }
    }

    // Create edges between matching paths and verbs
    let mut i = 0;
    for (req, req_path) in frontend_requests {
        for (endpoint, _) in &backend_endpoints {
            if paths_match(&req_path, &endpoint.name) && verbs_match(&req, endpoint) {
                let edge = Edge::calls(
                    NodeType::Request,
                    &req,
                    NodeType::Endpoint,
                    endpoint,
                    CallsMeta::default(),
                );
                graph.add_edge(edge);
                i += 1;
            }
        }
    }
    info!("linked {} api nodes", i);

    Ok(())
}

pub fn normalize_frontend_path(path: &str) -> Option<String> {
    // Skip paths that are entirely template literals
    if path.starts_with("${") && path.ends_with("}") && !path[2..].contains("${") {
        return None;
    }

    // Extract path part after any leading template prefix (like ${ROOT}/...)
    let path_part = if path.starts_with("${") {
        // Find the end of the first template literal
        if let Some(close_brace) = path.find('}') {
            &path[close_brace + 1..]
        } else {
            return None;
        }
    } else {
        path
    };

    // Replace remaining template expressions like ${var} with :param
    let re = Regex::new(r"\$\{[^}]+\}").ok()?;
    let normalized = re
        .replace_all(path_part, ":param")
        .to_string()
        .trim_start_matches('/')
        .to_string();

    // Ensure the path starts with /
    Some(format!("/{}", normalized))
}

pub fn normalize_backend_path(path: &str) -> Option<String> {
    // Handle various backend parameter formats:
    let re_patterns = [
        // Flask/FastAPI "<type:param>" or "<param>" style - needs to come first
        (Regex::new(r"<[^>]*:?[^>]+>").unwrap(), ":param"),
        // Express/Rails ":param" style
        (Regex::new(r":[^/]+").unwrap(), ":param"),
        // Go/Rust "{param}" style
        (Regex::new(r"\{[^}]+\}").unwrap(), ":param"),
        // Optional parameters
        (Regex::new(r"\([^)]+\)").unwrap(), ":param"),
        // Optional parameters with curly braces
        (Regex::new(r"\{[^}]+\?\}").unwrap(), ":param"),
    ];

    let mut normalized = path.to_string();
    for (re, replacement) in re_patterns.iter() {
        normalized = re.replace_all(&normalized, *replacement).to_string();
    }

    // Remove trailing slashes except for root path
    if normalized.len() > 1 && normalized.ends_with('/') {
        normalized.pop();
    }

    // Ensure the path starts with /
    if !normalized.starts_with('/') {
        return Some(format!("/{}", normalized));
    }

    Some(normalized)
}

fn verbs_match(req: &NodeData, endpoint: &NodeData) -> bool {
    match (req.meta.get("verb"), endpoint.meta.get("verb")) {
        (Some(req_verb), Some(endpoint_verb)) => {
            req_verb.to_uppercase() == endpoint_verb.to_uppercase()
        }
        _ => false,
    }
}

fn paths_match(frontend_path: &str, backend_path: &str) -> bool {
    let frontend_segments: Vec<&str> = frontend_path.split('/').filter(|s| !s.is_empty()).collect();
    let backend_segments: Vec<&str> = backend_path.split('/').filter(|s| !s.is_empty()).collect();

    // If segments length doesn't match, paths don't match
    if frontend_segments.len() != backend_segments.len() {
        return false;
    }

    // Both paths should start with 'api' if either does
    if (frontend_segments.first() == Some(&"api") || backend_segments.first() == Some(&"api"))
        && frontend_segments.first() != backend_segments.first()
    {
        return false;
    }

    frontend_segments
        .iter()
        .zip(backend_segments.iter())
        .all(|(f, b)| {
            f == b || // exact match
            (f.starts_with(':') && !b.starts_with(':')) || // frontend parameter matching concrete backend
            (b.starts_with(':') && !f.starts_with(':')) || // backend parameter matching concrete frontend
            (f.starts_with(':') && b.starts_with(':')) // both are parameters
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::graphs::Node;

    #[test]
    fn test_normalize_frontend_path() {
        assert_eq!(
            normalize_frontend_path("${ROOT}/api/user/${id}"),
            Some("/api/user/:param".to_string())
        );
        assert_eq!(
            normalize_frontend_path("${SOME_CONSTANT}/user/${id}"),
            Some("/user/:param".to_string())
        );
        assert_eq!(normalize_frontend_path("${ENDPOINTS.something}"), None);
    }

    #[test]
    fn test_normalize_backend_path() {
        let test_cases = vec![
            // Express.js/Rails
            ("api/users/:id", "/api/users/:param"),
            ("/users/:userId/posts/:postId", "/users/:param/posts/:param"),
            // Flask/FastAPI
            ("/api/users/<id>", "/api/users/:param"),
            ("/api/users/<int:id>", "/api/users/:param"),
            // Go/Rust
            ("/api/users/{id}", "/api/users/:param"),
            (
                "/users/{userId}/posts/{postId}",
                "/users/:param/posts/:param",
            ),
            // Optional parameters
            ("/api/users/(id)", "/api/users/:param"),
            ("/api/users/{id?}", "/api/users/:param"),
            // Trailing slashes
            ("/api/users/", "/api/users"),
            ("/", "/"),
        ];

        for (input, expected) in test_cases {
            assert_eq!(
                normalize_backend_path(input),
                Some(expected.to_string()),
                "Failed for input: {}",
                input
            );
        }
    }

    #[test]
    fn test_paths_match() {
        assert!(paths_match("/api/user/:param", "/api/user/:id"));
        assert!(paths_match("/api/users/123", "/api/users/:id"));
        assert!(!paths_match("/api/user/:param", "/api/posts/:id"));
        assert!(!paths_match("/user/:param", "/api/user/:id"));
        assert!(!paths_match("/api/user/:param/extra", "/api/user/:id"));
    }

    #[test]
    fn test_link_api_nodes() -> Result<()> {
        use crate::lang::graphs::ArrayGraph;
        let mut graph = ArrayGraph::new();

        // Valid matching pair
        let mut req1 = NodeData::name_file("api/user/${id}", "src/components/User.tsx");
        req1.meta.insert("verb".to_string(), "GET".to_string());

        let mut endpoint1 = NodeData::name_file("/api/user/:id", "src/routes/user.ts");
        endpoint1.meta.insert("verb".to_string(), "GET".to_string());

        // Non-matching pair (different verbs)
        let mut req2 = NodeData::name_file("/api/posts/${id}", "src/components/Post.tsx");
        req2.meta.insert("verb".to_string(), "POST".to_string());

        let mut endpoint2 = NodeData::name_file("/api/posts/:id", "src/routes/posts.ts");
        endpoint2.meta.insert("verb".to_string(), "GET".to_string());

        // Add nodes to graph
        graph.nodes.push(Node::new(NodeType::Request, req1));
        graph.nodes.push(Node::new(NodeType::Request, req2));
        graph.nodes.push(Node::new(NodeType::Endpoint, endpoint1));
        graph.nodes.push(Node::new(NodeType::Endpoint, endpoint2));

        link_api_nodes(&mut graph)?;

        // Should only create one edge for the matching pair
        assert_eq!(graph.edges.len(), 1);

        Ok(())
    }
}
