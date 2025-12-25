use crate::lang::graphs::{EdgeType, Graph, NodeType};
use crate::lang::{Edge, Language, NodeData};
use lsp::language::PROGRAMMING_LANGUAGES;
use regex::Regex;
use shared::{Context, Error, Result};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use tracing::{debug, info, instrument};

type EdgeKey = (String, String, usize);
type EdgeIndex = HashMap<EdgeKey, HashSet<(EdgeKey, EdgeType)>>;

#[instrument(skip(edges))]
fn build_edge_index(edges: &[Edge]) -> EdgeIndex {
    let mut index: EdgeIndex = HashMap::new();
    for edge in edges {
        let source_key = (
            edge.source.node_data.name.clone(),
            edge.source.node_data.file.clone(),
            edge.source.node_data.start,
        );
        let target_key = (
            edge.target.node_data.name.clone(),
            edge.target.node_data.file.clone(),
            edge.target.node_data.start,
        );
        index
            .entry(source_key)
            .or_default()
            .insert((target_key, edge.edge.clone()));
    }
    index
}

#[instrument(skip(graph))]
pub fn link_integration_tests<G: Graph>(graph: &mut G) -> Result<()> {
    let tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    if tests.is_empty() {
        return Ok(());
    }
    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    if endpoints.is_empty() {
        return Ok(());
    }

    let all_functions = graph.find_nodes_by_type(NodeType::Function);
    let all_requests = graph.find_nodes_by_type(NodeType::Request);
    let all_edges = graph.get_edges_vec();
    let edge_index = build_edge_index(&all_edges);
    let edges_count = &all_edges.len();

    let mut endpoint_index: HashMap<(String, String), Vec<&NodeData>> = HashMap::new();
    for ep in &endpoints {
        if let Some(normalized_path) = normalize_backend_path(&ep.name) {
            if let Some(verb) = ep.meta.get("verb") {
                let key = (normalized_path, verb.to_uppercase());
                endpoint_index.entry(key).or_default().push(ep);
            }
        }
    }

    debug!(
        "linking {} integration tests against {} endpoints ({} functions, {} requests, {} edges)",
        tests.len(),
        endpoints.len(),
        all_functions.len(),
        all_requests.len(),
        edges_count
    );

    let mut added_direct = 0;
    let mut added_indirect = 0;

    for t in &tests {
        let body_lc = t.body.to_lowercase();
        let test_verbs = extract_http_verbs_from_test(&t.body);

        for ep in &endpoints {
            if body_lc.contains(&ep.name.to_lowercase()) {
                let matches = if !test_verbs.is_empty() {
                    ep.meta
                        .get("verb")
                        .map(|v| test_verbs.iter().any(|tv| tv.eq_ignore_ascii_case(v)))
                        .unwrap_or(true)
                } else {
                    true
                };

                if matches {
                    let edge =
                        Edge::test_calls(NodeType::IntegrationTest, t, NodeType::Endpoint, ep);
                    graph.add_edge(edge);
                    added_direct += 1;
                }
            }
        }

        let helper_functions = get_called_helpers(&all_functions, &edge_index, t);

        if !helper_functions.is_empty() {
            debug!(
                "test '{}' calls {} helper functions",
                t.name,
                helper_functions.len()
            );
        }

        for helper in &helper_functions {
            let requests_in_helper =
                get_requests_from_helper(&all_functions, &all_requests, &edge_index, helper);

            for request in &requests_in_helper {
                let normalized_request_path = normalize_frontend_path(&request.name);
                if normalized_request_path.is_none() {
                    continue;
                }
                let req_path = normalized_request_path.unwrap();

                if let Some(req_verb) = request.meta.get("verb") {
                    let key = (req_path, req_verb.to_uppercase());
                    if let Some(matching_endpoints) = endpoint_index.get(&key) {
                        for ep in matching_endpoints {
                            debug!(
                                "indirect link: test '{}' -> helper '{}' -> request '{}' -> endpoint '{} {}'",
                                t.name, helper.name, request.name, req_verb, ep.name
                            );
                            let mut updated_ep = (*ep).clone();
                            updated_ep.add_indirect_test(&t.name);
                            updated_ep.add_test_helper(&helper.name);
                            graph.add_node(NodeType::Endpoint, updated_ep);
                            added_indirect += 1;
                        }
                    }
                }
            }
        }
    }

    info!(
        "linked {} direct + {} indirect integration test edges",
        added_direct, added_indirect
    );
    Ok(())
}

fn get_called_helpers(
    all_functions: &[NodeData],
    edge_index: &EdgeIndex,
    test: &NodeData,
) -> Vec<NodeData> {
    all_functions
        .iter()
        .filter(|function| has_edge_by_index(edge_index, test, function, &EdgeType::Calls))
        .cloned()
        .collect()
}

fn get_requests_from_helper(
    all_functions: &[NodeData],
    all_requests: &[NodeData],
    edge_index: &EdgeIndex,
    helper: &NodeData,
) -> Vec<NodeData> {
    let mut requests = Vec::new();

    for request in all_requests {
        if has_edge_by_index(edge_index, helper, request, &EdgeType::Calls)
            || (request.file == helper.file
                && request.start >= helper.start
                && request.start <= helper.end)
        {
            requests.push(request.clone());
        }
    }

    let nested_helpers = get_called_helpers(all_functions, edge_index, helper);
    for nested_helper in nested_helpers {
        let nested_requests =
            get_requests_from_nested_helper(all_requests, edge_index, &nested_helper);
        requests.extend(nested_requests);
    }

    requests
}

fn get_requests_from_nested_helper(
    all_requests: &[NodeData],
    edge_index: &EdgeIndex,
    helper: &NodeData,
) -> Vec<NodeData> {
    all_requests
        .iter()
        .filter(|request| {
            has_edge_by_index(edge_index, helper, request, &EdgeType::Calls)
                || (request.file == helper.file
                    && request.start >= helper.start
                    && request.start <= helper.end)
        })
        .cloned()
        .collect()
}

fn has_edge_by_index(
    edge_index: &EdgeIndex,
    source: &NodeData,
    target: &NodeData,
    edge_type: &EdgeType,
) -> bool {
    let source_key = (source.name.clone(), source.file.clone(), source.start);
    let target_key = (target.name.clone(), target.file.clone(), target.start);
    edge_index
        .get(&source_key)
        .map(|targets| targets.contains(&(target_key, edge_type.clone())))
        .unwrap_or(false)
}

pub fn link_e2e_tests_pages<G: Graph>(graph: &mut G) -> Result<()> {
    let tests = graph.find_nodes_by_type(NodeType::E2eTest);
    if tests.is_empty() {
        return Ok(());
    }
    let pages = graph.find_nodes_by_type(NodeType::Page);
    if pages.is_empty() {
        return Ok(());
    }
    let mut added = 0;
    for t in &tests {
        let body_lc = t.body.to_lowercase();
        for p in &pages {
            if body_lc.contains(&p.name.to_lowercase()) {
                let edge = Edge::test_calls(NodeType::E2eTest, t, NodeType::Page, p);
                graph.add_edge(edge);
                added += 1;
            }
        }
    }
    info!("linked {} e2e test->page edges", added);
    Ok(())
}

pub fn link_tests<G: Graph>(graph: &mut G) -> Result<()> {
    link_integration_tests(graph)?;
    link_e2e_tests_pages(graph)?;
    link_e2e_tests(graph)?;
    Ok(())
}
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

pub fn infer_lang(nd: &NodeData) -> Result<Language> {
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
    Err(Error::Custom(format!(
        "could not infer language for file {}",
        nd.file
    )))
}

pub fn extract_test_ids(content: &str, lang: &Language) -> Result<Vec<String>> {
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

/*
The patterns catch things like:
GET("/path"), POST("/path")
.get("/path"), .post(url)
method: "PUT", type: 'DELETE'
client.get(...), request.post(...)
axios.get(...), axios.post(...), etc.
ky.get(...), ky.post(...), etc.
superagent.get(...), superagent.post(...), etc.
api.get(...), api.post(...), etc. (common API client pattern)
fetch(..., { method: "POST" }) pattern with explicit method option
*/

#[instrument]
pub fn extract_http_verbs_from_test(body: &str) -> Vec<String> {
    let mut verbs = Vec::new();

    let patterns = [
        r#"(?i)\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s*\("#,
        r#"(?i)\.(get|post|put|delete|patch|head|options)\s*\("#,
        r#"(?i)method\s*:\s*["']?(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)["']?"#,
        r#"(?i)type\s*:\s*["']?(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)["']?"#,
        r#"(?i)axios\.(get|post|put|delete|patch|head|options)\("#,
        r#"(?i)ky\.(get|post|put|delete|patch|head|options)\("#,
        r#"(?i)superagent\.(get|post|put|delete|patch|head|options)\("#,
        r#"(?i)api\.(get|post|put|delete|patch|head|options)\("#,
        r#"(?i)client\.(get|post|put|delete|patch|head|options)\("#,
        r#"(?i)request\.(get|post|put|delete|patch|head|options)\("#,
        r#"(?i)fetch\s*\([^,)]*,\s*\{\s*method\s*:\s*["']?(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)["']?"#,
    ];

    for pattern in patterns.iter() {
        if let Ok(re) = Regex::new(pattern) {
            for capture in re.captures_iter(body) {
                if let Some(verb_match) = capture.get(1) {
                    let verb = verb_match.as_str().to_uppercase();
                    if !verbs.contains(&verb) {
                        verbs.push(verb);
                    }
                }
            }
        }
    }

    verbs
}

#[instrument(skip(graph))]
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
                let edge = Edge::calls(NodeType::Request, &req, NodeType::Endpoint, endpoint);
                graph.add_edge(edge);
                i += 1;
            }
        }
    }
    info!("linked {} api nodes", i);

    Ok(())
}

pub fn normalize_frontend_path(path: &str) -> Option<String> {
    if path.starts_with("${") && path.ends_with("}") && !path[2..].contains("${") {
        return None;
    }

    let path_stripped = path
        .replace("http://localhost:3000", "");

    let path_part = if path_stripped.starts_with("${") {
        if let Some(close_brace) = path_stripped.find('}') {
            &path_stripped[close_brace + 1..]
        } else {
            return None;
        }
    } else {
        &path_stripped
    };

    let re = Regex::new(r"\$\{[^}]+\}").ok()?;
    let normalized = re
        .replace_all(path_part, ":param")
        .to_string()
        .trim_start_matches('/')
        .to_string();

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
        // Next.js catch-all "[...param]" style
        (Regex::new(r"\[\.\.\.[^\]]+\]").unwrap(), ":param"),
        // Next.js "[param]" style
        (Regex::new(r"\[[^\]]+\]").unwrap(), ":param"),
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

pub fn verbs_match(req: &NodeData, endpoint: &NodeData) -> bool {
    match (req.meta.get("verb"), endpoint.meta.get("verb")) {
        (Some(req_verb), Some(endpoint_verb)) => {
            req_verb.to_uppercase() == endpoint_verb.to_uppercase()
        }
        _ => false,
    }
}

pub fn paths_match(frontend_path: &str, backend_path: &str) -> bool {
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
        let mut graph = ArrayGraph::new(String::new(), Language::Typescript);

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
