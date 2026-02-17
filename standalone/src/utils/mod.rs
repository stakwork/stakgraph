use crate::types::ProcessBody;
use ast::lang::NodeType;
use ast::repo::Repo;
use reqwest::Client;
use shared::Result;
use std::str::FromStr;
use std::time::Duration;

pub async fn call_mcp_mocks(repo_url: &str, username: Option<&str>, pat: Option<&str>, sync: bool) {
    // MCP_URL default: http://repo2graph.sphinx:3355 (production swarm)
    // For local dev: http://localhost:3355
    let mcp_url =
        std::env::var("MCP_URL").unwrap_or_else(|_| "http://repo2graph.sphinx:3355".to_string());

    let encoded_url = urlencoding::encode(repo_url);
    let mut url = format!("{}/mocks?repo_url={}", mcp_url, encoded_url);
    if let Some(u) = username {
        url.push_str(&format!("&username={}", urlencoding::encode(u)));
    }
    if let Some(p) = pat {
        url.push_str(&format!("&pat={}", urlencoding::encode(p)));
    }
    if sync {
        url.push_str("&sync=true");
    }
    println!(
        "[mcp_mocks] Calling MCP to discover mocks (sync={}): for {}",
        sync, repo_url
    );

    let client = Client::new();
    let mut req = client.get(&url).timeout(Duration::from_secs(300));
    if let Ok(token) = std::env::var("API_TOKEN") {
        req = req.header("x-api-token", token);
    }
    match req.send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                println!("[mcp_mocks] MCP mocks call succeeded");
            } else {
                println!(
                    "[mcp_mocks] MCP mocks call returned status: {}",
                    resp.status()
                );
            }
        }
        Err(e) => {
            println!("[mcp_mocks] MCP mocks call failed (non-fatal): {}", e);
        }
    }
}

pub async fn call_mcp_docs(repo_url: &str, sync: bool) {
    let mcp_url =
        std::env::var("MCP_URL").unwrap_or_else(|_| "http://repo2graph.sphinx:3355".to_string());

    let encoded_url = urlencoding::encode(repo_url);
    let mut url = format!("{}/learn_docs?repo_url={}", mcp_url, encoded_url);
    if sync {
        url.push_str("&sync=true");
    }
    println!(
        "[mcp_docs] Calling MCP to learn docs (sync={}): for {}",
        sync, repo_url
    );

    let client = Client::new();
    let mut req = client.post(&url).timeout(Duration::from_secs(300));
    if let Ok(token) = std::env::var("API_TOKEN") {
        req = req.header("x-api-token", token);
    }
    match req.send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                println!("[mcp_docs] MCP docs call succeeded");
            } else {
                println!(
                    "[mcp_docs] MCP docs call returned status: {}",
                    resp.status()
                );
            }
        }
        Err(e) => {
            println!("[mcp_docs] MCP docs call failed (non-fatal): {}", e);
        }
    }
}

pub fn parse_node_type(node_type: &str) -> Result<NodeType> {
    let mut chars: Vec<char> = node_type.chars().collect();
    if !chars.is_empty() {
        chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
    }
    let titled_case = chars.into_iter().collect::<String>();
    NodeType::from_str(&titled_case)
}

pub fn parse_node_types(node_types_str: &str) -> Result<Vec<NodeType>> {
    node_types_str
        .split(',')
        .map(|s| parse_node_type(s.trim()))
        .collect()
}

fn env_not_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

pub fn extract_repo_name(url: &str) -> Result<String> {
    let gurl = git_url_parse::GitUrl::parse(url)?;
    Ok(gurl.name)
}

pub async fn call_mcp_embed(
    repo_url: &str,
    cost_limit: f32,
    file_paths: Vec<String>,
    sync: bool,
) {
    let mcp_url =
        std::env::var("MCP_URL").unwrap_or_else(|_| "http://repo2graph.sphinx:3355".to_string());

    let url = format!("{}/repo/describe", mcp_url);
    println!(
        "[mcp_embed] Calling MCP to embed descriptions (sync={}, files={}): {}",
        sync,
        file_paths.len(),
        repo_url
    );

    let client = Client::new();
    let body = serde_json::json!({
        "cost_limit": cost_limit,
        "repo_url": repo_url,
        "file_paths": file_paths,
    });

    let mut req = client
        .post(&url)
        .json(&body)
        .timeout(Duration::from_secs(300));
    if let Ok(token) = std::env::var("API_TOKEN") {
        req = req.header("x-api-token", token);
    }
    match req.send().await {
        Ok(resp) => {
            if resp.status().is_success() {
                println!("[mcp_embed] MCP embed call succeeded for repo: {}", repo_url);
            } else {
                println!(
                    "[mcp_embed] MCP embed call returned status: {}",
                    resp.status()
                );
            }
        }
        Err(e) => {
            println!("[mcp_embed] MCP embed call failed (non-fatal): {}", e);
        }
    }
}

pub fn should_call_mcp_for_repo(param_value: &Option<String>, repo_url: &str) -> bool {
    match param_value {
        None => false,
        Some(value) => {
            if value.eq_ignore_ascii_case("true") {
                return true;
            }

            let repo_name = match extract_repo_name(repo_url) {
                Ok(name) => name,
                Err(_) => return false,
            };

            value
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .any(|pattern| repo_name.ends_with(pattern))
        }
    }
}

fn parse_repo_urls(urls: &str) -> Vec<String> {
    urls.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn resolve_repo(
    body: &ProcessBody,
) -> Result<(
    Vec<String>,
    Vec<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
)> {
    let repo_path = body
        .repo_path
        .clone()
        .or_else(|| env_not_empty("REPO_PATH"));
    let repo_url = body.repo_url.clone().or_else(|| env_not_empty("REPO_URL"));
    let username = body.username.clone().or_else(|| env_not_empty("USERNAME"));
    let pat = body.pat.clone().or_else(|| env_not_empty("PAT"));
    let commit = body.commit.clone();
    let branch = body.branch.clone();

    if repo_path.is_none() && repo_url.is_none() {
        return Err(shared::Error::Custom(
            "Neither REPO_PATH nor REPO_URL is set in the body or environment".into(),
        ));
    }

    if let Some(path) = repo_path {
        Ok((
            vec![path.clone()],
            vec![repo_url.unwrap_or_default()],
            username,
            pat,
            commit,
            branch,
        ))
    } else {
        let url_string = repo_url.unwrap();
        let urls = parse_repo_urls(&url_string);

        if urls.is_empty() {
            return Err(shared::Error::Custom(
                "REPO_URL is empty after parsing".into(),
            ));
        }

        let mut paths = Vec::new();
        for url in &urls {
            let tmp_path = Repo::get_path_from_url(url)?;
            paths.push(tmp_path);
        }

        Ok((paths, urls, username, pat, commit, branch))
    }
}
