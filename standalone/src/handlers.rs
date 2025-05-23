use crate::types::{AppError, ProcessBody, ProcessResponse, Result};
#[cfg(feature = "neo4j")]
use ast::lang::graphs::graph_ops::GraphOps;
use ast::lang::Graph;
use ast::repo::Repo;
use axum::Json;
use lsp::git::{get_commit_hash, git_pull_or_clone};
use tracing::info;

pub async fn process(body: Json<ProcessBody>) -> Result<Json<ProcessResponse>> {
    #[cfg(feature = "neo4j")]
    {
        let (final_repo_path, final_repo_url, need_clone, username, pat) = resolve_repo(&body)?;

        clone_repo(
            need_clone,
            &final_repo_url,
            &final_repo_path,
            username.clone(),
            pat.clone(),
        )
        .await?;

        let repo_path = &final_repo_path;
        let repo_url = &final_repo_url;

        let current_hash = match get_commit_hash(&repo_path).await {
            Ok(hash) => hash,
            Err(e) => {
                return Err(AppError::Anyhow(anyhow::anyhow!(
                    "Could not get current hash: {}",
                    e
                )))
            }
        };

        let mut graph_ops = GraphOps::new();
        graph_ops.connect()?;

        let stored_hash = match graph_ops.graph.get_repository_hash(&repo_url) {
            Ok(hash) => Some(hash),
            Err(_) => None,
        };

        info!(
            "Current hash: {} | Stored hash: {:?}",
            current_hash, stored_hash
        );

        if let Some(hash) = &stored_hash {
            if hash == &current_hash {
                let (nodes, edges) = graph_ops.graph.get_graph_size();
                return Ok(Json(ProcessResponse {
                    status: "success".to_string(),
                    message: "Repository already processed".to_string(),
                    nodes: nodes as usize,
                    edges: edges as usize,
                }));
            }
        }

        let (nodes, edges) = if let Some(hash) = stored_hash {
            info!("Updating repository hash from {} to {}", hash, current_hash);
            graph_ops.update_incremental(&repo_url, &repo_path, &current_hash, &hash)?
        } else {
            info!("Adding new repository hash: {}", current_hash);
            graph_ops.update_full(&repo_url, &repo_path, &current_hash)?
        };

        Ok(Json(ProcessResponse {
            status: "success".to_string(),
            message: "Repository processed successfully".to_string(),
            nodes: nodes as usize,
            edges: edges as usize,
        }))
    }
    #[cfg(not(feature = "neo4j"))]
    {
        Err(AppError::Anyhow(anyhow::anyhow!(
            "Neo4j feature is not enabled. Please run with --features neo4j."
        )))
    }
}

pub async fn clear_graph() -> Result<Json<ProcessResponse>> {
    #[cfg(feature = "neo4j")]
    {
        let mut graph_ops = GraphOps::new();
        graph_ops.connect()?;
        let (nodes, edges) = graph_ops.clear()?;
        Ok(Json(ProcessResponse {
            status: "success".to_string(),
            message: "Graph cleared".to_string(),
            nodes: nodes as usize,
            edges: edges as usize,
        }))
    }
    #[cfg(not(feature = "neo4j"))]
    {
        Err(AppError::Anyhow(anyhow::anyhow!(
            "Neo4j feature is not enabled. Please run with --features neo4j."
        )))
    }
}

pub async fn ingest(body: Json<ProcessBody>) -> Result<Json<ProcessResponse>> {
    #[cfg(feature = "neo4j")]
    {
        let (final_repo_path, final_repo_url, need_clone, username, pat) = resolve_repo(&body)?;
        clone_repo(
            need_clone,
            &final_repo_url,
            &final_repo_path,
            username.clone(),
            pat.clone(),
        )
        .await?;

        let repo_path = &final_repo_path;
        let repo_url = &final_repo_url;

        let current_hash = match get_commit_hash(&repo_path).await {
            Ok(hash) => hash,
            Err(e) => {
                return Err(AppError::Anyhow(anyhow::anyhow!(
                    "Could not get current hash: {}",
                    e
                )))
            }
        };

        let repos = Repo::new_multi_detect(
            repo_path,
            Some(repo_url.to_string()),
            Vec::new(),
            Vec::new(),
        );
        info!("Building BTreeMapGraph graph for repo: {}", repo_path);
        let btree_graph = repos.build_graphs_btree().await?;

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let base_name = format!("ingest_{}", timestamp);

        info!("Exporting graph to JSONL files...");
        ast::utils::print_json(&btree_graph, &base_name)?;

        let node_file = format!("ast/examples/{}-nodes.jsonl", base_name);
        let edge_file = format!("ast/examples/{}-edges.jsonl", base_name);

        info!("Processing JSONL files and uploading to Neo4j...");
        let mut graph_ops = GraphOps::new();
        graph_ops.connect()?;

        let (nodes, edges) =
            graph_ops.build_from_files(&node_file, &edge_file, repo_url, &current_hash)?;

        let _ = std::fs::remove_file(&node_file);
        let _ = std::fs::remove_file(&edge_file);

        Ok(Json(ProcessResponse {
            status: "success".to_string(),
            message: "Repository ingested fully".to_string(),
            nodes: nodes as usize,
            edges: edges as usize,
        }))
    }

    #[cfg(not(feature = "neo4j"))]
    {
        Err(AppError::Anyhow(anyhow::anyhow!(
            "Neo4j feature is not enabled. Please run with --features neo4j."
        )))
    }
}

fn env_not_empty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}
fn resolve_repo(
    body: &ProcessBody,
) -> Result<(String, String, bool, Option<String>, Option<String>)> {
    let repo_path = body
        .repo_path
        .clone()
        .or_else(|| env_not_empty("REPO_PATH"));
    let repo_url = body.repo_url.clone().or_else(|| env_not_empty("REPO_URL"));
    let username = body.username.clone().or_else(|| env_not_empty("USERNAME"));
    let pat = body.pat.clone().or_else(|| env_not_empty("PAT"));

    if repo_path.is_none() && repo_url.is_none() {
        return Err(AppError::Anyhow(anyhow::anyhow!(
            "Neither REPO_PATH nor REPO_URL is set in the body or environment"
        )));
    }

    if let Some(path) = repo_path {
        Ok((path, repo_url.unwrap_or_default(), false, username, pat))
    } else {
        let url = repo_url.unwrap();
        let tmp_path = Repo::get_path_from_url(&url)?;
        Ok((tmp_path, url, true, username, pat))
    }
}
async fn clone_repo(
    need_clone: bool,
    repo_url: &str,
    repo_path: &str,
    username: Option<String>,
    pat: Option<String>,
) -> Result<()> {
    if need_clone {
        info!("Cloning or Pulling repo from {} to {}", repo_url, repo_path);
        if let Err(e) = git_pull_or_clone(repo_url, repo_path, username, pat).await {
            return Err(AppError::Anyhow(anyhow::anyhow!(
                "Git pull or clone failed : {}",
                e
            )));
        }
    }
    Ok(())
}
