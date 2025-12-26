use crate::types::{AppState, ProcessBody, ProcessResponse, Result, WebError};
use crate::utils::{call_mcp_mocks, resolve_repo};
use ast::lang::{graphs::graph_ops::GraphOps, Graph};
use ast::repo::{clone_repo, Repo};
use axum::{extract::State, Json};
use lsp::{git::get_commit_hash, git::validate_git_credentials, strip_tmp};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

#[axum::debug_handler]
pub async fn ingest(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> Result<Json<ProcessResponse>> {
    let start_total = Instant::now();
    let (final_repo_path, final_repo_url, username, pat, commit, branch) = resolve_repo(&body)?;
    let use_lsp = body.use_lsp;
    let repo_url = final_repo_url.clone();

    let start_clone = Instant::now();
    let mut repos = if body.repo_path.is_some() || std::env::var("REPO_PATH").is_ok() {
        info!("Using local repository at: {}", final_repo_path);
        Repo::new_multi_detect(
            &final_repo_path,
            Some(final_repo_url.clone()),
            Vec::new(),
            Vec::new(),
            use_lsp,
        )
        .await
        .map_err(|e| {
            WebError(shared::Error::Custom(format!(
                "Repo detection Failed: {}",
                e
            )))
        })?
    } else {
        Repo::new_clone_multi_detect(
            &repo_url,
            username.clone(),
            pat.clone(),
            Vec::new(),
            Vec::new(),
            commit.as_deref(),
            branch.as_deref(),
            use_lsp,
        )
        .await
        .map_err(|e| {
            WebError(shared::Error::Custom(format!(
                "Repo detection Failed: {}",
                e
            )))
        })?
    };
    let clone_s = start_clone.elapsed().as_secs_f64();
    info!(
        "[perf][ingest] phase=clone_detect repo={} s={:.2}",
        final_repo_url.clone(),
        clone_s
    );

    repos.set_status_tx(state.tx.clone()).await;
    let streaming = body.realtime.unwrap_or(false);
    if streaming {
        let mut graph_ops = GraphOps::new();
        graph_ops.connect().await?;
        for repo in &repos.0 {
            let stripped_root = strip_tmp(&repo.root).display().to_string();
            info!("[Stream] Pre-clearing old data for {}...", stripped_root);
            graph_ops.clear_existing_graph(&stripped_root).await?;
        }
    }

    let start_build = Instant::now();
    let btree_graph = repos
        .build_graphs_btree_with_streaming(streaming)
        .await
        .map_err(|e| {
            WebError(shared::Error::Custom(format!(
                "Failed to build graphs: {}",
                e
            )))
        })?;
    let build_s = start_build.elapsed().as_secs_f64();
    info!(
        "[perf][ingest] phase=build repo={} streaming={} s={:.2}",
        final_repo_url.clone(),
        streaming,
        build_s
    );
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    if !streaming {
        for repo in &repos.0 {
            let stripped_root = strip_tmp(&repo.root).display().to_string();
            info!("Clearing old data for {}...", stripped_root);
            graph_ops.clear_existing_graph(&stripped_root).await?;
        }
    }

    let start_upload = Instant::now();

    let (nodes, edges) = if streaming {
        graph_ops.graph.get_graph_size()
    } else {
        info!("Uploading to Neo4j...");
        let res = graph_ops
            .upload_btreemap_to_neo4j(&btree_graph, Some(state.tx.clone()))
            .await?;
        graph_ops.graph.create_indexes().await?;
        res
    };

    // Only set missing properties if not using streaming (for backward compatibility)
    if !streaming {
        info!("Setting Data_Bank property for nodes missing it...");
        if let Err(e) = graph_ops.set_missing_data_bank().await {
            tracing::warn!("Error setting Data_Bank property: {:?}", e);
        }

        info!("Setting default namespace for nodes missing it...");
        if let Err(e) = graph_ops.set_default_namespace().await {
            tracing::warn!("Error setting default namespace: {:?}", e);
        }
    } else {
        info!("Skipping post-processing - properties already set during streaming");
    }

    let _ = state.tx.send(ast::repo::StatusUpdate {
        status: "Complete".to_string(),
        message: "Graph building completed successfully".to_string(),
        step: 16,
        total_steps: 16,
        progress: 100,
        stats: Some(std::collections::HashMap::from([
            ("total_nodes".to_string(), nodes as usize),
            ("total_edges".to_string(), edges as usize),
        ])),
        step_description: Some("Graph building completed".to_string()),
    });

    let upload_s = start_upload.elapsed().as_secs_f64();
    info!(
        "[perf][ingest] phase=upload repo={} streaming={} s={:.2}",
        final_repo_url, streaming, upload_s
    );

    let build_upload_s = build_s + upload_s;
    let total_s = start_total.elapsed().as_secs_f64();
    info!(
        "[perf][ingest][results] repo={} streaming={} clone_s={:.2} build_s={:.2} upload_s={:.2} build_upload_s={:.2} total_s={:.2} nodes={} edges={}",
        final_repo_url,
        streaming,
        clone_s,
        build_s,
        upload_s,
        build_upload_s,
        total_s,
        nodes,
        edges
    );

    if let Ok(diry) = std::env::var("PRINT_ROOT") {
        // add timestamp to the filename
        let timestamp = Instant::now().elapsed().as_millis();
        let filename = format!("{}/standalone-{}", diry, timestamp);
        info!("Printing nodes and edges to files... {}", filename);
        if let Err(e) = ast::utils::print_json(&btree_graph, &filename) {
            tracing::warn!("Error printing nodes and edges to files: {}", e);
        }
    }

    call_mcp_mocks(&repo_url, username.as_deref(), pat.as_deref(), false).await;

    Ok(Json(ProcessResponse { nodes, edges }))
}

#[axum::debug_handler]
pub async fn sync(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> Result<Json<ProcessResponse>> {
    if body.repo_url.clone().unwrap_or_default().contains(",") {
        return Err(WebError(shared::Error::Custom(
            "Multiple repositories are not supported in a single request".into(),
        )));
    }
    let (final_repo_path, final_repo_url, username, pat, _, branch) = resolve_repo(&body)?;

    if let Err(e) = validate_git_credentials(&final_repo_url, username.clone(), pat.clone()).await {
        return Err(WebError(e));
    }

    let use_lsp = body.use_lsp;

    let total_start = Instant::now();

    let repo_path = &final_repo_path;
    let repo_url = &final_repo_url;

    clone_repo(
        &repo_url,
        &repo_path,
        username.clone(),
        pat.clone(),
        None,
        branch.as_deref(),
    )
    .await?;

    let current_hash = match get_commit_hash(&repo_path).await {
        Ok(hash) => hash,
        Err(e) => {
            return Err(WebError(shared::Error::Custom(format!(
                "Could not get current hash: {e}"
            ))));
        }
    };

    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    let stored_hash = match graph_ops.graph.get_repository_hash(&repo_url).await {
        Ok(hash) => Some(hash),
        Err(_) => None,
    };

    info!(
        "Current hash: {} | Stored hash: {:?}",
        current_hash, stored_hash
    );

    if let Some(hash) = &stored_hash {
        if hash == &current_hash {
            info!(
                "Repository already processed with hash: {}\n\n",
                current_hash
            );
            let (nodes, edges) = graph_ops.graph.get_graph_size();
            return Ok(Json(ProcessResponse { nodes, edges }));
        }
    }

    let hash = stored_hash.as_deref().unwrap_or_default();

    let (prev_nodes, prev_edges) = graph_ops.graph.get_graph_size();

    info!("Updating repository hash from {} to {}", hash, current_hash);
    let (nodes, edges) = graph_ops
        .update_incremental(
            &repo_url,
            username.clone(),
            pat.clone(),
            &current_hash,
            hash,
            None,
            branch.as_deref(),
            use_lsp,
            Some(state.tx.clone()),
        )
        .await?;

    info!(
        "\n\n ==>> Total processing time: {:.2?} \n\n",
        total_start.elapsed()
    );

    let delta_nodes = nodes - prev_nodes;
    let delta_edges = edges - prev_edges;

    Ok(Json(ProcessResponse {
        nodes: delta_nodes,
        edges: delta_edges,
    }))
}

pub async fn clear_graph() -> Result<Json<ProcessResponse>> {
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;
    let (nodes, edges) = graph_ops.clear().await?;
    Ok(Json(ProcessResponse { nodes, edges }))
}
