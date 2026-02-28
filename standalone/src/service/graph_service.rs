use crate::types::{AppState, ProcessBody, ProcessResponse, Result, WebError};
use crate::utils::{
    call_mcp_docs, call_mcp_embed, call_mcp_mocks, has_rules_file_changes, resolve_repo,
    should_call_mcp_for_repo,
};
use ast::lang::{graphs::graph_ops::GraphOps, Graph};
use ast::repo::{check_revs_files, clone_repo, Repo};
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
    let (repo_paths, repo_urls, username, pat, commit, branch) = resolve_repo(&body)?;
    let use_lsp = body.use_lsp;
    let docs_param = body.docs.clone();
    let mocks_param = body.mocks.clone();
    let embeddings_param = body.embeddings.clone();
    let embeddings_limit = body.embeddings_limit.unwrap_or(5.0);

    let repo_url_joined = repo_urls.join(",");
    let final_repo_path = repo_paths.first().cloned().unwrap_or_default();

    let start_clone = Instant::now();
    let mut repos = if body.repo_path.is_some() || std::env::var("REPO_PATH").is_ok() {
        info!("Using local repository at: {}", final_repo_path);
        Repo::new_multi_detect(
            &final_repo_path,
            Some(repo_url_joined.clone()),
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
            &repo_url_joined,
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
        "[perf][ingest] phase=clone_detect repos={} count={} s={:.2}",
        repo_url_joined,
        repo_urls.len(),
        clone_s
    );

    repos.set_status_tx(state.tx.clone()).await;
    let enable_batch_upload = body.realtime.unwrap_or(false);
    if enable_batch_upload {
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
        .build_graphs_btree_with_streaming(enable_batch_upload)
        .await
        .map_err(|e| {
            WebError(shared::Error::Custom(format!(
                "Failed to build graphs: {}",
                e
            )))
        })?;
    let build_s = start_build.elapsed().as_secs_f64();
    info!(
        "[perf][ingest] phase=build repos={} streaming={} s={:.2}",
        repo_url_joined, enable_batch_upload, build_s
    );
    let mut graph_ops = GraphOps::new();
    graph_ops.connect().await?;

    if !enable_batch_upload {
        for repo in &repos.0 {
            let stripped_root = strip_tmp(&repo.root).display().to_string();
            info!("Clearing old data for {}...", stripped_root);
            graph_ops.clear_existing_graph(&stripped_root).await?;
        }
    }

    let start_upload = Instant::now();

    let (nodes, edges) = if enable_batch_upload {
        graph_ops.graph.get_graph_size()
    } else {
        info!("Uploading to Neo4j...");
        let res = graph_ops
            .upload_btreemap_to_neo4j(&btree_graph, Some(state.tx.clone()))
            .await?;
        graph_ops.graph.create_indexes().await?;
        res
    };

    // Only set missing properties if not using batch upload (for backward compatibility)
    if !enable_batch_upload {
        info!("Setting Data_Bank property for nodes missing it...");
        if let Err(e) = graph_ops.set_missing_data_bank().await {
            tracing::warn!("Error setting Data_Bank property: {:?}", e);
        }

        info!("Setting default namespace for nodes missing it...");
        if let Err(e) = graph_ops.set_default_namespace().await {
            tracing::warn!("Error setting default namespace: {:?}", e);
        }
    } else {
        info!("Skipping post-processing - properties already set during batch upload");
    }

    if let Err(send_err) = state.tx.send(ast::repo::StatusUpdate {
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
    }) {
        tracing::warn!(
            "No status subscribers available for completion update: {:?}",
            send_err
        );
    }

    let upload_s = start_upload.elapsed().as_secs_f64();
    info!(
        "[perf][ingest] phase=upload repos={} streaming={} s={:.2}",
        repo_url_joined, enable_batch_upload, upload_s
    );

    let build_upload_s = build_s + upload_s;
    let total_s = start_total.elapsed().as_secs_f64();
    info!(
        "[perf][ingest][results] repos={} count={} streaming={} clone_s={:.2} build_s={:.2} upload_s={:.2} build_upload_s={:.2} total_s={:.2} nodes={} edges={}",
        repo_url_joined,
        repo_urls.len(),
        enable_batch_upload,
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

    for repo_url in &repo_urls {
        if should_call_mcp_for_repo(&docs_param, repo_url) {
            call_mcp_docs(repo_url, false).await;
        }
        if should_call_mcp_for_repo(&mocks_param, repo_url) {
            call_mcp_mocks(repo_url, username.as_deref(), pat.as_deref(), false).await;
        }
        if should_call_mcp_for_repo(&embeddings_param, repo_url) {
            call_mcp_embed(repo_url, embeddings_limit, vec![], false).await;
        }
    }

    Ok(Json(ProcessResponse { nodes, edges }))
}

#[axum::debug_handler]
pub async fn sync(
    State(state): State<Arc<AppState>>,
    body: Json<ProcessBody>,
) -> Result<Json<ProcessResponse>> {
    let (repo_paths, repo_urls, username, pat, _, branch) = resolve_repo(&body)?;

    if repo_urls.len() > 1 {
        return Err(WebError(shared::Error::Custom(
            "sync only supports a single repository. Use ingest for multiple repositories.".into(),
        )));
    }

    let final_repo_path = &repo_paths[0];
    let final_repo_url = &repo_urls[0];

    if let Err(e) = validate_git_credentials(final_repo_url, username.clone(), pat.clone()).await {
        return Err(WebError(e));
    }

    let use_lsp = body.use_lsp;
    let docs_param = body.docs.clone();
    let mocks_param = body.mocks.clone();
    let embeddings_param = body.embeddings.clone();
    let embeddings_limit = body.embeddings_limit.unwrap_or(5.0);

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

    let modified_files = if !hash.is_empty() {
        check_revs_files(&repo_path, vec![hash.to_string(), current_hash.clone()])
    } else {
        None
    };

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

    if should_call_mcp_for_repo(&docs_param, repo_url) {
        if let Some(files) = &modified_files {
            if has_rules_file_changes(files) {
                call_mcp_docs(repo_url, true).await;
            }
        }
    }

    if should_call_mcp_for_repo(&mocks_param, repo_url) {
        call_mcp_mocks(repo_url, username.as_deref(), pat.as_deref(), true).await;
    }

    if should_call_mcp_for_repo(&embeddings_param, repo_url) {
        if let Some(files) = &modified_files {
            call_mcp_embed(repo_url, embeddings_limit, files.clone(), true).await;
        }
    }

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
