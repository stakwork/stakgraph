use crate::types::ProcessBody;
use crate::types::{AppError, ProcessResponse, Result};
#[cfg(feature = "neo4j")]
use ast::lang::graphs::Neo4jGraph;
use ast::lang::Graph;
use ast::repo::check_revs_files;
use ast::Repo;
use axum::Json;
use lsp::git::{get_commit_hash, git_pull_or_clone};
use tracing::info;
pub async fn process(Json(body): Json<ProcessBody>) -> Result<Json<ProcessResponse>> {
    let repo_url = body.repo_url.clone();
    let username = body.username.clone();
    let pat = body.pat.clone();

    let result = tokio::task::spawn_blocking(move || {
        #[cfg(feature = "neo4j")]
        {
            use futures::executor::block_on;

            info!("Processing repository: {}", repo_url);
            let mut graph = Neo4jGraph::default();

            if let Err(e) = block_on(graph.connect()) {
                return Err(AppError::Anyhow(anyhow::anyhow!(
                    "Neo4j connection error: {}",
                    e
                )));
            }

            let repo_path = Repo::get_path_from_url(&repo_url)?;

            if let Err(e) = block_on(git_pull_or_clone(&repo_url, &repo_path, username, pat)) {
                return Err(AppError::Anyhow(anyhow::anyhow!(
                    "Git pull or clone failed : {}",
                    e
                )));
            }

            let current_hash = match block_on(get_commit_hash(&repo_path)) {
                Ok(hash) => hash,
                Err(e) => {
                    return Err(AppError::Anyhow(anyhow::anyhow!(
                        "Could not get current hash: {}",
                        e
                    )))
                }
            };

            let (old_nodes, old_edges) = graph.get_graph_size();
            info!("Old graph size: {} nodes, {} edges", old_nodes, old_edges);

            let stored_hash = match graph.get_repository_hash(&repo_url) {
                Ok(hash) => Some(hash),
                Err(_) => None,
            };
            info!(
                "Current hash: {} | Stored hash: {:?}",
                current_hash, stored_hash
            );

            if let Some(hash) = &stored_hash {
                if hash == &current_hash {
                    let (nodes, edges) = graph.get_graph_size();
                    return Ok(ProcessResponse {
                        status: "success".to_string(),
                        message: "Repository already processed".to_string(),
                        nodes: nodes as usize,
                        edges: edges as usize,
                    });
                }
            }

            if let Some(hash) = stored_hash {
                info!("Updating repository hash from {} to {}", hash, current_hash);
                let revs = vec![hash.clone(), current_hash.clone()];

                if let Some(changed_files) = check_revs_files(&repo_path, revs.clone()) {
                    info!(
                        "Processing {} changed files between commits",
                        changed_files.len()
                    );

                    let mut deleted_files = Vec::new();
                    for file_path in &changed_files {
                        let full_path = std::path::Path::new(&repo_path).join(file_path);
                        if !full_path.exists() {
                            info!("File deleted: {}", file_path);
                            deleted_files.push(file_path.clone());

                            let (nodes_before, edges_before) = graph.get_graph_size();
                            let deleted_count = graph.remove_nodes_by_file(file_path)?;
                            let (nodes_after, edges_after) = graph.get_graph_size();

                            if deleted_count > 0 {
                                info!(
                                    "Removed {} nodes and {} edges for deleted file {}",
                                    nodes_before - nodes_after,
                                    edges_before - edges_after,
                                    file_path
                                );
                            } else {
                                info!("No nodes removed for deleted file {}", file_path);
                                // return Err(AppError::Anyhow(anyhow::anyhow!(
                                //     "No nodes removed for deleted file {}",
                                //     file_path
                                // )));
                            }
                        }
                    }

                    let modified_files: Vec<String> = changed_files
                        .iter()
                        .filter(|f| !deleted_files.contains(f))
                        .cloned()
                        .collect();

                    if !modified_files.is_empty() {
                        for file in &modified_files {
                            let incoming_edges = graph.get_incoming_edges_for_file(file);

                            graph.remove_nodes_by_file(file)?;

                            let file_repos = block_on(Repo::new_multi_detect(
                                &repo_path,
                                Some(repo_url.clone()),
                                vec![file.clone()],
                                Vec::new(),
                            ))?;

                            for repo in &file_repos.0 {
                                let file_graph = block_on(repo.build_graph_inner::<Neo4jGraph>())?;

                                let (nodes_before, edges_before) = graph.get_graph_size();
                                graph.extend_graph(file_graph);

                                for (edge, _target_data) in &incoming_edges {
                                    let source_exists = graph
                                        .find_nodes_by_name(
                                            edge.source.node_type.clone(),
                                            &edge.source.node_data.name,
                                        )
                                        .iter()
                                        .any(|n| n.file == edge.source.node_data.file);
                                    let target_exists = graph
                                        .find_nodes_by_name(
                                            edge.target.node_type.clone(),
                                            &edge.target.node_data.name,
                                        )
                                        .iter()
                                        .any(|n| n.file == edge.target.node_data.file);
                                    if source_exists && target_exists {
                                        graph.add_edge(edge.clone());
                                    }
                                }

                                let (nodes_after, edges_after) = graph.get_graph_size();

                                info!(
                                    "Updated file {}: added {} nodes and {} edges",
                                    file,
                                    nodes_after - nodes_before,
                                    edges_after - edges_before
                                );
                            }
                        }
                    }
                }

                graph.update_repository_hash(&repo_url, &current_hash)?;
            } else {
                info!("Adding new repository hash: {}", current_hash);

                let repos = match block_on(Repo::new_multi_detect(
                    &repo_path,
                    Some(repo_url.clone()),
                    Vec::new(),
                    Vec::new(),
                )) {
                    Ok(r) => r,
                    Err(e) => return Err(AppError::Anyhow(e.into())),
                };

                let temp_graph = match block_on(repos.build_graphs_inner::<Neo4jGraph>()) {
                    Ok(g) => g,
                    Err(e) => return Err(AppError::Anyhow(e.into())),
                };

                graph.extend_graph(temp_graph);

                graph.update_repository_hash(&repo_url, &current_hash)?;
            }

            let (nodes, edges) = graph.get_graph_size();
            info!("Updated graph - Nodes: {}, Edges: {}", nodes, edges);
            Ok(ProcessResponse {
                status: "success".to_string(),
                message: "Repository processed successfully".to_string(),
                nodes: nodes as usize,
                edges: edges as usize,
            })
        }

        #[cfg(not(feature = "neo4j"))]
        {
            Err(AppError::Anyhow(anyhow::anyhow!(
                "Neo4j feature is not enabled"
            )))
        }
    })
    .await
    .map_err(|e| AppError::Anyhow(anyhow::anyhow!("An error occured: {}", e)))?;

    Ok(Json(result?))
}

pub async fn clear_graph() -> Result<Json<ProcessResponse>> {
    #[cfg(feature = "neo4j")]
    {
        let mut graph = Neo4jGraph::default();
        graph.clear();

        let (nodes, edges) = graph.get_graph_size();
        info!("Graph cleared - Nodes: {}, Edges: {}", nodes, edges);
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
            "Neo4j feature is not enabled"
        )))
    }
}
