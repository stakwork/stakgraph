// use crate::lang::Graph;
// use crate::repo::{check_revs_files, Repo};
// use crate::utils::logger;
// use anyhow::{Context, Result};
// use std::env;
// use tracing::{debug, info};

// #[cfg(feature = "neo4j")]
// use crate::lang::graphs::Neo4jGraph;
// #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
// async fn test_neo4j_revs() -> Result<()> {
//     logger();

//     let repo_path = env::var("REPO_PATH").ok();
//     let repo_urls = env::var("REPO_URL")
//         .unwrap_or_else(|_| "https://github.com/fayekelmith/demorepo.git".to_string());
//     if repo_path.is_none() && repo_urls.is_empty() {
//         return Err(anyhow::anyhow!("no REPO_PATH or REPO_URL"));
//     }

//     let repos_without_filter = if let Some(path) = &repo_path {
//         info!("Using local repository at {}", path);
//         Repo::new_multi_detect(path, None, Vec::new(), Vec::new())
//             .await
//             .context("Failed to create repo without filter")?
//     } else {
//         let urls = &repo_urls;
//         info!("Using remote repositories: {}", urls);

//         let username = env::var("USERNAME").ok();
//         let pat = env::var("PAT").ok();
//         Repo::new_clone_multi_detect(urls, username, pat, Vec::new(), Vec::new())
//             .await
//             .context("Failed to clone repository")?
//     };

//     if repos_without_filter.0.is_empty() {
//         return Err(anyhow::anyhow!("No repositories detected"));
//     }

//     info!(
//         "Found {} repositories to analyze",
//         repos_without_filter.0.len()
//     );

//     let mut all_changed_files = Vec::new();
//     let mut repo_info = Vec::new();

//     for (idx, repo) in repos_without_filter.0.iter().enumerate() {
//         let path = repo
//             .root
//             .to_str()
//             .ok_or_else(|| anyhow::anyhow!("Invalid repository path"))?;

//         let revs = match env::var("REV").ok().filter(|v| !v.is_empty()) {
//             Some(rev_str) => {
//                 info!("Using revisions from env: {}", rev_str);
//                 rev_str.split(',').map(String::from).collect::<Vec<_>>()
//             }
//             None => match Repo::get_last_revisions(path, 2) {
//                 Ok(revs) => revs,
//                 Err(_) => {
//                     debug!("Using default HEAD revisions for {}", path);
//                     vec!["HEAD~1".to_string(), "HEAD".to_string()]
//                 }
//             },
//         };

//         if revs.len() < 2 {
//             info!(
//                 "Repository {} has fewer than 2 revisions, skipping",
//                 idx + 1
//             );
//             continue;
//         }

//         repo_info.push((path.to_string(), revs.clone()));

//         info!(
//             "Repository {}: {} -> {}",
//             path.to_string(),
//             revs[0],
//             revs[1]
//         );

//         if let Some(changed_files) = check_revs_files(path, revs.clone()) {
//             for file in &changed_files {
//                 info!("  -> {}", file);
//                 all_changed_files.push(file.clone());
//             }
//         }
//     }

//     info!(
//         "=> {} files changed across all repositories",
//         all_changed_files.len()
//     );

//     let mut complete_graph = Neo4jGraph::default();
//     complete_graph.connect().await?;
//     complete_graph.clear();

//     println!("=====Building complete graph=====");
//     let complete_graph = repos_without_filter
//         .build_graphs_inner::<Neo4jGraph>()
//         .await
//         .context("Failed to build complete graph")?;

//     let (complete_nodes, complete_edges) = complete_graph.get_graph_size();
//     println!(
//         "Complete graph: \n {} nodes \n{} edges",
//         complete_nodes, complete_edges
//     );

//     let mut all_revs = Vec::new();
//     for (_, revs) in &repo_info {
//         all_revs.extend(revs.clone());
//     }

//     let repos_with_filter = if let Some(path) = &repo_path {
//         Repo::new_multi_detect(path, None, Vec::new(), all_revs)
//             .await
//             .context("Failed to create repo with filter")?
//     } else {
//         let urls = &repo_urls;
//         let username = env::var("USERNAME").ok();
//         let pat = env::var("PAT").ok();

//         Repo::new_clone_multi_detect(urls, username, pat, Vec::new(), all_revs)
//             .await
//             .context("Failed to create repo with filter")?
//     };

//     let mut filtered_graph = Neo4jGraph::default();
//     filtered_graph.connect().await?;
//     filtered_graph.clear();

//     println!("=====Building filtered graph with revs=====");
//     let filtered_graph = repos_with_filter
//         .build_graphs_inner::<Neo4jGraph>()
//         .await
//         .context("Failed to build filtered graph")?;

//     let (filtered_nodes, filtered_edges) = filtered_graph.get_graph_size();
//     println!(
//         "Filtered graph: \n{} nodes \n{} edges",
//         filtered_nodes, filtered_edges
//     );

//     assert!(
//         filtered_nodes <= complete_nodes,
//         " {} Filtered Nodes vs {} Complete Nodes",
//         filtered_nodes,
//         complete_nodes
//     );

//     Ok(())
// }
