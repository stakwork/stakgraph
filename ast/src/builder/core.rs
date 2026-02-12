#[cfg(feature = "neo4j")]
use super::streaming::{flush_stage_nodes, flush_stage_nodes_and_edges, StreamingUploadContext};
use super::utils::*;
#[cfg(feature = "neo4j")]
use crate::lang::graphs::Neo4jGraph;
use crate::lang::{
    graphs::{Edge, Graph},
};

use crate::lang::{
    asg::NodeData,
    graphs::NodeType,
};
use crate::lang::BTreeMapGraph;
use crate::repo::Repo;
use crate::lang::call_finder::{parse_imports_for_file, IMPORT_CACHE};

use git_url_parse::GitUrl;
use lsp::{git::get_commit_hash, strip_tmp, Cmd as LspCmd, DidOpen};
use shared::error::Result;
use std::{path::PathBuf, time::Instant};
use std::collections::HashSet;
use tokio::fs;
use tracing::{debug, info, trace};

use super::memory;

#[derive(Debug, Clone)]
pub struct ImplementsRelationship {
    pub class_name: String,
    pub trait_name: String,
    pub file_path: String,
}

impl Repo {
    pub async fn build_graph_local(&self) -> Result<BTreeMapGraph> {
        self.build_graph_inner_with_streaming(false).await
    }
    pub async fn build_graph(&self) -> Result<BTreeMapGraph> {
        self.build_graph_local().await
    }
    pub async fn build_graph_with_batch_upload(&self) -> Result<BTreeMapGraph> {
        self.build_graph_inner_with_streaming(true).await
    }
    pub async fn build_graph_inner<G: Graph + Sync>(&self) -> Result<G> {
        let enable_batch_upload = std::env::var("STREAM_UPLOAD").is_ok();
        self.build_graph_inner_with_streaming(enable_batch_upload).await
    }
    #[allow(unused)]
    pub async fn build_graph_inner_with_streaming<G: Graph + Sync>(
        &self,
        #[cfg_attr(not(feature = "neo4j"), allow(unused))]
        enable_batch_upload: bool,
    ) -> Result<G> {
        let graph_root = strip_tmp(&self.root).display().to_string();
        let mut graph = G::new(graph_root, self.lang.kind.clone());
        graph.set_allow_unverified_calls(self.allow_unverified_calls);

        let mut stats = std::collections::HashMap::new();

        #[cfg(feature = "neo4j")]
        let mut streaming_ctx: Option<StreamingUploadContext> = if enable_batch_upload && std::any::type_name::<G>().contains("BTreeMapGraph") {
            let g = Neo4jGraph::default();
            let _ = g.connect().await;
            Some(StreamingUploadContext::new(g))
        } else {
            None
        };

        self.send_status_update("initialization", 1);
        memory::log_memory("init");

        let stage_start = Instant::now();
        self.add_repository_and_language_nodes(&mut graph).await?;
        info!(
            "[perf][stage] repository_language s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("repository_language");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes(ctx, &graph, "repository_language").await?;
        }
        let stage_start = Instant::now();
        let files = self.collect_and_add_directories(&mut graph)?;
        info!(
            "[perf][stage] directories s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("directories");

        stats.insert("directories".to_string(), files.len());

        let stage_start = Instant::now();
        let filez = self.process_and_add_files(&mut graph, &files).await?;
        info!(
            "[perf][stage] files s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("files");

        stats.insert("files".to_string(), filez.len());

        self.send_status_with_stats(stats.clone());
        self.send_status_progress(100, 100, 1);
        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "files").await?;
        }

        let stage_start = Instant::now();
        self.setup_lsp(&filez)?;
        info!(
            "[perf][stage] lsp_setup s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("lsp_setup");

        let allowed_files = filez
            .iter()
            .filter(|(f, _)| is_allowed_file(&std::path::PathBuf::from(f), &self.lang.kind))
            .cloned()
            .collect::<Vec<_>>();
        let stage_start = Instant::now();
        self.process_libraries(&mut graph, &allowed_files)?;
        info!(
            "[perf][stage] libraries s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("libraries");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "libraries").await?;
        }
        self.process_import_sections(&mut graph, &filez)?;
        info!(
            "[perf][stage] imports s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("imports");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "imports").await?;
        }
        self.process_variables(&mut graph, &allowed_files)?;
        info!(
            "[perf][stage] variables s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("variables");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "variables").await?;
        }
        let impl_relationships = self.process_classes(&mut graph, &allowed_files)?;
        info!(
            "[perf][stage] classes s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("classes");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "classes").await?;
        }
        self.process_instances_and_traits(&mut graph, &allowed_files)?;
        info!(
            "[perf][stage] instances_traits s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("instances_traits");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "instances_traits").await?;
        }
        self.resolve_implements_edges(&mut graph, impl_relationships)?;
        info!(
            "[perf][stage] implements s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("implements");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "implements").await?;
        }
        self.process_data_models(&mut graph, &allowed_files)?;
        info!(
            "[perf][stage] data_models s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("data_models");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "data_models").await?;
        }
        self.process_functions_and_tests(&mut graph, &allowed_files)
            .await?;
        info!(
            "[perf][stage] functions_tests s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("functions_tests");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "functions_tests").await?
        }
        self.process_pages_and_templates(&mut graph, &filez)?;
        info!(
            "[perf][stage] pages_templates s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("pages_templates");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "pages_templates").await?;
        }
        self.process_endpoints(&mut graph, &allowed_files)?;
        info!(
            "[perf][stage] endpoints s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("endpoints");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "endpoints").await?;
        }
        self.finalize_graph(&mut graph, &allowed_files, &mut stats)
            .await?;
        info!(
            "[perf][stage] finalize s={:.2}",
            stage_start.elapsed().as_secs_f64()
        );
        memory::log_memory("finalize");

        #[cfg(feature = "neo4j")]
        if let Some(ctx) = &mut streaming_ctx {
            flush_stage_nodes_and_edges(ctx, &graph, "finalize").await?;
        }

        let graph = filter_by_revs(
            self.root.to_str().unwrap(),
            self.revs.clone(),
            graph,
            self.lang.kind.clone(),
        );

        let (num_of_nodes, num_of_edges) = graph.get_graph_size();
        info!(
            "Returning Graph with {} nodes and {} edges",
            num_of_nodes, num_of_edges
        );

        stats.insert("total_nodes".to_string(), num_of_nodes as usize);
        stats.insert("total_edges".to_string(), num_of_edges as usize);

        self.send_status_with_stats(stats);

        Ok(graph)
    }
}

impl Repo {
    fn collect_and_add_directories<G: Graph>(&self, graph: &mut G) -> Result<Vec<PathBuf>> {
        debug!("collecting dirs...");
        let dirs = self.collect_dirs_with_tmp()?; // /tmp/stakwork/stakgraph/my_directory
        let all_files = self.collect_all_files()?; // /tmp/stakwork/stakgraph/my_directory/my_file.go
        let mut files: Vec<PathBuf> = all_files
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        files.sort();
        info!("Collected {} files using collect_all_files", files.len());

        info!("adding {} dirs...", dirs.len());

        let mut i = 0;
        let total_dirs = dirs.len();
        for dir in &dirs {
            i += 1;
            if i % 10 == 0 || i == total_dirs {
                self.send_status_progress(i, total_dirs, 1);
            }

            let dir_no_tmp_buf = strip_tmp(dir);
            let mut dir_no_tmp = dir_no_tmp_buf.display().to_string();

            // remove leading /
            dir_no_tmp = dir_no_tmp.trim_start_matches('/').to_string();

            let root_no_tmp = strip_tmp(&self.root).display().to_string();

            let mut dir_no_root = dir_no_tmp.strip_prefix(&root_no_tmp).unwrap_or(&dir_no_tmp);
            dir_no_root = dir_no_root.trim_start_matches('/');

            let (parent_type, parent_file) = if dir_no_root.contains("/") {
                // remove LAST slash and any characters after it:
                // let parent = dir_no_tmp.rsplit('/').skip(1).collect::<Vec<_>>().join("/");
                let mut parts: Vec<_> = dir_no_tmp.rsplit('/').skip(1).collect();
                parts.reverse();
                let parent = parts.join("/");
                (NodeType::Directory, parent)
            } else {
                let repo_file = strip_tmp(&self.root).display().to_string();
                (NodeType::Repository, repo_file)
            };

            let dir_name = dir_no_tmp.rsplit('/').next().unwrap().to_string();
            let mut dir_data = NodeData::in_file(&dir_no_tmp);
            dir_data.name = dir_name;

            graph.add_node_with_parent(&NodeType::Directory, &dir_data, &parent_type, &parent_file);
        }
        Ok(files)
    }
    async fn process_and_add_files<G: Graph>(
        &self,
        graph: &mut G,
        files: &[PathBuf],
    ) -> Result<Vec<(String, String)>> {
        info!("parsing {} files...", files.len());
        let mut i = 0;
        let total_files = files.len();
        let mut ret = Vec::new();
        // let mut i = 0;
        for filepath in files {
            i += 1;
            if i % 10 == 0 || i == total_files {
                self.send_status_progress(i, total_files, 2);
            }

            let filename = strip_tmp(filepath);
            let file_name = filename.display().to_string();
            let meta = fs::metadata(&filepath).await?;
            let code = if meta.len() > MAX_FILE_SIZE {
                debug!("Skipping large file: {:?}", filename);
                "".to_string()
            } else {
                match std::fs::read_to_string(filepath) {
                    Ok(content) => {
                        ret.push((file_name, content.clone()));
                        content
                    }
                    Err(_) => {
                        debug!(
                            "Could not read file as string (likely binary): {:?}",
                            filename
                        );
                        "".to_string()
                    }
                }
            };

            let path = filename.display().to_string();

            if !graph.find_nodes_by_name(NodeType::File, &path).is_empty() {
                continue;
            }

            let mut file_data = self.prepare_file_data(&path, &code);

            if self.lang.kind.is_package_file(&path) {
                file_data
                    .meta
                    .insert("pkg_file".to_string(), "true".to_string());
            }

            let (parent_type, parent_file) = self.get_parent_info(filepath);

            graph.add_node_with_parent(&NodeType::File, &file_data, &parent_type, &parent_file);
        }
        Ok(ret)
    }
    fn setup_lsp(&self, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("setup_lsp", 2);
        info!("=> DidOpen...");
        if let Some(lsp_tx) = self.lsp_tx.as_ref() {
            let mut i = 0;
            let total = filez.len();
            for (filename, code) in filez {
                i += 1;
                if i % 5 == 0 || i == total {
                    self.send_status_progress(i, total, 4);
                }

                if !self.lang.kind.is_source_file(filename) {
                    continue;
                }
                let didopen = DidOpen {
                    file: filename.into(),
                    text: code.to_string(),
                    lang: self.lang.kind.clone(),
                };
                trace!("didopen: {:?}", didopen);
                let _ = LspCmd::DidOpen(didopen).send(lsp_tx)?;
            }
            self.send_status_progress(100, 100, 2);
        }
        Ok(())
    }
    async fn add_repository_and_language_nodes<G: Graph>(&self, graph: &mut G) -> Result<()> {
        info!("Root: {:?}", self.root);
        let commit_hash = get_commit_hash(self.root.to_str().unwrap()).await?;
        info!("Commit(commit_hash): {:?}", commit_hash);

        let (org, repo_name) = if !self.url.is_empty() {
            let gurl = GitUrl::parse(&self.url)?;
            (gurl.owner.unwrap_or_default(), gurl.name)
        } else {
            ("".to_string(), format!("{:?}", self.lang.kind))
        };

        let repo_file = strip_tmp(&self.root).display().to_string();
        let full_name = format!("{}/{}", org, repo_name);

        let existing_repos = graph.find_nodes_by_name(NodeType::Repository, &full_name);
        let repo_data = if existing_repos.is_empty() {
            info!(
                "Creating Repository node: {} (name: {})",
                repo_file, full_name
            );
            let mut repo_data = NodeData {
                name: full_name,
                file: repo_file.clone(),
                hash: Some(commit_hash.to_string()),
                ..Default::default()
            };
            repo_data.add_source_link(&self.url);
            graph.add_node_with_parent(
                &NodeType::Repository,
                &repo_data,
                &NodeType::Repository,
                "",
            );
            repo_data
        } else {
            info!(
                "Repository node already exists for: {} (adding language: {})",
                repo_file, self.lang.kind
            );
            existing_repos.first().unwrap().clone()
        };

        debug!("add language for: {}", repo_file);
        let mut lang_data = NodeData::in_file(&repo_file);
        lang_data.name = self.lang.kind.to_string();
        graph.add_node(&NodeType::Language, &lang_data);

        graph.add_edge(&Edge::of_typed(
            NodeType::Repository,
            &repo_data,
            NodeType::Language,
            &lang_data,
        ));

        let mut stats = std::collections::HashMap::new();
        stats.insert(
            "repository".to_string(),
            if existing_repos.is_empty() { 1 } else { 0 },
        );
        stats.insert("language".to_string(), 1);
        self.send_status_with_stats(stats);

        Ok(())
    }
    fn process_classes<G: Graph + Sync>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<Vec<ImplementsRelationship>> {
        self.send_status_update("process_classes", 6);
        let mut i = 0;
        let mut class_count = 0;
        let total = filez.len();
        let mut impl_relationships = Vec::new();

        info!("=> get_classes...");
        info!("=> get_classes...");

        let lang = &self.lang;

        for (filename, code) in filez {
            i += 1;
            if i % 20 == 0 || i == total {
                self.send_status_progress(i, total, 8);
            }

            if !lang.kind.is_source_file(filename) {
                continue;
            }

            let qo = lang.q(&lang.lang().class_definition_query(), &NodeType::Class);
            let classes = lang.collect_classes::<G>(&qo, code, filename, graph)?;

            class_count += classes.len();
            for (class, assoc_edges) in classes {
                graph.add_node_with_parent(&NodeType::Class, &class, &NodeType::File, &class.file);
                for edge in assoc_edges {
                    graph.add_edge(&edge);
                }
            }

            if let Some(impl_query) = lang.lang().implements_query() {
                let q = lang.q(&impl_query, &NodeType::Class);
                let impls = lang.collect_implements(&q, code, filename)?;
                impl_relationships.extend(impls.into_iter().map(
                    |(class_name, trait_name, file_path)| ImplementsRelationship {
                        class_name,
                        trait_name,
                        file_path,
                    },
                ));
            }
        }

        let mut stats = std::collections::HashMap::new();
        stats.insert("classes".to_string(), class_count);
        self.send_status_with_stats(stats);
        self.send_status_progress(100, 100, 6);

        info!("=> got {} classes", class_count);
        graph.class_inherits();
        graph.class_includes();
        Ok(impl_relationships)
    }
    fn resolve_implements_edges<G: Graph>(
        &self,
        graph: &mut G,
        impl_relationships: Vec<ImplementsRelationship>,
    ) -> Result<()> {
        use std::collections::HashMap;

        if impl_relationships.is_empty() {
            return Ok(());
        }

        let classes = graph.find_nodes_by_type(NodeType::Class);
        let traits = graph.find_nodes_by_type(NodeType::Trait);

        let mut classes_by_file: HashMap<&str, Vec<&NodeData>> = HashMap::new();
        for class in &classes {
            classes_by_file
                .entry(class.file.as_str())
                .or_default()
                .push(class);
        }

        let mut traits_by_file: HashMap<&str, Vec<&NodeData>> = HashMap::new();
        for trait_node in &traits {
            traits_by_file
                .entry(trait_node.file.as_str())
                .or_default()
                .push(trait_node);
        }

        let mut edge_count = 0;
        let mut same_file_hits = 0;

        for rel in impl_relationships {
            let class_node = classes_by_file
                .get(rel.file_path.as_str())
                .and_then(|classes| {
                    classes.iter().find(|c| c.name == rel.class_name).map(|c| {
                        same_file_hits += 1;
                        *c
                    })
                })
                .or_else(|| {
                    // Fallback: search all classes by name
                    classes.iter().find(|c| c.name == rel.class_name)
                });

            let trait_node = traits_by_file
                .get(rel.file_path.as_str())
                .and_then(|traits| traits.iter().find(|t| t.name == rel.trait_name).copied())
                .or_else(|| traits.iter().find(|t| t.name == rel.trait_name));

            if let (Some(class), Some(trait_)) = (class_node, trait_node) {
                graph.add_edge(&Edge::implements(class, trait_));
                edge_count += 1;
            }
        }

        info!(
            "=> created {} IMPLEMENTS edges ({} same-file optimizations)",
            edge_count, same_file_hits
        );
        Ok(())
    }
}
