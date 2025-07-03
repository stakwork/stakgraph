use super::utils::*;
use crate::lang::graphs::Graph;
#[cfg(feature = "neo4j")]
use crate::lang::graphs::Neo4jGraph;

use crate::lang::{asg::NodeData, graphs::NodeType};
use crate::lang::{ArrayGraph, BTreeMapGraph};
use crate::repo::Repo;
use anyhow::Result;
use git_url_parse::GitUrl;
use lsp::{git::get_commit_hash, strip_root, Cmd as LspCmd, DidOpen};
use std::collections::HashSet;
use std::path::PathBuf;
use tokio::fs;
use tracing::{debug, info, trace};

impl Repo {
    pub async fn build_graph(&self) -> Result<ArrayGraph> {
        self.build_graph_inner().await
    }

    pub async fn build_graph_btree(&self) -> Result<BTreeMapGraph> {
        self.build_graph_inner().await
    }

    #[cfg(feature = "neo4j")]
    pub async fn build_graph_neo4j(&self) -> Result<Neo4jGraph> {
        self.build_graph_inner().await
    }

    pub async fn build_graph_inner<G: Graph>(&self) -> Result<G> {
        #[cfg(feature = "neo4j")]
        {
            use crate::lang::graphs::neo4j_utils::Neo4jConnectionManager;
            if let Err(e) = Neo4jConnectionManager::initialize_from_env().await {
                info!("Failed to initialize Neo4j connection: {}", e);
            }
        }

        let mut graph = G::new();

        self.add_repository_and_language_nodes(&mut graph).await?;
        let files = self.collect_and_add_directories(&mut graph)?;
        self.process_and_add_files(&mut graph, &files).await?;

        let filez = fileys(&files, &self.root)?;
        self.setup_lsp(&filez)?;

        self.process_libraries(&mut graph, &filez)?;
        self.process_imports(&mut graph, &filez)?;
        self.process_variables(&mut graph, &filez)?;
        self.process_classes(&mut graph, &filez)?;
        self.process_instances_and_traits(&mut graph, &filez)?;
        self.process_data_models(&mut graph, &filez)?;
        self.process_functions_and_tests(&mut graph, &filez).await?;
        self.process_pages_and_templates(&mut graph, &filez)?;
        self.process_endpoints(&mut graph, &filez)?;
        self.finalize_graph(&mut graph, &filez).await?;

        let mut graph = filter_by_revs(&self.root.to_str().unwrap(), self.revs.clone(), graph);

        graph.prefix_paths(&self.root_less_tmp());

        println!("done!");
        let (num_of_nodes, num_of_edges) = graph.get_graph_size();
        println!(
            "Returning Graph with {} nodes and {} edges",
            num_of_nodes, num_of_edges
        );
        Ok(graph)
    }
}

impl Repo {
    fn collect_and_add_directories<G: Graph>(&self, graph: &mut G) -> Result<Vec<PathBuf>> {
        self.send_status_update("collect_and_add_directories", 2);
        debug!("collecting dirs...");
        let dirs = self.collect_dirs()?;
        let all_files = self.collect_all_files()?;
        let mut files: Vec<PathBuf> = all_files
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        files.sort();
        info!("Collected {} files using collect_all_files", files.len());

        let mut dirs_not_empty = Vec::new();
        for d in &dirs {
            let child = files.iter().find(|f| match f.parent() {
                None => false,
                Some(p) => {
                    if &strip_root(p, &self.root) == d {
                        true
                    } else {
                        false
                    }
                }
            });
            if child.is_some() {
                dirs_not_empty.push(d.clone());
                continue;
            }
        }

        info!("adding {} dirs...", dirs_not_empty.len());
        let mut processed_dirs = HashSet::new();

        // let mut i = 0;
        for dir in &dirs_not_empty {
            // self.send_status_progress(i, dirs_not_empty.len());
            // i += 1;
            let dir_path = dir.display().to_string();
            let segments: Vec<&str> = dir_path.split('/').collect();

            let mut current_path = String::new();
            for (idx, segment) in segments.iter().enumerate() {
                if idx > 0 {
                    current_path.push('/');
                }
                current_path.push_str(segment);

                if processed_dirs.contains(&current_path) {
                    continue;
                }

                let mut dir_data = NodeData::in_file(&current_path);
                dir_data.name = segment.to_string();

                let (parent_type, parent_file) = if idx == 0 {
                    (NodeType::Repository, "main".to_string())
                } else {
                    let parent = segments[..idx].join("/");
                    (NodeType::Directory, parent)
                };

                graph.add_node_with_parent(
                    NodeType::Directory,
                    dir_data,
                    parent_type,
                    &parent_file,
                );
                processed_dirs.insert(current_path.clone());
            }
        }
        self.send_status_progress(100, 100);
        Ok(files)
    }
    async fn process_and_add_files<G: Graph>(
        &self,
        graph: &mut G,
        files: &[PathBuf],
    ) -> Result<()> {
        self.send_status_update("process_and_add_files", 3);
        info!("parsing {} files...", files.len());
        // let mut i = 0;
        for filepath in files {
            // self.send_status_progress(i, files.len());
            // i += 1;
            let filename = strip_root(filepath, &self.root);
            let meta = fs::metadata(&filepath).await?;
            let code = if meta.len() > MAX_FILE_SIZE {
                debug!("Skipping large file: {:?}", filename);
                "".to_string()
            } else {
                match std::fs::read_to_string(&filepath) {
                    Ok(content) => content,
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

            if graph.find_nodes_by_name(NodeType::File, &path).len() > 0 {
                continue;
            }

            let mut file_data = self.prepare_file_data(&path, &code);

            if self.lang.kind.is_package_file(&path) {
                file_data
                    .meta
                    .insert("pkg_file".to_string(), "true".to_string());
            }

            let (parent_type, parent_file) = self.get_parent_info(&path);

            graph.add_node_with_parent(NodeType::File, file_data, parent_type, &parent_file);
        }
        self.send_status_progress(100, 100);
        Ok(())
    }
    fn setup_lsp(&self, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("setup_lsp", 4);
        info!("=> DidOpen...");
        if let Some(lsp_tx) = self.lsp_tx.as_ref() {
            let mut i = 0;
            for (filename, code) in filez {
                self.send_status_progress(i, filez.len());
                i += 1;
                let didopen = DidOpen {
                    file: filename.into(),
                    text: code.to_string(),
                    lang: self.lang.kind.clone(),
                };
                trace!("didopen: {:?}", didopen);
                let _ = LspCmd::DidOpen(didopen).send(&lsp_tx)?;
            }
        }
        Ok(())
    }
    fn process_libraries<G: Graph>(&self, graph: &mut G, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("process_libraries", 5);
        let mut i = 0;
        let pkg_files = filez
            .iter()
            .filter(|(f, _)| self.lang.kind.is_package_file(f));
        for (pkg_file, code) in pkg_files {
            info!("=> get_packages in... {:?}", pkg_file);

            let mut file_data = self.prepare_file_data(&pkg_file, code);
            file_data.meta.insert("lib".to_string(), "true".to_string());

            let (parent_type, parent_file) = self.get_parent_info(&pkg_file);

            graph.add_node_with_parent(NodeType::File, file_data, parent_type, &parent_file);

            let libs = self.lang.get_libs::<G>(&code, &pkg_file)?;
            i += libs.len();

            for lib in libs {
                graph.add_node_with_parent(NodeType::Library, lib, NodeType::File, &pkg_file);
            }
        }
        self.send_status_progress(100, 100);
        info!("=> got {} libs", i);
        Ok(())
    }
    fn process_imports<G: Graph>(&self, graph: &mut G, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("process_imports", 6);
        let mut i = 0;
        let mut cnt = 0;
        info!("=> get_imports...");
        for (filename, code) in filez {
            self.send_status_progress(cnt, filez.len());
            cnt += 1;
            let imports = self.lang.get_imports::<G>(&code, &filename)?;

            let import_section = combine_imports(imports);
            if !import_section.is_empty() {
                i += 1;
            }
            for import in import_section {
                graph.add_node_with_parent(
                    NodeType::Import,
                    import.clone(),
                    NodeType::File,
                    &import.file,
                );
            }
        }
        info!("=> got {} import sections", i);
        Ok(())
    }
    fn process_variables<G: Graph>(&self, graph: &mut G, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("process_variables", 7);
        let mut i = 0;
        let mut cnt = 0;
        info!("=> get_vars...");
        for (filename, code) in filez {
            self.send_status_progress(cnt, filez.len());
            cnt += 1;
            let variables = self.lang.get_vars::<G>(&code, &filename)?;

            i += variables.len();
            for variable in variables {
                graph.add_node_with_parent(
                    NodeType::Var,
                    variable.clone(),
                    NodeType::File,
                    &variable.file,
                );
            }
        }
        info!("=> got {} all vars", i);
        Ok(())
    }
    async fn add_repository_and_language_nodes<G: Graph>(&self, graph: &mut G) -> Result<()> {
        self.send_status_update("add_repository_and_language_nodes", 1);
        println!("Root: {:?}", self.root);
        let commit_hash = get_commit_hash(&self.root.to_str().unwrap()).await?;
        println!("Commit(commit_hash): {:?}", commit_hash);

        let (org, repo_name) = if !self.url.is_empty() {
            let gurl = GitUrl::parse(&self.url)?;
            (gurl.owner.unwrap_or_default(), gurl.name)
        } else {
            ("".to_string(), format!("{:?}", self.lang.kind))
        };
        debug!("add repository...");
        let mut repo_data = NodeData {
            name: format!("{}/{}", org, repo_name),
            file: format!("main"),
            hash: Some(commit_hash.to_string()),
            ..Default::default()
        };
        repo_data.add_source_link(&self.url);
        graph.add_node_with_parent(NodeType::Repository, repo_data, NodeType::Repository, "");

        debug!("add language...");
        let lang_data = NodeData {
            name: self.lang.kind.to_string(),
            file: "".to_string(),
            ..Default::default()
        };
        graph.add_node_with_parent(NodeType::Language, lang_data, NodeType::Repository, "main");
        self.send_status_progress(100, 100);
        Ok(())
    }
    fn process_classes<G: Graph>(&self, graph: &mut G, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("process_classes", 8);
        let mut i = 0;
        let mut cnt = 0;
        info!("=> get_classes...");
        for (filename, code) in filez {
            self.send_status_progress(cnt, filez.len());
            cnt += 1;
            let qo = self
                .lang
                .q(&self.lang.lang().class_definition_query(), &NodeType::Class);
            let classes = self
                .lang
                .collect_classes::<G>(&qo, &code, &filename, &graph)?;
            i += classes.len();
            for (class, assoc_edges) in classes {
                graph.add_node_with_parent(
                    NodeType::Class,
                    class.clone(),
                    NodeType::File,
                    &class.file,
                );
                for edge in assoc_edges {
                    graph.add_edge(edge);
                }
            }
        }
        info!("=> got {} classes", i);
        graph.class_inherits();
        graph.class_includes();
        Ok(())
    }
    fn process_instances_and_traits<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_instances_and_traits", 9);
        let mut cnt = 0;
        info!("=> get_instances...");
        for (filename, code) in filez {
            self.send_status_progress(cnt, filez.len());
            cnt += 1;
            let q = self.lang.lang().instance_definition_query();
            let instances =
                self.lang
                    .get_query_opt::<G>(q, &code, &filename, NodeType::Instance)?;

            graph.add_instances(instances);
        }
        let mut i = 0;
        info!("=> get_traits...");
        for (filename, code) in filez {
            let traits = self.lang.get_traits::<G>(&code, &filename)?;
            i += traits.len();

            for tr in traits {
                graph.add_node_with_parent(NodeType::Trait, tr.clone(), NodeType::File, &tr.file);
            }
        }
        info!("=> got {} traits", i);
        Ok(())
    }
    fn process_data_models<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_data_models", 10);
        let mut i = 0;
        let mut cnt = 0;
        info!("=> get_structs...");
        for (filename, code) in filez {
            self.send_status_progress(cnt, filez.len());
            cnt += 1;
            if let Some(dmf) = self.lang.lang().data_model_path_filter() {
                if !filename.contains(&dmf) {
                    continue;
                }
            }
            let q = self.lang.lang().data_model_query();
            let structs = self
                .lang
                .get_query_opt::<G>(q, &code, &filename, NodeType::DataModel)?;
            i += structs.len();

            for st in &structs {
                graph.add_node_with_parent(
                    NodeType::DataModel,
                    st.clone(),
                    NodeType::File,
                    &st.file,
                );
            }
            for dm in &structs {
                let edges = self.lang.collect_class_contains_datamodel_edge(dm, graph)?;
                for edge in edges {
                    graph.add_edge(edge);
                }
            }
        }
        info!("=> got {} data models", i);
        Ok(())
    }
    async fn process_functions_and_tests<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_functions_and_tests", 11);
        let mut i = 0;
        let mut cnt = 0;
        info!("=> get_functions_and_tests...");
        for (filename, code) in filez {
            self.send_status_progress(cnt, filez.len());
            cnt += 1;
            let (funcs, tests) =
                self.lang
                    .get_functions_and_tests(&code, &filename, graph, &self.lsp_tx)?;
            i += funcs.len();
            graph.add_functions(funcs.clone());
            i += tests.len();

            for test in tests {
                graph.add_node_with_parent(
                    NodeType::Test,
                    test.0.clone(),
                    NodeType::File,
                    &test.0.file,
                );
            }
        }
        info!("=> got {} functions and tests", i);
        Ok(())
    }
    fn process_pages_and_templates<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_pages_and_templates", 12);
        let mut i = 0;
        let mut cnt = 0;
        info!("=> get_pages");
        for (filename, code) in filez {
            self.send_status_progress(cnt, filez.len());
            cnt += 1;
            if self.lang.lang().is_router_file(&filename, &code) {
                let pages = self.lang.get_pages(&code, &filename, &self.lsp_tx, graph)?;
                i += pages.len();
                graph.add_pages(pages);
            }
        }
        info!("=> got {} pages", i);

        if self.lang.lang().use_extra_page_finder() {
            info!("=> get_extra_pages");
            let closure = |fname: &str| self.lang.lang().is_extra_page(fname);
            let extra_pages = self.collect_extra_pages(closure)?;
            info!("=> got {} extra pages", extra_pages.len());
            for pagepath in extra_pages {
                if let Some(pagename) = get_page_name(&pagepath) {
                    let code = filez
                        .iter()
                        .find(|(f, _)| f.ends_with(&pagepath) || pagepath.ends_with(f))
                        .map(|(_, c)| c.as_str())
                        .unwrap_or("");
                    let mut nd = NodeData::name_file(&pagename, &pagepath);
                    nd.body = code.to_string();
                    let edge = self
                        .lang
                        .lang()
                        .extra_page_finder(&pagepath, &|name, filename| {
                            graph.find_node_by_name_and_file_end_with(
                                NodeType::Function,
                                name,
                                filename,
                            )
                        });
                    graph.add_page((nd, edge));
                }
            }
        }

        i = 0;
        info!("=> get_component_templates");
        for (filename, code) in filez {
            if let Some(ext) = self.lang.lang().template_ext() {
                if filename.ends_with(ext) {
                    let template_edges = self
                        .lang
                        .get_component_templates::<G>(&code, &filename, &graph)?;
                    i += template_edges.len();
                    for edge in template_edges {
                        let mut page = NodeData::name_file(
                            &edge.source.node_data.name,
                            &edge.source.node_data.file,
                        );
                        page.body = code.clone();
                        graph.add_node_with_parent(
                            NodeType::Page,
                            page,
                            NodeType::File,
                            &edge.source.node_data.file,
                        );
                        graph.add_edge(edge);
                    }
                }
            }
        }
        info!("=> got {} component templates/styles", i);

        let selector_map = self.lang.lang().component_selector_to_template_map(filez);
        if !selector_map.is_empty() {
            info!("=> get_page_component_renders");
            let mut page_renders_count = 0;
            for (filename, code) in filez {
                let page_edges = self.lang.lang().page_component_renders_finder(
                    filename,
                    code,
                    &selector_map,
                    &|file_path| {
                        graph
                            .find_nodes_by_file_ends_with(NodeType::Page, file_path)
                            .first()
                            .cloned()
                    },
                );
                page_renders_count += page_edges.len();
                for edge in page_edges {
                    graph.add_edge(edge);
                }
            }
            info!("=> got {} page component renders", page_renders_count);
        }

        Ok(())
    }
    fn process_endpoints<G: Graph>(&self, graph: &mut G, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("process_endpoints", 13);
        let mut i = 0;
        let mut cnt = 0;
        info!("=> get_endpoints...");
        for (filename, code) in filez {
            self.send_status_progress(cnt, filez.len());
            cnt += 1;
            if let Some(epf) = self.lang.lang().endpoint_path_filter() {
                if !filename.contains(&epf) {
                    continue;
                }
            }
            if self.lang.lang().is_test_file(&filename) {
                continue;
            }
            debug!("get_endpoints in {:?}", filename);
            let endpoints =
                self.lang
                    .collect_endpoints(&code, &filename, Some(graph), &self.lsp_tx)?;
            i += endpoints.len();

            graph.add_endpoints(endpoints);
        }
        info!("=> got {} endpoints", i);

        info!("=> get_endpoint_groups...");
        for (filename, code) in filez {
            if self.lang.lang().is_test_file(&filename) {
                continue;
            }
            let q = self.lang.lang().endpoint_group_find();
            let endpoint_groups =
                self.lang
                    .get_query_opt::<G>(q, &code, &filename, NodeType::Endpoint)?;
            let _ = graph.process_endpoint_groups(endpoint_groups, &self.lang);
        }

        if self.lang.lang().use_data_model_within_finder() {
            info!("=> get_data_models_within...");
            graph.get_data_models_within(&self.lang);
        }
        Ok(())
    }

    async fn finalize_graph<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        let mut i = 0;
        info!("=> get_import_edges...");
        for (filename, code) in filez {
            if let Some(import_query) = self.lang.lang().imports_query() {
                let q = self.lang.q(&import_query, &NodeType::Import);
                let import_edges =
                    self.lang
                        .collect_import_edges(&q, &code, &filename, graph, &self.lsp_tx)?;
                for edge in import_edges {
                    graph.add_edge(edge);
                    i += 1;
                }
            }
        }
        info!("=> got {} import edges", i);

        self.send_status_update("process_integration_tests", 14);

        i = 0;
        let mut cnt = 0;
        if self.lang.lang().use_integration_test_finder() {
            info!("=> get_integration_tests...");
            for (filename, code) in filez {
                self.send_status_progress(cnt, filez.len());
                cnt += 1;
                if !self.lang.lang().is_test_file(&filename) {
                    continue;
                }
                let int_tests = self.lang.collect_integration_tests(code, filename, graph)?;
                i += int_tests.len();
                for (nd, tt, edge_opt) in int_tests {
                    graph.add_test_node(nd, tt, edge_opt);
                }
            }
        }
        info!("=> got {} integration tests", i);

        let skip_calls = std::env::var("DEV_SKIP_CALLS").is_ok();
        if skip_calls {
            println!("=> Skipping function_calls...");
        } else {
            self.send_status_update("process_function_calls", 15);
            i = 0;
            let mut cnt = 0;
            info!("=> get_function_calls...");
            for (filename, code) in filez {
                self.send_status_progress(cnt, filez.len());
                cnt += 1;
                let all_calls = self
                    .lang
                    .get_function_calls(&code, &filename, graph, &self.lsp_tx)
                    .await?;
                i += all_calls.0.len();
                graph.add_calls(all_calls);
            }
            info!("=> got {} function calls", i);
        }

        self.lang
            .lang()
            .clean_graph(&mut |parent_type, child_type, child_meta_key| {
                graph.filter_out_nodes_without_children(parent_type, child_type, child_meta_key);
            });

        Ok(())
    }
}
