use super::utils::*;
use crate::lang::graphs::Graph;
#[cfg(feature = "neo4j")]
use crate::lang::graphs::Neo4jGraph;

use crate::lang::{asg::NodeData, graphs::NodeType};
use crate::lang::{ArrayGraph, BTreeMapGraph};
use crate::repo::Repo;
use anyhow::Result;
use git_url_parse::GitUrl;
use lsp::{git::get_commit_hash, strip_tmp, Cmd as LspCmd, DidOpen};
use std::collections::HashSet;
use std::path::PathBuf;
use tokio::fs;
use tracing::{debug, info, trace};

impl Repo {
    pub async fn build_graph(&self) -> Result<BTreeMapGraph> {
        self.build_graph_inner().await
    }
    pub async fn build_graph_array(&self) -> Result<ArrayGraph> {
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
        let graph_root = strip_tmp(&self.root).display().to_string();
        let mut graph = G::new(graph_root, self.lang.kind.clone());
        let mut stats = std::collections::HashMap::new();

        self.add_repository_and_language_nodes(&mut graph).await?;
        let files = self.collect_and_add_directories(&mut graph)?;
        stats.insert("directories".to_string(), files.len());

        let filez = self.process_and_add_files(&mut graph, &files).await?;
        stats.insert("files".to_string(), filez.len());
        self.send_status_with_stats(stats.clone());

        self.setup_lsp(&filez)?;

        self.process_libraries(&mut graph, &filez)?;
        self.process_import_sections(&mut graph, &filez)?;
        // self.process_independent_nodes(&mut graph, &filez)?;
        self.process_variables(&mut graph, &filez)?;
        self.process_classes(&mut graph, &filez)?;
        self.process_instances_and_traits(&mut graph, &filez)?;
        self.process_data_models(&mut graph, &filez)?;
        self.process_functions_and_tests(&mut graph, &filez).await?;
        self.process_pages_and_templates(&mut graph, &filez)?;
        self.process_endpoints(&mut graph, &filez)?;
        self.finalize_graph(&mut graph, &filez, &mut stats).await?;
        let graph = filter_by_revs(
            &self.root.to_str().unwrap(),
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
        self.send_status_update("collect_and_add_directories", 2);
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
                self.send_status_progress(i, total_dirs);
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

            graph.add_node_with_parent(NodeType::Directory, dir_data, parent_type, &parent_file);
        }
        self.send_status_progress(100, 100);
        Ok(files)
    }
    async fn process_and_add_files<G: Graph>(
        &self,
        graph: &mut G,
        files: &[PathBuf],
    ) -> Result<Vec<(String, String)>> {
        self.send_status_update("process_and_add_files", 3);
        info!("parsing {} files...", files.len());
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
        let mut i = 0;
        let total_files = files.len();
        let mut ret = Vec::new();
        // let mut i = 0;
        for filepath in files {
            i += 1;
            if i % 10 == 0 || i == total_files {
                self.send_status_progress(i, total_files);
            }

            let filename = strip_tmp(filepath);
            let file_name = filename.display().to_string();
            let meta = fs::metadata(&filepath).await?;
            let code = if meta.len() > MAX_FILE_SIZE {
                debug!("Skipping large file: {:?}", filename);
                "".to_string()
            } else {
                match std::fs::read_to_string(&filepath) {
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

            if graph.find_nodes_by_name(NodeType::File, &path).len() > 0 {
                continue;
            }

            let mut file_data = self.prepare_file_data(&path, &code);

            if self.lang.kind.is_package_file(&path) {
                file_data
                    .meta
                    .insert("pkg_file".to_string(), "true".to_string());
            }

            let (parent_type, parent_file) = self.get_parent_info(&filepath);

            graph.add_node_with_parent(NodeType::File, file_data, parent_type, &parent_file);
        }
        self.send_status_progress(100, 100);
        Ok(ret)
    }
    fn setup_lsp(&self, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("setup_lsp", 4);
        info!("=> DidOpen...");
        if let Some(lsp_tx) = self.lsp_tx.as_ref() {
            let mut i = 0;
            let total = filez.len();
            for (filename, code) in filez {
                i += 1;
                if i % 5 == 0 || i == total {
                    self.send_status_progress(i, total);
                }

                if !self.lang.kind.is_source_file(&filename) {
                    continue;
                }
                let didopen = DidOpen {
                    file: filename.into(),
                    text: code.to_string(),
                    lang: self.lang.kind.clone(),
                };
                trace!("didopen: {:?}", didopen);
                let _ = LspCmd::DidOpen(didopen).send(&lsp_tx)?;
            }
            self.send_status_progress(100, 100);
        }
        Ok(())
    }
    fn process_libraries<G: Graph>(&self, graph: &mut G, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("process_libraries", 5);
        let mut i = 0;
        let mut lib_count = 0;
        let pkg_files = filez
            .iter()
            .filter(|(f, _)| self.lang.kind.is_package_file(f))
            .collect::<Vec<_>>();

        let total_pkg_files = pkg_files.len();

        for (pkg_file, code) in pkg_files {
            i += 1;
            if i % 2 == 0 || i == total_pkg_files {
                self.send_status_progress(i, total_pkg_files);
            }

            info!("=> get_packages in... {:?}", pkg_file);

            let mut file_data = self.prepare_file_data(&pkg_file, code);
            file_data.meta.insert("lib".to_string(), "true".to_string());

            let (parent_type, parent_file) = self.get_parent_info(&pkg_file.into());

            graph.add_node_with_parent(NodeType::File, file_data, parent_type, &parent_file);

            let libs = self.lang.get_libs::<G>(&code, &pkg_file)?;
            lib_count += libs.len();

            for lib in libs {
                graph.add_node_with_parent(NodeType::Library, lib, NodeType::File, &pkg_file);
            }
        }

        let mut stats = std::collections::HashMap::new();
        stats.insert("libraries".to_string(), lib_count);
        self.send_status_with_stats(stats);

        self.send_status_progress(100, 100);
        info!("=> got {} libs", lib_count);
        Ok(())
    }
    fn process_import_sections<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_imports", 6);
        let mut i = 0;
        let mut import_count = 0;
        let total = filez.len();

        info!("=> get_imports...");
        for (filename, code) in filez {
            i += 1;
            if i % 20 == 0 || i == total {
                self.send_status_progress(i, total);
            }

            let imports = self.lang.get_imports::<G>(&code, &filename)?;

            let import_section = combine_import_sections(imports);
            import_count += import_section.len();

            for import in import_section {
                graph.add_node_with_parent(
                    NodeType::Import,
                    import.clone(),
                    NodeType::File,
                    &import.file,
                );
            }
        }

        let mut stats = std::collections::HashMap::new();
        stats.insert("imports".to_string(), import_count);
        self.send_status_with_stats(stats);

        info!("=> got {} import sections", import_count);
        Ok(())
    }

    fn process_independent_nodes<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        let mut var_count = 0;
        let mut class_count = 0;
        let mut variables = Vec::new();
        let mut classes = Vec::new();
        let total = filez.len();

        info!("=> get_vars_and_classes...");
        for (i, (filename, code)) in filez.iter().enumerate() {
            if i % 20 == 0 || i + 1 == total {
                self.send_status_progress(i + 1, total);
            }

            let vars = self.lang.get_vars::<G>(code, filename)?;
            var_count += vars.len();
            for variable in vars {
                variables.push(variable.clone());
            }

            let qo = self
                .lang
                .q(&self.lang.lang().class_definition_query(), &NodeType::Class);
            let class_items = self.lang.collect_classes::<G>(&qo, code, filename, graph)?;
            class_count += class_items.len();
            for (class, assoc_edges) in class_items {
                classes.push((class.clone(), assoc_edges));
            }
        }

        self.send_status_update("process_variables", 7);
        for variable in variables {
            graph.add_node_with_parent(
                NodeType::Var,
                variable.clone(),
                NodeType::File,
                &variable.file,
            );
        }

        self.send_status_update("process_classes", 8);
        for (class, assoc_edges) in classes {
            graph.add_node_with_parent(NodeType::Class, class.clone(), NodeType::File, &class.file);
            for edge in assoc_edges {
                graph.add_edge(edge);
            }
        }

        let mut stats = std::collections::HashMap::new();
        stats.insert("variables".to_string(), var_count);
        stats.insert("classes".to_string(), class_count);
        self.send_status_with_stats(stats);

        info!("=> got {} all vars and {} classes", var_count, class_count);
        graph.class_inherits();
        graph.class_includes();

        Ok(())
    }
    fn process_variables<G: Graph>(&self, graph: &mut G, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("process_variables", 7);
        let mut i = 0;
        let mut var_count = 0;
        let total = filez.len();

        info!("=> get_vars...");
        for (filename, code) in filez {
            i += 1;
            if i % 20 == 0 || i == total {
                self.send_status_progress(i, total);
            }

            let variables = self.lang.get_vars::<G>(&code, &filename)?;

            var_count += variables.len();
            for variable in variables {
                graph.add_node_with_parent(
                    NodeType::Var,
                    variable.clone(),
                    NodeType::File,
                    &variable.file,
                );
            }
        }

        let mut stats = std::collections::HashMap::new();
        stats.insert("variables".to_string(), var_count);
        self.send_status_with_stats(stats);

        info!("=> got {} all vars", var_count);
        Ok(())
    }
    async fn add_repository_and_language_nodes<G: Graph>(&self, graph: &mut G) -> Result<()> {
        self.send_status_update("add_repository_and_language_nodes", 1);
        info!("Root: {:?}", self.root);
        let commit_hash = get_commit_hash(&self.root.to_str().unwrap()).await?;
        info!("Commit(commit_hash): {:?}", commit_hash);

        let (org, repo_name) = if !self.url.is_empty() {
            let gurl = GitUrl::parse(&self.url)?;
            (gurl.owner.unwrap_or_default(), gurl.name)
        } else {
            ("".to_string(), format!("{:?}", self.lang.kind))
        };
        info!("add repository... {}", self.root.display());
        let repo_file = strip_tmp(&self.root).display().to_string();
        let mut repo_data = NodeData {
            name: format!("{}/{}", org, repo_name),
            file: repo_file.clone(),
            hash: Some(commit_hash.to_string()),
            ..Default::default()
        };
        repo_data.add_source_link(&self.url);
        graph.add_node_with_parent(NodeType::Repository, repo_data, NodeType::Repository, "");

        debug!("add language...");
        let lang_data = NodeData {
            name: self.lang.kind.to_string(),
            file: strip_tmp(&self.root).display().to_string(),
            ..Default::default()
        };
        graph.add_node_with_parent(
            NodeType::Language,
            lang_data,
            NodeType::Repository,
            &repo_file,
        );

        let mut stats = std::collections::HashMap::new();
        stats.insert("repository".to_string(), 1);
        stats.insert("language".to_string(), 1);
        self.send_status_with_stats(stats);

        self.send_status_progress(100, 100);
        Ok(())
    }
    fn process_classes<G: Graph>(&self, graph: &mut G, filez: &[(String, String)]) -> Result<()> {
        self.send_status_update("process_classes", 8);
        let mut i = 0;
        let mut class_count = 0;
        let total = filez.len();

        info!("=> get_classes...");
        for (filename, code) in filez {
            i += 1;
            if i % 20 == 0 || i == total {
                self.send_status_progress(i, total);
            }

            if !self.lang.kind.is_source_file(&filename) {
                continue;
            }
            let qo = self
                .lang
                .q(&self.lang.lang().class_definition_query(), &NodeType::Class);
            let classes = self
                .lang
                .collect_classes::<G>(&qo, &code, &filename, &graph)?;
            class_count += classes.len();
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

        let mut stats = std::collections::HashMap::new();
        stats.insert("classes".to_string(), class_count);
        self.send_status_with_stats(stats);

        info!("=> got {} classes", class_count);
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
        let mut instance_count = 0;
        let mut trait_count = 0;
        let total = filez.len();

        info!("=> get_instances...");
        for (filename, code) in filez {
            cnt += 1;
            if cnt % 20 == 0 || cnt == total {
                self.send_status_progress(cnt, total);
            }

            if !self.lang.kind.is_source_file(&filename) {
                continue;
            }
            let q = self.lang.lang().instance_definition_query();
            let instances =
                self.lang
                    .get_query_opt::<G>(q, &code, &filename, NodeType::Instance)?;
            instance_count += instances.len();
            graph.add_instances(instances);
        }

        info!("=> get_traits...");
        for (filename, code) in filez {
            if !self.lang.kind.is_source_file(&filename) {
                continue;
            }
            let traits = self.lang.get_traits::<G>(&code, &filename)?;
            trait_count += traits.len();

            for tr in traits {
                graph.add_node_with_parent(NodeType::Trait, tr.clone(), NodeType::File, &tr.file);
            }
        }

        let mut stats = std::collections::HashMap::new();
        stats.insert("instances".to_string(), instance_count);
        stats.insert("traits".to_string(), trait_count);
        self.send_status_with_stats(stats);

        info!("=> got {} traits", trait_count);
        Ok(())
    }
    fn process_data_models<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_data_models", 10);
        let mut i = 0;
        let mut datamodel_count = 0;
        let total = filez.len();

        info!("=> get_structs...");
        for (filename, code) in filez {
            i += 1;
            if i % 20 == 0 || i == total {
                self.send_status_progress(i, total);
            }

            if !self.lang.kind.is_source_file(&filename) {
                continue;
            }
            if let Some(dmf) = self.lang.lang().data_model_path_filter() {
                if !filename.contains(&dmf) {
                    continue;
                }
            }
            let q = self.lang.lang().data_model_query();
            let structs = self
                .lang
                .get_query_opt::<G>(q, &code, &filename, NodeType::DataModel)?;
            datamodel_count += structs.len();

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

        let mut stats = std::collections::HashMap::new();
        stats.insert("data_models".to_string(), datamodel_count);
        self.send_status_with_stats(stats);

        info!("=> got {} data models", datamodel_count);
        Ok(())
    }
    async fn process_functions_and_tests<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_functions_and_tests", 11);
        let mut i = 0;
        let mut function_count = 0;
        let mut test_count = 0;
        let total = filez.len();

        info!("=> get_functions_and_tests...");
        for (filename, code) in filez {
            i += 1;
            if i % 10 == 0 || i == total {
                self.send_status_progress(i, total);
            }

            if !self.lang.kind.is_source_file(&filename) {
                continue;
            }
            let (funcs, tests) =
                self.lang
                    .get_functions_and_tests(&code, &filename, graph, &self.lsp_tx)?;
            function_count += funcs.len();
            graph.add_functions(funcs.clone());
            test_count += tests.len();

            for test in tests {
                graph.add_node_with_parent(
                    NodeType::Test,
                    test.0.clone(),
                    NodeType::File,
                    &test.0.file,
                );
            }
        }

        let mut stats = std::collections::HashMap::new();
        stats.insert("functions".to_string(), function_count);
        stats.insert("tests".to_string(), test_count);
        self.send_status_with_stats(stats);

        info!("=> got {} functions and tests", function_count + test_count);
        Ok(())
    }
    fn process_pages_and_templates<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_pages_and_templates", 12);
        let mut i = 0;
        let mut page_count = 0;
        let mut template_count = 0;
        let total = filez.len();

        info!("=> get_pages");
        for (filename, code) in filez {
            i += 1;
            if i % 10 == 0 || i == total {
                self.send_status_progress(i, total);
            }

            if !self.lang.kind.is_source_file(&filename) {
                continue;
            }
            if self.lang.lang().is_router_file(&filename, &code) {
                let pages = self.lang.get_pages(&code, &filename, &self.lsp_tx, graph)?;
                page_count += pages.len();
                graph.add_pages(pages);
            }
        }
        info!("=> got {} pages", page_count);

        if self.lang.lang().use_extra_page_finder() {
            info!("=> get_extra_pages");
            let closure = |fname: &str| self.lang.lang().is_extra_page(fname);
            let extra_pages = self.collect_extra_pages(closure)?;
            let extra_page_count = extra_pages.len();
            info!("=> got {} extra pages", extra_page_count);
            page_count += extra_page_count;

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

        let mut _i = 0;
        info!("=> get_component_templates");
        for (filename, code) in filez {
            if let Some(ext) = self.lang.lang().template_ext() {
                if filename.ends_with(ext) {
                    let template_edges = self
                        .lang
                        .get_component_templates::<G>(&code, &filename, &graph)?;
                    template_count += template_edges.len();
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

        let mut stats = std::collections::HashMap::new();
        stats.insert("pages".to_string(), page_count);
        stats.insert("templates".to_string(), template_count);
        self.send_status_with_stats(stats);

        info!("=> got {} component templates/styles", template_count);

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
        let mut _i = 0;
        let mut endpoint_count = 0;
        let total = filez.len();

        info!("=> get_endpoints...");
        for (filename, code) in filez {
            _i += 1;
            if _i % 10 == 0 || _i == total {
                self.send_status_progress(_i, total);
            }

            if !self.lang.kind.is_source_file(&filename) {
                continue;
            }
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
            endpoint_count += endpoints.len();

            graph.add_endpoints(endpoints);
        }

        let mut stats = std::collections::HashMap::new();
        stats.insert("endpoints".to_string(), endpoint_count);
        self.send_status_with_stats(stats);

        info!("=> got {} endpoints", endpoint_count);

        info!("=> get_endpoint_groups...");
        let mut _endpoint_group_count = 0;
        for (filename, code) in filez {
            if self.lang.lang().is_test_file(&filename) {
                continue;
            }
            let q = self.lang.lang().endpoint_group_find();
            let endpoint_groups =
                self.lang
                    .get_query_opt::<G>(q, &code, &filename, NodeType::Endpoint)?;
            _endpoint_group_count += endpoint_groups.len();
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
        stats: &mut std::collections::HashMap<String, usize>,
    ) -> Result<()> {
        let mut _i = 0;
        let mut import_edges_count = 0;
        info!("=> get_import_edges...");
        for (filename, code) in filez {
            if let Some(import_query) = self.lang.lang().imports_query() {
                let q = self.lang.q(&import_query, &NodeType::Import);
                let import_edges =
                    self.lang
                        .collect_import_edges(&q, &code, &filename, graph, &self.lsp_tx)?;
                for edge in import_edges {
                    graph.add_edge(edge);
                    import_edges_count += 1;
                    _i += 1;
                }
            }
        }
        stats.insert("import_edges".to_string(), import_edges_count);
        info!("=> got {} import edges", import_edges_count);

        self.send_status_update("process_integration_tests", 14);

        let mut _i = 0;
        let mut cnt = 0;
        let mut integration_test_count = 0;
        let total = filez.len();

        if self.lang.lang().use_integration_test_finder() {
            info!("=> get_integration_tests...");
            for (filename, code) in filez {
                cnt += 1;
                if cnt % 10 == 0 || cnt == total {
                    self.send_status_progress(cnt, total);
                }

                if !self.lang.lang().is_test_file(&filename) {
                    continue;
                }
                let int_tests = self.lang.collect_integration_tests(code, filename, graph)?;
                integration_test_count += int_tests.len();
                _i += int_tests.len();
                for (nd, tt, edge_opt) in int_tests {
                    graph.add_test_node(nd, tt, edge_opt);
                }
            }
        }
        stats.insert("integration_tests".to_string(), integration_test_count);
        info!("=> got {} integration tests", _i);

        let skip_calls = std::env::var("DEV_SKIP_CALLS").is_ok();
        if skip_calls {
            info!("=> Skipping function_calls...");
        } else {
            self.send_status_update("process_function_calls", 15);
            _i = 0;
            let mut cnt = 0;
            let mut function_call_count = 0;
            let total = filez.len();

            info!("=> get_function_calls...");
            for (filename, code) in filez {
                cnt += 1;
                if cnt % 5 == 0 || cnt == total {
                    self.send_status_progress(cnt, total);
                }

                let all_calls = self
                    .lang
                    .get_function_calls(&code, &filename, graph, &self.lsp_tx)
                    .await?;
                function_call_count += all_calls.0.len();
                _i += all_calls.0.len();
                graph.add_calls(all_calls);
            }
            stats.insert("function_calls".to_string(), function_call_count);
            info!("=> got {} function calls", _i);
        }

        self.lang
            .lang()
            .clean_graph(&mut |parent_type, child_type, child_meta_key| {
                graph.filter_out_nodes_without_children(parent_type, child_type, child_meta_key);
            });

        Ok(())
    }
}
