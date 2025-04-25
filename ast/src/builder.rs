use super::repo::{check_revs_files, Repo};
use crate::lang::graphs::Graph;
use crate::lang::{asg::NodeData, graphs::NodeType};
use crate::lang::{ArrayGraph, BTreeMapGraph, Node};
use crate::utils::create_node_key;
use anyhow::{Ok, Result};
use git_url_parse::GitUrl;
use lsp::{git::get_commit_hash, strip_root, Cmd as LspCmd, DidOpen};
use std::collections::HashSet;
use std::path::PathBuf;
use tokio::fs;
use tracing::{debug, info};

const MAX_FILE_SIZE: u64 = 100_000; // 100kb max file size

impl Repo {
    pub async fn build_graph(&self) -> Result<ArrayGraph> {
        self.build_graph_inner().await
    }

    pub async fn build_graph_btree(&self) -> Result<BTreeMapGraph> {
        self.build_graph_inner().await
    }
    pub async fn build_graph_inner<G: Graph>(&self) -> Result<G> {
        let mut graph = G::new();

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

        debug!("collecting dirs...");
        let dirs = self.collect_dirs()?;
        let files_1 = self.collect()?;
        let files: Vec<PathBuf> = files_1
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

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

        let mut i = dirs_not_empty.len();
        info!("adding {} dirs... {:?}", i, dirs_not_empty);
        let mut processed_dirs = HashSet::new();

        for dir in &dirs_not_empty {
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

        info!("parsing {} files...", files.len());
        for filepath in &files {
            let filename = strip_root(filepath, &self.root);
            let meta = fs::metadata(&filepath).await?;
            let code = if meta.len() > MAX_FILE_SIZE {
                debug!("Skipping large file: {:?}", filename);
                "".to_string()
            } else {
                std::fs::read_to_string(&filepath)?
            };

            let path = filename.display().to_string();

            if graph.find_nodes_by_name(NodeType::File, &path).len() > 0 {
                continue;
            }
            if path.ends_with(self.lang.kind.pkg_file()) {
                continue;
            }
            let file_data = self.prepare_file_data(&path, &code);

            let (parent_type, parent_file) = if path.contains('/') {
                let mut paths: Vec<&str> = path.split('/').collect();
                paths.pop();
                (NodeType::Directory, paths.join("/"))
            } else {
                (NodeType::Repository, "main".to_string())
            };

            graph.add_node_with_parent(NodeType::File, file_data, parent_type, &parent_file);
        }

        let filez = fileys(&files, &self.root)?;
        info!("=> DidOpen...");
        if let Some(lsp_tx) = self.lsp_tx.as_ref() {
            for (filename, code) in &filez {
                let didopen = DidOpen {
                    file: filename.into(),
                    text: code.to_string(),
                    lang: self.lang.kind.clone(),
                };
                debug!("didopen: {:?}", didopen);
                let _ = LspCmd::DidOpen(didopen).send(&lsp_tx)?;
            }
        }

        i = 0;
        let pkg_files = filez
            .iter()
            .filter(|(f, _)| f.ends_with(self.lang.kind.pkg_file()));
        for (pkg_file, code) in pkg_files {
            info!("=> get_packages in... {:?}", pkg_file);

            let file_data = self.prepare_file_data(&pkg_file, code);

            let (parent_type, parent_file) = self.get_parent_info(&pkg_file);

            graph.add_node_with_parent(NodeType::File, file_data, parent_type, &parent_file);

            let libs = self.lang.get_libs::<G>(&code, &pkg_file)?;
            i += libs.len();

            for lib in libs {
                graph.add_node_with_parent(NodeType::Library, lib, NodeType::File, &pkg_file);
            }
        }
        info!("=> got {} libs", i);

        i = 0;
        info!("=> get_imports...");
        for (filename, code) in &filez {
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

        i = 0;
        info!("=> get_classes...");
        for (filename, code) in &filez {
            let classes = self.lang.get_classes::<G>(&code, &filename)?;
            i += classes.len();

            for class in classes {
                graph.add_node_with_parent(
                    NodeType::Class,
                    class.clone(),
                    NodeType::File,
                    &class.file,
                );
            }
        }
        info!("=> got {} classes", i);
        graph.class_inherits();
        graph.class_includes();

        info!("=> get_instances...");
        for (filename, code) in &filez {
            let q = self.lang.lang().instance_definition_query();
            let instances =
                self.lang
                    .get_query_opt::<G>(q, &code, &filename, NodeType::Instance)?;

            graph.add_instances(instances);
        }

        i = 0;
        info!("=> get_traits...");
        for (filename, code) in &filez {
            let traits = self.lang.get_traits::<G>(&code, &filename)?;
            i += traits.len();

            for tr in traits {
                graph.add_node_with_parent(NodeType::Trait, tr.clone(), NodeType::File, &tr.file);
            }
        }
        info!("=> got {} traits", i);

        i = 0;
        info!("=> get_structs...");
        for (filename, code) in &filez {
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
        }
        info!("=> got {} data models", i);

        // this also adds requests and data models inside
        i = 0;
        info!("=> get_functions_and_tests...");
        for (filename, code) in &filez {
            let (funcs, tests) =
                self.lang
                    .get_functions_and_tests(&code, &filename, &graph, &self.lsp_tx)?;
            i += funcs.len();

            graph.add_functions(funcs);
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

        // frontend "pages" (react-router-dom etc)
        i = 0;
        info!("=> get_pages");
        for (filename, code) in &filez {
            if self.lang.lang().is_router_file(&filename, &code) {
                let pages = self
                    .lang
                    .get_pages(&code, &filename, &self.lsp_tx, &graph)?;
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
                    let nd = NodeData::name_file(&pagename, &pagepath);
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

        // these are more subjective queries (with regex)
        i = 0;
        info!("=> get_endpoints...");
        for (filename, code) in &filez {
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
                    .collect_endpoints(&code, &filename, Some(&graph), &self.lsp_tx)?;
            i += endpoints.len();

            graph.add_endpoints(endpoints);
        }
        info!("=> got {} endpoints", i);

        info!("=> get_endpoint_groups...");
        for (filename, code) in &filez {
            if self.lang.lang().is_test_file(&filename) {
                continue;
            }
            let q = self.lang.lang().endpoint_group_find();
            let endpoint_groups =
                self.lang
                    .get_query_opt::<G>(q, &code, &filename, NodeType::Endpoint)?;
            let _ = graph.process_endpoint_groups(endpoint_groups, &self.lang);
        }

        // try again on the endpoints to add data models, if manual
        if self.lang.lang().use_data_model_within_finder() {
            info!("=> get_data_models_within...");
            graph.get_data_models_within(&self.lang);
        }

        i = 0;
        if self.lang.lang().use_integration_test_finder() {
            info!("=> get_integration_tests...");
            for (filename, code) in &filez {
                if !self.lang.lang().is_test_file(&filename) {
                    continue;
                }
                let int_tests = self
                    .lang
                    .collect_integration_tests(code, filename, &graph)?;
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
            i = 0;
            info!("=> get_function_calls...");
            for (filename, code) in &filez {
                let all_calls = self
                    .lang
                    .get_function_calls(&code, &filename, &graph, &self.lsp_tx)
                    .await?;
                i += all_calls.0.len();

                graph.add_calls(all_calls);
            }
            info!("=> got {} function calls", i);
        }

        //Clean graph by filtering out classes without methods
        self.lang
            .lang()
            .clean_graph(&mut |parent_type, child_type, child_meta_key| {
                graph.filter_out_nodes_without_children(parent_type, child_type, child_meta_key);
            });

        // filter by revs
        graph = filter_by_revs(&self.root.to_str().unwrap(), self.revs.clone(), graph);

        // prefix the "file" of each node and edge with the root
        graph.prefix_paths(&self.root_less_tmp());

        println!("done!");
        let (num_of_nodes, num_of_edges) = graph.get_graph_size();
        println!(
            "Returning Graph with {} nodes and {} edges",
            num_of_nodes, num_of_edges
        );
        Ok(graph)
    }
    fn root_less_tmp(&self) -> String {
        let mut ret = self.root.display().to_string();
        if ret.starts_with("/tmp/") {
            ret.drain(0..5);
            ret
        } else {
            ret
        }
    }
    fn prepare_file_data(&self, path: &str, code: &str) -> NodeData {
        let mut file_data = NodeData::in_file(path);
        let filename = path.split('/').last().unwrap_or(path);
        file_data.name = filename.to_string();

        let skip_file_content = std::env::var("DEV_SKIP_FILE_CONTENT").is_ok();
        if !skip_file_content {
            file_data.body = code.to_string();
        }
        file_data.hash = Some(sha256::digest(&file_data.body));
        file_data
    }
    fn get_parent_info(&self, path: &str) -> (NodeType, String) {
        if path.contains('/') {
            let mut paths: Vec<&str> = path.split('/').collect();
            paths.pop();
            (NodeType::Directory, paths.join("/"))
        } else {
            (NodeType::Repository, "main".to_string())
        }
    }
}

fn filter_by_revs<G: Graph>(root: &str, revs: Vec<String>, graph: G) -> G {
    if revs.is_empty() {
        return graph;
    }
    match check_revs_files(root, revs) {
        Some(final_filter) => graph.create_filtered_graph(&final_filter),
        None => graph,
    }
}

// (file, code)
fn fileys(files: &Vec<PathBuf>, root: &PathBuf) -> Result<Vec<(String, String)>> {
    let mut ret = Vec::new();
    for f in files {
        let filename = strip_root(&f, root).display().to_string();
        let code = std::fs::read_to_string(&f)?;
        ret.push((filename, code));
    }
    Ok(ret)
}

fn _filenamey(f: &PathBuf) -> String {
    let full = f.display().to_string();
    if !f.starts_with("/tmp/") {
        return full;
    }
    let mut parts = full.split("/").collect::<Vec<&str>>();
    parts.drain(0..4);
    parts.join("/")
}

pub fn get_page_name(path: &str) -> Option<String> {
    let parts = path.split("/").collect::<Vec<&str>>();
    if parts.last().is_none() {
        return None;
    }
    Some(parts.last().unwrap().to_string())
}

pub fn combine_imports(nodes: Vec<NodeData>) -> Vec<NodeData> {
    if nodes.is_empty() {
        return Vec::new();
    }
    let import_name = create_node_key(&Node::new(NodeType::Import, nodes[0].clone()));
    let mut combined_body = String::new();
    let mut current_position = nodes[0].start;
    for (i, node) in nodes.iter().enumerate() {
        // Add extra newlines if there's a gap between this node and the previous position
        if node.start > current_position {
            let extra_newlines = node.start - current_position;
            combined_body.push_str(&"\n".repeat(extra_newlines));
        }
        // Add the node body
        combined_body.push_str(&node.body);
        // Add a newline separator between nodes (except after the last one)
        if i < nodes.len() - 1 {
            combined_body.push('\n');
            current_position = node.end + 1; // +1 for the newline we just added
        } else {
            current_position = node.end;
        }
    }
    // Use the file from the first node
    let file = if !nodes.is_empty() {
        nodes[0].file.clone()
    } else {
        String::new()
    };

    vec![NodeData {
        name: import_name,
        file,
        body: combined_body,
        start: nodes[0].start,
        end: nodes.last().unwrap().end,
        ..Default::default()
    }]
}
