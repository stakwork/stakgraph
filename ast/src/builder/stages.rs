use super::utils::*;
use crate::lang::{
    asg::{NodeData, TestRecord},
    graphs::{Graph, NodeType},
    linker::link_tests,
};
use crate::repo::Repo;
use rayon::prelude::*;
use shared::error::Result;
use std::{any::type_name, collections::HashMap, path::Path};
use tracing::{debug, info};

impl Repo {
    pub fn process_libraries<G: Graph + Sync>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_libraries", 3);
        let use_parallel = !type_name::<G>().contains("Neo4jGraph") && self.lsp_tx.is_none();
        let mut i = 0;
        let mut lib_count = 0;
        let pkg_files_res: Vec<_> = if use_parallel {
            filez
                .par_iter()
                .filter(|(f, _)| self.lang.kind.is_package_file(f))
                .map(|(pkg_file, code)| {
                    let mut file_data = self.prepare_file_data(pkg_file, code);
                    file_data.meta.insert("lib".to_string(), "true".to_string());
                    let (parent_type, parent_file) = self.get_parent_info(Path::new(pkg_file));
                    let libs = self.lang.get_libs::<G>(code, pkg_file)?;
                    Ok((pkg_file, file_data, parent_type, parent_file, libs))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            filez
                .iter()
                .filter(|(f, _)| self.lang.kind.is_package_file(f))
                .map(|(pkg_file, code)| {
                    let mut file_data = self.prepare_file_data(pkg_file, code);
                    file_data.meta.insert("lib".to_string(), "true".to_string());
                    let (parent_type, parent_file) = self.get_parent_info(Path::new(pkg_file));
                    let libs = self.lang.get_libs::<G>(code, pkg_file)?;
                    Ok((pkg_file, file_data, parent_type, parent_file, libs))
                })
                .collect::<Result<Vec<_>>>()?
        };

        let total_pkg_files = pkg_files_res.len();
        for (pkg_file, file_data, parent_type, parent_file, libs) in pkg_files_res {
            i += 1;
            if i % 2 == 0 || i == total_pkg_files {
                self.send_status_progress(i, total_pkg_files, 5);
            }

            info!("=> get_packages in... {:?}", pkg_file);

            graph.add_node_with_parent(&NodeType::File, &file_data, &parent_type, &parent_file);

            lib_count += libs.len();

            for lib in libs {
                graph.add_node_with_parent(&NodeType::Library, &lib, &NodeType::File, pkg_file);
            }
        }

        let mut stats = HashMap::new();
        stats.insert("libraries".to_string(), lib_count);
        self.send_status_with_stats(stats);

        self.send_status_progress(100, 100, 3);
        info!("=> got {} libs", lib_count);
        Ok(())
    }
    pub fn process_import_sections<G: Graph + Sync>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_imports", 4);
        let use_parallel = !type_name::<G>().contains("Neo4jGraph") && self.lsp_tx.is_none();
        let mut i = 0;
        let mut import_count = 0;
        let total = filez.len();

        info!("=> get_imports...");

        let lang = &self.lang;
        let results: Vec<_> = if use_parallel {
            filez
                .par_iter()
                .map(|(filename, code)| {
                    let imports = lang.get_imports::<G>(code, filename)?;
                    let import_section = combine_import_sections(imports);
                    Ok(import_section)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            filez
                .iter()
                .map(|(filename, code)| {
                    let imports = lang.get_imports::<G>(code, filename)?;
                    let import_section = combine_import_sections(imports);
                    Ok(import_section)
                })
                .collect::<Result<Vec<_>>>()?
        };

        for import_section in results {
            i += 1;
            if i % 20 == 0 || i == total {
                self.send_status_progress(i, total, 6);
            }
            import_count += import_section.len();

            for import in import_section {
                graph.add_node_with_parent(
                    &NodeType::Import,
                    &import,
                    &NodeType::File,
                    &import.file,
                );
            }
        }

        let mut stats = HashMap::new();
        stats.insert("imports".to_string(), import_count);
        self.send_status_with_stats(stats);
        self.send_status_progress(100, 100, 4);
        info!("=> got {} import sections", import_count);
        Ok(())
    }
    pub fn process_variables<G: Graph + Sync>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_variables", 5);
        let use_parallel = !type_name::<G>().contains("Neo4jGraph") && self.lsp_tx.is_none();
        let mut i = 0;
        let mut var_count = 0;
        let total = filez.len();

        info!("=> get_vars...");

        let lang = &self.lang;
        let results: Vec<_> = if use_parallel {
            filez
                .par_iter()
                .map(|(filename, code)| {
                    let variables = lang.get_vars::<G>(code, filename)?;
                    Ok(variables)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            filez
                .iter()
                .map(|(filename, code)| {
                    let variables = lang.get_vars::<G>(code, filename)?;
                    Ok(variables)
                })
                .collect::<Result<Vec<_>>>()?
        };

        for variables in results {
            i += 1;
            if i % 20 == 0 || i == total {
                self.send_status_progress(i, total, 7);
            }

            var_count += variables.len();
            for variable in variables {
                graph.add_node_with_parent(
                    &NodeType::Var,
                    &variable,
                    &NodeType::File,
                    &variable.file,
                );
            }
        }

        let mut stats = HashMap::new();
        stats.insert("variables".to_string(), var_count);
        self.send_status_with_stats(stats);
        self.send_status_progress(100, 100, 5);

        info!("=> got {} all vars", var_count);
        Ok(())
    }
    pub fn process_instances_and_traits<G: Graph + Sync>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_instances_and_traits", 7);
        let use_parallel = !type_name::<G>().contains("Neo4jGraph") && self.lsp_tx.is_none();
        let mut cnt = 0;
        let mut instance_count = 0;
        let mut trait_count = 0;
        let total = filez.len();

        info!("=> get_instances...");

        let lang = &self.lang;

        let results: Vec<_> = if use_parallel {
            filez
                .par_iter()
                .filter(|(filename, _)| lang.kind.is_source_file(filename))
                .map(|(filename, code)| {
                    let q_inst = lang.lang().instance_definition_query();
                    let instances =
                        lang.get_query_opt::<G>(q_inst, code, filename, NodeType::Instance)?;

                    let traits = lang.get_traits::<G>(code, filename)?;

                    Ok((instances, traits))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            filez
                .iter()
                .filter(|(filename, _)| lang.kind.is_source_file(filename))
                .map(|(filename, code)| {
                    let q_inst = lang.lang().instance_definition_query();
                    let instances =
                        lang.get_query_opt::<G>(q_inst, code, filename, NodeType::Instance)?;

                    let traits = lang.get_traits::<G>(code, filename)?;

                    Ok((instances, traits))
                })
                .collect::<Result<Vec<_>>>()?
        };
        info!("=> get_traits...");

        for (instances, traits) in results {
            cnt += 1;
            if cnt % 20 == 0 || cnt == total {
                self.send_status_progress(cnt, total, 9);
            }

            instance_count += instances.len();
            graph.add_instances(&instances);

            trait_count += traits.len();
            for tr in traits {
                graph.add_node_with_parent(&NodeType::Trait, &tr, &NodeType::File, &tr.file);
            }
        }

        let mut stats = HashMap::new();
        stats.insert("instances".to_string(), instance_count);
        stats.insert("traits".to_string(), trait_count);
        self.send_status_with_stats(stats);
        self.send_status_progress(100, 100, 7);

        info!("=> got {} traits", trait_count);
        Ok(())
    }
    pub fn process_data_models<G: Graph + Sync>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_data_models", 8);
        let use_parallel = !std::any::type_name::<G>().contains("Neo4jGraph") && self.lsp_tx.is_none();
        let mut i = 0;
        let mut datamodel_count = 0;
        let total = filez.len();

        info!("=> get_structs...");
        info!("=> get_structs...");

        let lang = &self.lang;
        let graph_ref = &*graph;

        let results: Vec<_> = if use_parallel {
            filez
                .par_iter()
                .filter(|(filename, _)| {
                    if !lang.kind.is_source_file(filename) {
                        return false;
                    }
                    if let Some(dmf) = lang.lang().data_model_path_filter() {
                        if !filename.contains(&dmf) {
                            return false;
                        }
                    }
                    true
                })
                .map(|(filename, code)| {
                    let structs = lang.get_data_models::<G>(code, filename)?;
                    let mut all_edges = Vec::new();
                    for dm in &structs {
                        let edges = lang.collect_class_contains_datamodel_edge(dm, graph_ref)?;
                        all_edges.extend(edges);
                    }
                    Ok((structs, all_edges))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            filez
                .iter()
                .filter(|(filename, _)| {
                    if !lang.kind.is_source_file(filename) {
                        return false;
                    }
                    if let Some(dmf) = lang.lang().data_model_path_filter() {
                        if !filename.contains(&dmf) {
                            return false;
                        }
                    }
                    true
                })
                .map(|(filename, code)| {
                    let structs = lang.get_data_models::<G>(code, filename)?;
                    let mut all_edges = Vec::new();
                    for dm in &structs {
                        let edges = lang.collect_class_contains_datamodel_edge(dm, graph_ref)?;
                        all_edges.extend(edges);
                    }
                    Ok((structs, all_edges))
                })
                .collect::<Result<Vec<_>>>()?
        };

        for (structs, edges) in results {
            i += 1;
            if i % 20 == 0 || i == total {
                self.send_status_progress(i, total, 10);
            }

            datamodel_count += structs.len();

            for st in &structs {
                graph.add_node_with_parent(&NodeType::DataModel, st, &NodeType::File, &st.file);
            }
            for edge in edges {
                graph.add_edge(&edge);
            }
        }

        let mut stats = HashMap::new();
        stats.insert("data_models".to_string(), datamodel_count);
        self.send_status_with_stats(stats);
        self.send_status_progress(100, 100, 8);

        info!("=> got {} data models", datamodel_count);
        Ok(())
    }
    pub async fn process_functions_and_tests<G: Graph + Sync>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_functions_and_tests", 9);
        let use_parallel = !std::any::type_name::<G>().contains("Neo4jGraph") && self.lsp_tx.is_none();
        let mut i = 0;
        let mut function_count = 0;
        let mut test_count = 0;
        let total = filez.len();

        info!("=> get_functions_and_tests...");

        let lang = &self.lang;
        let lsp_tx = &self.lsp_tx;
        let graph_ref = &*graph;

        let results: Vec<_> = if use_parallel {
            filez
                .par_iter()
                .filter(|(filename, _)| lang.kind.is_source_file(filename))
                .map(|(filename, code)| {
                    lang.get_functions_and_tests(code, filename, graph_ref, lsp_tx)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            filez
                .iter()
                .filter(|(filename, _)| lang.kind.is_source_file(filename))
                .map(|(filename, code)| {
                    lang.get_functions_and_tests(code, filename, graph_ref, lsp_tx)
                })
                .collect::<Result<Vec<_>>>()?
        };

        for (funcs, tests) in results {
            i += 1;
            if i % 10 == 0 || i == total {
                self.send_status_progress(i, total, 11);
            }

            function_count += funcs.len();
            graph.add_functions(&funcs);

            test_count += tests.len();
            graph.add_tests(&tests);
        }

        let mut stats = HashMap::new();
        stats.insert("functions".to_string(), function_count);
        stats.insert("tests".to_string(), test_count);
        self.send_status_with_stats(stats);
        self.send_status_progress(100, 100, 9);

        info!("=> got {} functions and tests", function_count + test_count);
        Ok(())
    }
    pub fn process_pages_and_templates<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_pages_and_templates", 10);
        let mut i = 0;
        let mut page_count = 0;
        let mut template_count = 0;
        let total = filez.len();

        info!("=> get_pages");
        for (filename, code) in filez {
            i += 1;
            if i % 10 == 0 || i == total {
                self.send_status_progress(i, total, 12);
            }

            if self.lang.lang().is_router_file(filename, code) {
                let pages = self.lang.get_pages(code, filename, &self.lsp_tx, graph)?;
                page_count += pages.len();
                graph.add_pages(&pages);
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
                if let Some((page_node, edge)) = self.lang.lang().extra_page_finder(
                    &pagepath,
                    &|name, filename| {
                        graph.find_node_by_name_and_file_end_with(
                            NodeType::Function,
                            name,
                            filename,
                        )
                    },
                    &|filename| graph.find_nodes_by_file_ends_with(NodeType::Function, filename),
                ) {
                    let code = filez
                        .iter()
                        .find(|(f, _)| f.ends_with(&pagepath) || pagepath.ends_with(f))
                        .map(|(_, c)| c.as_str())
                        .unwrap_or("");
                    let mut page_node = page_node;
                    if page_node.body.is_empty() {
                        page_node.body = code.to_string();
                    }
                    graph.add_page((page_node, edge));
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
                        .get_component_templates::<G>(code, filename, graph)?;
                    template_count += template_edges.len();
                    for edge in template_edges {
                        let mut page = NodeData::name_file(
                            &edge.source.node_data.name,
                            &edge.source.node_data.file,
                        );
                        page.body = code.clone();
                        graph.add_node_with_parent(
                            &NodeType::Page,
                            &page,
                            &NodeType::File,
                            &edge.source.node_data.file,
                        );
                        graph.add_edge(&edge);
                    }
                }
            }
        }

        let mut stats = HashMap::new();
        stats.insert("pages".to_string(), page_count);
        stats.insert("templates".to_string(), template_count);
        self.send_status_with_stats(stats);
        self.send_status_progress(100, 100, 10);

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
                    graph.add_edge(&edge);
                }
            }
            info!("=> got {} page component renders", page_renders_count);
        }

        Ok(())
    }
    pub fn process_endpoints<G: Graph + Sync>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
    ) -> Result<()> {
        self.send_status_update("process_endpoints", 11);
        let use_parallel = !std::any::type_name::<G>().contains("Neo4jGraph") && self.lsp_tx.is_none();
        info!("=> get_endpoints...");

        let lang = &self.lang;
        let lsp_tx = &self.lsp_tx;
        let graph_ref = &*graph;

        let results: Vec<_> = if use_parallel {
            filez
                .par_iter()
                .map(|(filename, code)| {
                    let mut endpoints = Vec::new();
                    let mut endpoint_groups = Vec::new();

                    if lang.kind.is_source_file(filename) {
                        let pass_filter = if let Some(epf) = lang.lang().endpoint_path_filter() {
                            filename.contains(&epf)
                        } else {
                            true
                        };

                        if pass_filter && !lang.lang().is_test_file(filename) {
                            debug!("get_endpoints in {:?}", filename);
                            endpoints =
                                lang.collect_endpoints(code, filename, Some(graph_ref), lsp_tx)?;
                        }
                    }

                    if !lang.lang().is_test_file(filename) {
                        let q = lang.lang().endpoint_group_find();
                        endpoint_groups =
                            lang.get_query_opt::<G>(q, code, filename, NodeType::Endpoint)?;
                    }

                    Ok((endpoints, endpoint_groups))
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            filez
                .iter()
                .map(|(filename, code)| {
                    let mut endpoints = Vec::new();
                    let mut endpoint_groups = Vec::new();

                    if lang.kind.is_source_file(filename) {
                        let pass_filter = if let Some(epf) = lang.lang().endpoint_path_filter() {
                            filename.contains(&epf)
                        } else {
                            true
                        };

                        if pass_filter && !lang.lang().is_test_file(filename) {
                            debug!("get_endpoints in {:?}", filename);
                            endpoints =
                                lang.collect_endpoints(code, filename, Some(graph_ref), lsp_tx)?;
                        }
                    }

                    if !lang.lang().is_test_file(filename) {
                        let q = lang.lang().endpoint_group_find();
                        endpoint_groups =
                            lang.get_query_opt::<G>(q, code, filename, NodeType::Endpoint)?;
                    }

                    Ok((endpoints, endpoint_groups))
                })
                .collect::<Result<Vec<_>>>()?
        };

        info!("=> process endpoints and groups Results...");

        let mut all_endpoint_groups = Vec::new();
        for (endpoints, endpoint_groups) in results {
            graph.add_endpoints(&endpoints);
            all_endpoint_groups.extend(endpoint_groups);
        }

        if !all_endpoint_groups.is_empty() {
            let _ = graph.process_endpoint_groups(&all_endpoint_groups, lang);
        }

        if self.lang.lang().use_data_model_within_finder() {
            info!("=> get_data_models_within...");
            graph.get_data_models_within(&self.lang);
        }
        Ok(())
    }
    pub async fn finalize_graph<G: Graph>(
        &self,
        graph: &mut G,
        filez: &[(String, String)],
        stats: &mut HashMap<String, usize>,
    ) -> Result<()> {
        let mut _i = 0;
        let mut import_edges_count = 0;
        info!("=> get_import_edges...");
        for (filename, code) in filez {
            if let Some(import_query) = self.lang.lang().imports_query() {
                let q = self.lang.q(&import_query, &NodeType::Import);
                let import_edges =
                    self.lang
                        .collect_import_edges(&q, code, filename, graph, &self.lsp_tx)?;
                for edge in import_edges {
                    graph.add_edge(&edge);
                    import_edges_count += 1;
                    _i += 1;
                }
            }
        }
        stats.insert("import_edges".to_string(), import_edges_count);
        info!("=> got {} import edges", import_edges_count);

        self.send_status_update("process_integration_tests", 12);

        let mut _i = 0;
        let mut cnt = 0;
        let mut integration_test_count = 0;
        let total = filez.len();

        if self.lang.lang().use_integration_test_finder() {
            info!("=> get_integration_tests...");
            for (filename, code) in filez {
                cnt += 1;
                if cnt % 10 == 0 || cnt == total {
                    self.send_status_progress(cnt, total, 12);
                }

                if !self.lang.lang().is_test_file(filename) {
                    continue;
                }
                let int_tests = self.lang.collect_integration_tests(code, filename, graph)?;
                integration_test_count += int_tests.len();
                _i += int_tests.len();
                let test_records: Vec<TestRecord> = int_tests
                    .into_iter()
                    .map(|(nd, tt, edge_opt)| TestRecord::new(nd, tt, edge_opt))
                    .collect();
                graph.add_tests(&test_records);
            }
        }
        stats.insert("integration_tests".to_string(), integration_test_count);
        info!("=> got {} integration tests", _i);

        if self.skip_calls {
            info!("=> Skipping function_calls...");
        } else {
            self.send_status_update("process_function_calls", 13);
            _i = 0;
            let mut cnt = 0;
            let mut function_call_count = 0;
            let total = filez.len();

            info!("=> get_function_calls...");
            for (filename, code) in filez {
                cnt += 1;
                if cnt % 5 == 0 || cnt == total {
                    self.send_status_progress(cnt, total, 13);
                }

                let all_calls = self
                    .lang
                    .get_function_calls(code, filename, graph, &self.lsp_tx)
                    .await?;
                function_call_count += all_calls.0.len();
                _i += all_calls.0.len();
                graph.add_calls(
                    (&all_calls.0, &all_calls.1, &all_calls.2, &all_calls.3),
                    &self.lang,
                );
            }
            stats.insert("function_calls".to_string(), function_call_count);
            info!("=> got {} function calls", _i);
        }

        link_tests(graph)?;

        self.lang
            .lang()
            .clean_graph(&mut |parent_type, child_type, operation| match operation {
                "operand" => {
                    graph.filter_out_nodes_without_children(parent_type, child_type, operation);
                }
                "deduplicate" => {
                    graph.deduplicate_nodes(parent_type, child_type, operation);
                }
                _ => {
                    graph.filter_out_nodes_without_children(parent_type, child_type, operation);
                }
            });

        Ok(())
    }
}
