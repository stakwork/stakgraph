use super::utils::trim_quotes;
use crate::lang::{graphs::Graph, *};
use anyhow::Result;
use lsp::{Cmd as LspCmd, Position, Res as LspRes};
use streaming_iterator::StreamingIterator;
use tree_sitter::Node as TreeNode;
impl Lang {
    pub fn collect<G: Graph>(
        &self,
        q: &Query,
        code: &str,
        file: &str,
        nt: NodeType,
    ) -> Result<Vec<NodeData>> {
        let tree = self.lang.parse(&code, &nt)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            let another = match nt {
                NodeType::Library => vec![self.format_library(&m, code, file, q)?],
                NodeType::Import => self.format_imports(&m, code, file, q)?,
                NodeType::Instance => vec![self.format_instance(&m, code, file, q)?],
                NodeType::Trait => vec![self.format_trait(&m, code, file, q)?],
                // req and endpoint are the same format in the query templates
                NodeType::Request | NodeType::Endpoint => self
                    .format_endpoint::<G>(&m, code, file, q, None, &None)?
                    .into_iter()
                    .map(|(nd, _e)| nd)
                    .collect(),
                NodeType::DataModel => vec![self.format_data_model(&m, code, file, q)?],
                NodeType::Var => self.format_variables(&m, code, file, q)?,
                _ => return Err(anyhow::anyhow!("collect: {nt:?} not implemented")),
            };
            res.extend(another);
        }
        Ok(res)
    }

    pub fn collect_classes<G: Graph>(
        &self,
        q: &Query,
        code: &str,
        file: &str,
        graph: &G,
    ) -> Result<Vec<(NodeData, Vec<Edge>)>> {
        let tree = self.lang.parse(&code, &NodeType::Class)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            let (cls, edges) = self.format_class_with_associations(&m, code, file, q, graph)?;
            res.push((cls, edges));
        }
        Ok(res)
    }

    pub fn collect_pages<G: Graph>(
        &self,
        q: &Query,
        code: &str,
        file: &str,
        lsp_tx: &Option<CmdSender>,
        graph: &G,
    ) -> Result<Vec<(NodeData, Vec<Edge>)>> {
        let tree = self.lang.parse(&code, &NodeType::Page)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            let page = self.format_page(&m, code, file, q, lsp_tx, graph)?;
            res.extend(page);
        }
        Ok(res)
    }
    pub fn collect_endpoints<G: Graph>(
        &self,
        code: &str,
        file: &str,
        graph: Option<&G>,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Vec<(NodeData, Option<Edge>)>> {
        if self.lang().endpoint_finders().is_empty() {
            return Ok(Vec::new());
        }
        let mut res = Vec::new();
        for ef in self.lang().endpoint_finders() {
            let q = self.lang.q(&ef, &NodeType::Endpoint);
            let tree = self.lang.parse(&code, &NodeType::Endpoint)?;
            let mut cursor = QueryCursor::new();
            let mut matches = cursor.matches(&q, tree.root_node(), code.as_bytes());
            while let Some(m) = matches.next() {
                let endys = self.format_endpoint(&m, code, file, &q, graph, lsp_tx)?;
                res.extend(endys);
            }
        }
        Ok(res)
    }
    pub fn collect_functions<G: Graph>(
        &self,
        q: &Query,
        code: &str,
        file: &str,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Vec<Function>> {
        let tree = self.lang.parse(&code, &NodeType::Function)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            if let Some(ff) = self.format_function(&m, code, file, &q, graph, lsp_tx)? {
                res.push(ff);
            }
        }
        Ok(res)
    }
    pub fn collect_tests(&self, q: &Query, code: &str, file: &str) -> Result<Vec<Function>> {
        let tree = self.lang.parse(&code, &NodeType::Test)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            let ff = self.format_test(&m, code, file, &q)?;
            // FIXME trait operand here as well?
            res.push((ff, None, vec![], vec![], None, vec![]));
        }
        Ok(res)
    }
    pub fn collect_calls_in_function<G: Graph>(
        &self,
        q: &Query,
        code: &str,
        file: &str,
        caller_node: TreeNode,
        caller_name: &str,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Vec<FunctionCall>> {
        trace!("collect_calls_in_function");
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, caller_node, code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            if let Some(fc) =
                self.format_function_call(&m, code, file, q, caller_name, graph, lsp_tx)?
            {
                res.push(fc);
            }
        }
        Ok(res)
    }
    pub fn collect_extras_in_function<G: Graph>(
        &self,
        q: &Query,
        code: &str,
        file: &str,
        caller_node: TreeNode,
        caller_name: &str,
        caller_start: usize,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Vec<Edge>> {
        trace!("collect_calls_in_function");
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, caller_node, code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            if let Some(fc) =
                self.format_extra(&m, code, file, q, caller_name, caller_start, graph, lsp_tx)?
            {
                res.push(fc);
            }
        }
        Ok(res)
    }

    pub fn collect_integration_test_calls<G: Graph>(
        &self,
        code: &str,
        file: &str,
        caller_node: TreeNode,
        caller_name: &str,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Vec<Edge>> {
        if self.lang.integration_test_query().is_none() {
            return Ok(Vec::new());
        }
        // manually find instead
        if self.lang.use_integration_test_finder() {
            return Ok(Vec::new());
        }
        let q = self.q(
            &self.lang.integration_test_query().unwrap(),
            &NodeType::Test,
        );
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&q, caller_node, code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            if let Some(fc) =
                self.format_integration_test_call(&m, code, file, &q, caller_name, graph, lsp_tx)?
            {
                res.push(fc);
            }
        }
        Ok(res)
    }
    pub fn collect_integration_tests<G: Graph>(
        &self,
        code: &str,
        file: &str,
        graph: &G,
    ) -> Result<Vec<(NodeData, NodeType, Option<Edge>)>> {
        if self.lang.integration_test_query().is_none() {
            return Ok(Vec::new());
        }
        let q = self.q(
            &self.lang.integration_test_query().unwrap(),
            &NodeType::Test,
        );
        let tree = self.lang.parse(&code, &NodeType::Test)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&q, tree.root_node(), code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            let (nd, tt) = self.format_integration_test(&m, code, file, &q)?;
            let test_edge_opt = self.lang.integration_test_edge_finder(
                &nd,
                &|name| {
                    graph
                        .find_nodes_by_name(NodeType::Class, name)
                        .first()
                        .cloned()
                },
                tt.clone(),
            );
            res.push((nd, tt, test_edge_opt));
        }
        Ok(res)
    }
    pub fn collect_import_edges<G: Graph>(
        &self,
        q: &Query,
        code: &str,
        file: &str,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Vec<Edge>> {
        if let Some(lsp) = lsp_tx {
            return self.collect_import_edges_with_lsp(code, file, graph, lsp);
        }
        let tree = self.lang.parse(&code, &NodeType::Import)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut edges = Vec::new();

        while let Some(m) = matches.next() {
            let mut import_names = Vec::new();
            let mut import_source = None;

            Self::loop_captures_multi(q, &m, code, |body, _node, o| {
                if o == IMPORTS_NAME {
                    import_names.push(body.clone());
                } else if o == IMPORTS_FROM {
                    import_source = Some(trim_quotes(&body).to_string());
                }
                Ok(())
            })?;

            if let Some(source_path) = import_source {
                let resolved_path = self.lang.resolve_import_path(&source_path, file);

                for import_name in &import_names {
                    for nt in [
                        NodeType::Function,
                        NodeType::Class,
                        NodeType::DataModel,
                        NodeType::Var,
                    ] {
                        let name = self.lang.resolve_import_name(import_name);
                        if name.is_empty() {
                            continue;
                        }
                        let targets = graph.find_nodes_by_name(nt.clone(), &name);

                        if !targets.is_empty() {
                            let target = targets
                                .iter()
                                .filter(|node_data| node_data.file.contains(&resolved_path))
                                .next();

                            let file_nodes =
                                graph.find_nodes_by_file_ends_with(NodeType::File, file);
                            let file_node = file_nodes
                                .first()
                                .cloned()
                                .unwrap_or_else(|| NodeData::in_file(file));
                            if let Some(target) = target {
                                edges.push(Edge::file_imports(&file_node, nt, &target));
                                break;
                            }
                        }
                    }
                }
            }
        }
        Ok(edges)
    }
    pub fn collect_import_edges_with_lsp<G: Graph>(
        &self,
        code: &str,
        file: &str,
        graph: &G,
        lsp: &CmdSender,
    ) -> Result<Vec<Edge>> {
        let mut edges = Vec::new();
        let mut processed = std::collections::HashSet::new();

        let query = self.q(&self.lang.identifier_query(), &NodeType::Var);
        let tree = self.lang.parse(code, &NodeType::Function)?;
        let mut cursor = tree_sitter::QueryCursor::new();
        let mut matches = cursor.matches(&query, tree.root_node(), code.as_bytes());

        // To guarantee a deterministic order of identifiers, we collect them and sort them before processing them.
        let mut identifiers = Vec::new();
        while let Some(m) = matches.next() {
            Self::loop_captures(&query, &m, code, |body, node, _o| {
                let p = node.start_position();
                identifiers.push((body.clone(), p.row as u32, p.column as u32));
                Ok(())
            })?;
        }

        identifiers.sort();

        for (target_name, row, col) in identifiers {
            let pos = Position::new(file, row, col)?;
            let res = lsp::Cmd::GotoDefinition(pos.clone()).send(lsp)?;

            if let lsp::Res::GotoDefinition(Some(gt)) = res {
                let target_file = gt.file.display().to_string();

                if self.lang.is_lib_file(&target_file) {
                    continue;
                }

                for nt in [
                    NodeType::Function,
                    NodeType::Class,
                    NodeType::DataModel,
                    NodeType::Var,
                ] {
                    if file == target_file {
                        continue;
                    }
                    let key = format!("{}:{}:{}:{:?}", file, target_name, target_file, nt);
                    if processed.contains(&key) {
                        continue;
                    }
                    let found = graph.find_node_by_name_and_file_end_with(
                        nt.clone(),
                        &target_name,
                        &target_file,
                    );
                    if let Some(ref target) = found {
                        let file_node = graph
                            .find_nodes_by_file_ends_with(NodeType::File, file)
                            .first()
                            .cloned()
                            .unwrap_or_else(|| NodeData::in_file(file));

                        edges.push(Edge::file_imports(&file_node, nt.clone(), &target));
                        processed.insert(key);
                        break;
                    }
                }
            }
        }

        Ok(edges)
    }
    pub fn collect_var_call_in_function<G: Graph>(
        &self,
        func: &NodeData,
        graph: &G,
        lsp_tsx: &Option<CmdSender>,
    ) -> Vec<Edge> {
        if let Some(lsp) = lsp_tsx {
            return self.collect_var_call_in_function_lsp(func, graph, lsp);
        }
        let mut edges = Vec::new();
        if func.body.is_empty() {
            return edges;
        }

        let all_vars = graph.find_nodes_by_type(NodeType::Var);

        let imports = graph.find_nodes_by_file_ends_with(NodeType::Import, &func.file);
        let import_body = imports
            .get(0)
            .map(|imp| imp.body.clone())
            .unwrap_or_default();

        for var in all_vars {
            if var.name.is_empty() {
                continue;
            }

            if func.body.contains(&var.name) {
                if var.file == func.file {
                    edges.push(Edge::contains(
                        NodeType::Function,
                        func,
                        NodeType::Var,
                        &var,
                    ));
                    continue;
                }

                if !import_body.is_empty() && import_body.contains(&var.name) {
                    edges.push(Edge::contains(
                        NodeType::Function,
                        func,
                        NodeType::Var,
                        &var,
                    ));
                }
            }
        }
        edges
    }
    pub fn collect_var_call_in_function_lsp<G: Graph>(
        &self,
        func: &NodeData,
        graph: &G,
        lsp: &CmdSender,
    ) -> Vec<Edge> {
        let mut edges = Vec::new();
        let mut processed = std::collections::HashSet::new();

        if func.body.is_empty() {
            return edges;
        }

        let code = &func.body;
        let tree = match self.lang.parse(code, &NodeType::Function) {
            Ok(tree) => tree,
            Err(_) => return edges,
        };
        let query = self.q(&self.lang.identifier_query(), &NodeType::Var);
        let mut cursor = tree_sitter::QueryCursor::new();
        let mut matches = cursor.matches(&query, tree.root_node(), code.as_bytes());

        let mut identifiers = Vec::new();
        while let Some(m) = matches.next() {
            Self::loop_captures(&query, &m, code, |body, node, _o| {
                let p = node.start_position();
                identifiers.push((body.clone(), p.row as u32, p.column as u32));
                Ok(())
            })
            .ok();
        }
        identifiers.sort();

        for (target_name, row, col) in &identifiers {
            let absolute_line = func.start as u32 + *row;
            let pos = Position::new(&func.file, absolute_line, *col).unwrap();

            let mut lsp_result = None;
            for _ in 0..2 {
                let res = LspCmd::GotoDefinition(pos.clone()).send(lsp);
                if let Ok(LspRes::GotoDefinition(Some(gt))) = res {
                    lsp_result = Some(gt);
                    break;
                }
            }
            let gt = match lsp_result {
                Some(gt) => gt,
                None => continue,
            };
            let target_file = gt.file.display().to_string();

            let key = format!(
                "{}:{}:{}:{}",
                func.file, func.name, target_name, target_file
            );

            if processed.contains(&key) {
                continue;
            }

            let all_vars_with_name = graph.find_nodes_by_name(NodeType::Var, target_name);
            if let Some(var) = all_vars_with_name
                .iter()
                .find(|v| target_file.ends_with(&v.file))
            {
                edges.push(Edge::contains(
                    NodeType::Function,
                    func,
                    NodeType::Var,
                    &var,
                ));
                processed.insert(key);
            }
        }

        edges
    }

    pub fn collect_class_contains_datamodel_edge<G: Graph>(
        &self,
        datamodel: &NodeData,
        graph: &G,
    ) -> Result<Vec<Edge>> {
        let mut edges = Vec::new();
        let classes = self
            .lang
            .class_contains_datamodel(datamodel, &|class_name| {
                graph
                    .find_nodes_by_name(NodeType::Class, class_name)
                    .first()
                    .cloned()
            });
        for class in classes {
            edges.push(Edge::contains(
                NodeType::Class,
                &class,
                NodeType::DataModel,
                datamodel,
            ));
        }
        Ok(edges)
    }
}
