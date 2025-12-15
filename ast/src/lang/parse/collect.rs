use super::utils::trim_quotes;
use crate::lang::{graphs::Graph, *};
use lsp::{Cmd as LspCmd, Position, Res as LspRes};
use shared::error::{Error, Result};
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
                _ => return Err(Error::Custom(format!("collect: {nt:?} not implemented"))),
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
            if let Some((cls, edges)) =
                self.format_class_with_associations(&m, code, file, q, graph)?
            {
                res.push((cls, edges));
            }
        }
        Ok(res)
    }
    pub fn collect_implements(
        &self,
        q: &Query,
        code: &str,
        file: &str,
    ) -> Result<Vec<(String, String, String)>> {
        let tree = self.lang.parse(&code, &NodeType::Class)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut results = Vec::new();

        while let Some(m) = matches.next() {
            let (class_name, trait_name) = self.format_implements(&m, code, q)?;
            results.push((class_name, trait_name, file.to_string()));
        }

        Ok(results)
    }

    pub fn collect_implements_edges<G: Graph>(
        &self,
        q: &Query,
        code: &str,
        graph: &G,
    ) -> Result<Vec<Edge>> {
        let tree = self.lang.parse(&code, &NodeType::Class)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut edges = Vec::new();

        while let Some(m) = matches.next() {
            let (class_name, trait_name) = self.format_implements(&m, code, q)?;
            let class_nodes = graph.find_nodes_by_name(NodeType::Class, &class_name);
            let trait_nodes = graph.find_nodes_by_name(NodeType::Trait, &trait_name);

            if let (Some(class), Some(trait_)) = (class_nodes.first(), trait_nodes.first()) {
                edges.push(Edge::implements(class, trait_));
            }
        }
        Ok(edges)
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
        identified_tests: &std::collections::HashSet<NodeKeys>,
    ) -> Result<Vec<Function>> {
        let tree = self.lang.parse(&code, &NodeType::Function)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            if let Some(ff) = self.format_function(&m, code, file, &q, graph, lsp_tx)? {
                let test_key = NodeKeys::new(&ff.0.name, &ff.0.file, ff.0.start);
                if identified_tests.contains(&test_key) {
                    continue;
                }
                res.push(ff);
            }
        }
        Ok(res)
    }
    pub fn collect_tests<G: Graph>(
        &self,
        q: &Query,
        code: &str,
        file: &str,
        graph: &G,
    ) -> Result<Vec<(Function, Option<Edge>)>> {
        let tree = self.lang.parse(&code, &NodeType::UnitTest)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, tree.root_node(), code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            let ff = self.format_test(&m, code, file, &q)?;

            let test_edge = if let Some(class_nd) =
                graph.find_nodes_by_name(NodeType::Class, &ff.name).first()
            {
                let test_type = self.lang.classify_test(&ff.name, file, &ff.body);
                Some(Edge::calls(test_type, &ff, NodeType::Class, class_nd))
            } else {
                None
            };

            res.push(((ff, None, vec![], vec![], None, vec![], Vec::new()), test_edge));
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
        caller_start: usize,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
        allow_unverified: bool,
    ) -> Result<Vec<FunctionCall>> {
        trace!("collect_calls_in_function");
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(q, caller_node, code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            if let Some(fc) = self.format_function_call(
                &m,
                code,
                file,
                q,
                caller_name,
                caller_start,
                graph,
                lsp_tx,
                allow_unverified,
            )? {
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
            &NodeType::IntegrationTest,
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
        let f = file.replace('\\', "/");
        if !(f.contains("/integration/")
            || f.contains(".int.")
            || f.contains(".integration.")
            || f.contains("integration"))
        {
            // Skip describe promotion unless path OR filename signals integration intent
            return Ok(Vec::new());
        }
        let q = self.q(
            &self.lang.integration_test_query().unwrap(),
            &NodeType::IntegrationTest,
        );
        let tree = self.lang.parse(&code, &NodeType::IntegrationTest)?;
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
    pub fn collect_e2e_tests(&self, code: &str, file: &str) -> Result<Vec<NodeData>> {
        if self.lang.e2e_test_query().is_none() {
            return Ok(Vec::new());
        }
        if !self.lang.is_e2e_test_file(file, code) {
            return Ok(Vec::new());
        }
        let q = self.q(&self.lang.e2e_test_query().unwrap(), &NodeType::E2eTest);
        let tree = self.lang.parse(&code, &NodeType::E2eTest)?;
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&q, tree.root_node(), code.as_bytes());
        let mut res = Vec::new();
        while let Some(m) = matches.next() {
            let (mut nd, tt) = self.format_integration_test(&m, code, file, &q)?;
            if tt != NodeType::E2eTest {
                nd.meta.insert("test_kind".into(), "e2e".into());
            }
            res.push(nd);
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
                    import_names.push(body);
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
                identifiers.push((body, p.row as u32, p.column as u32));
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
                identifiers.push((body, p.row as u32, p.column as u32));
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
    pub fn find_nested_functions<'a>(
        &self,
        functions: &'a [NodeData],
    ) -> Vec<(&'a NodeData, &'a NodeData)> {
        let mut nested = Vec::new();
        for child in functions {
            for parent in functions {
                if std::ptr::eq(child, parent) {
                    continue;
                }
                if child.start > parent.start && child.end < parent.end {
                    nested.push((child, parent));
                }
            }
        }
        nested
    }

    pub fn find_functions_nested_in_variables<'a>(
        &self,
        functions: &'a mut [NodeData],
        variables: &'a [NodeData],
    ) -> Vec<(&'a NodeData, &'a NodeData)> {
        use std::collections::HashMap;
        
        let mut nested = Vec::new();
        
        let mut var_index: HashMap<String, Vec<&NodeData>> = HashMap::new();
        for var in variables {
            var_index.entry(var.file.clone()).or_insert_with(Vec::new).push(var);
        }
        
        for func in functions.iter_mut() {
            if let Some(vars_in_file) = var_index.get(&func.file) {
                for var in vars_in_file {
                    if func.start > var.start && func.end < var.end {
                        func.add_nested_in(&var.name);
                        nested.push((func as &NodeData, var as &NodeData));
                        break;
                    }
                }
            }
        }
        
        nested
    }
}
