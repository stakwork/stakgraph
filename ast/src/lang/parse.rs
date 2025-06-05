use super::{graphs::Graph, *};
use anyhow::Result;
use lsp::{Cmd as LspCmd, Position, Res as LspRes};
use streaming_iterator::StreamingIterator;
use tracing::debug;
use tree_sitter::{Node as TreeNode, QueryMatch};

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
                NodeType::Endpoint | NodeType::Request => self
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
    pub fn format_class_with_associations<G: Graph>(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
        graph: &G,
    ) -> Result<(NodeData, Vec<Edge>)> {
        let mut cls = NodeData::in_file(file);
        let mut associations = Vec::new();
        let mut association_type = None;
        let mut assocition_target = None;

        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == CLASS_NAME {
                cls.name = body;
            } else if o == CLASS_DEFINITION {
                cls.body = body;
                cls.start = node.start_position().row;
                cls.end = node.end_position().row;
            } else if o == CLASS_PARENT {
                cls.add_parent(&body);
            } else if o == INCLUDED_MODULES {
                cls.add_includes(&body);
            } else if o == ASSOCIATION_TYPE {
                association_type = Some(body.clone());
            } else if o == ASSOCIATION_TARGET {
                assocition_target = Some(body.clone());
            }

            if let (Some(ref _ty), Some(ref target)) = (&association_type, &assocition_target) {
                //ty == assocition type like belongs_to, has_many, etc.
                let target_class_name = self.lang.convert_association_to_name(&trim_quotes(target));
                let target_classes = graph.find_nodes_by_name(NodeType::Class, &target_class_name);
                if let Some(target_class) = target_classes.first() {
                    let edge = Edge::calls(NodeType::Class, &cls, NodeType::Class, &target_class);
                    associations.push(edge);
                    association_type = None;
                    assocition_target = None;
                }
            }
            Ok(())
        })?;
        Ok((cls, associations))
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
    pub fn format_library(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
    ) -> Result<NodeData> {
        let mut cls = NodeData::in_file(file);
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == LIBRARY_NAME {
                cls.name = trim_quotes(&body).to_string();
            } else if o == LIBRARY {
                cls.body = body;
                cls.start = node.start_position().row;
                cls.end = node.end_position().row;
            } else if o == LIBRARY_VERSION {
                cls.add_version(&trim_quotes(&body).to_string());
            }
            Ok(())
        })?;
        Ok(cls)
    }
    pub fn format_imports(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
    ) -> Result<Vec<NodeData>> {
        let mut res = Vec::new();
        Self::loop_captures_multi(q, &m, code, |body, node, o| {
            let mut impy = NodeData::in_file(file);
            if o == IMPORTS {
                impy.name = "imports".to_string();
                impy.body = body;
                impy.start = node.start_position().row;
                impy.end = node.end_position().row;
                res.push(impy);
            }

            Ok(())
        })?;
        Ok(res)
    }
    pub fn format_variables(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
    ) -> Result<Vec<NodeData>> {
        let mut res = Vec::new();
        let mut v = NodeData::in_file(file);
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == VARIABLE_NAME {
                v.name = body.to_string();
            } else if o == VARIABLE_DECLARATION {
                v.body = body;
                v.start = node.start_position().row;
                v.end = node.end_position().row;
            } else if o == VARIABLE_TYPE {
                v.data_type = Some(body);
            }

            Ok(())
        })?;
        if !v.name.is_empty() && !v.body.is_empty() {
            res.push(v);
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
    pub fn format_page<G: Graph>(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
        lsp: &Option<CmdSender>,
        graph: &G,
    ) -> Result<Vec<(NodeData, Vec<Edge>)>> {
        let mut pag = NodeData::in_file(file);
        let mut components_positions_names = Vec::new();
        let mut page_renders = Vec::new();
        let mut page_names = Vec::new();
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == PAGE_PATHS {
                // page_names.push(trim_quotes(&body).to_string());
                page_names = self
                    .find_strings(node, code, file)?
                    .iter()
                    .map(|s| trim_quotes(&s).to_string())
                    .collect();
            } else if o == PAGE {
                pag.body = body;
                pag.start = node.start_position().row;
                pag.end = node.end_position().row;
            } else if o == PAGE_COMPONENT {
                let p = node.start_position();
                let pos = Position::new(file, p.row as u32, p.column as u32)?;
                components_positions_names.push((pos, body));
            } else if o == PAGE_CHILD {
                let p = node.start_position();
                let pos = Position::new(file, p.row as u32, p.column as u32)?;
                components_positions_names.push((pos, body));
            } else if o == PAGE_HEADER {
                let p = node.start_position();
                let pos = Position::new(file, p.row as u32, p.column as u32)?;
                components_positions_names.push((pos, body));
            }
            Ok(())
        })?;
        for (pos, comp_name) in components_positions_names {
            if let Some(lsp) = lsp {
                // use lsp to find the component
                log_cmd(format!("=> looking for component {:?}", comp_name));
                let res = LspCmd::GotoDefinition(pos.clone()).send(&lsp)?;
                if let LspRes::GotoDefinition(Some(gt)) = res {
                    let target_file = gt.file.display().to_string();
                    if let Some(_) = graph.find_node_by_name_in_file(
                        NodeType::Function,
                        &comp_name,
                        &target_file,
                    ) {
                        let target = NodeData::name_file(&comp_name, &target_file);
                        page_renders.push(Edge::renders(&pag, &target));
                        continue;
                    }
                }
            }
            // fallback
            let nodes = graph.find_nodes_by_name(NodeType::Function, &comp_name);
            // only take the first? FIXME
            if let Some(node) = nodes.first() {
                page_renders.push(Edge::renders(&pag, &node));
            }
        }
        if page_names.is_empty() {
            return Ok(Vec::new());
        }
        let mut pages = Vec::new();
        // push one for each page name
        for pn in page_names {
            let mut p = pag.clone();
            p.name = pn.clone();
            let mut pr = page_renders.clone();
            for er in pr.iter_mut() {
                er.source.node_data.name = pn.clone();
            }
            pages.push((p, pr));
        }
        Ok(pages)
    }
    // find any "string" within the node
    fn find_strings(&self, node: TreeNode, code: &str, file: &str) -> Result<Vec<String>> {
        let mut results = Vec::new();
        if node.kind() == self.lang.string_node_name() {
            let sname = node.utf8_text(code.as_bytes())?;
            results.push(sname.to_string());
        }
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                results.extend(self.find_strings(child, code, file)?);
            }
        }
        Ok(results)
    }
    pub fn format_trait(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
    ) -> Result<NodeData> {
        let mut tr = NodeData::in_file(file);
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == TRAIT_NAME {
                tr.name = body;
            } else if o == TRAIT {
                tr.body = body;
                tr.start = node.start_position().row;
                tr.end = node.end_position().row;
            }
            Ok(())
        })?;
        Ok(tr)
    }
    pub fn format_instance(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
    ) -> Result<NodeData> {
        let mut inst = NodeData::in_file(file);
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == INSTANCE_NAME {
                inst.name = body;
                inst.start = node.start_position().row;
                inst.end = node.end_position().row;
            } else if o == CLASS_NAME {
                inst.data_type = Some(body);
            } else if o == INSTANCE {
                inst.body = body;
            }
            Ok(())
        })?;
        Ok(inst)
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
    // endpoint, handlers
    pub fn format_endpoint<G: Graph>(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
        graph: Option<&G>,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Vec<(NodeData, Option<Edge>)>> {
        // println!("FORMAT ENDPOINT");
        let mut endp = NodeData::in_file(file);
        let mut handler = None;
        let mut call = None;
        let mut params = HandlerParams::default();
        let mut handler_position = None;
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == ENDPOINT {
                let namey = trim_quotes(&body);
                if namey.len() > 0 {
                    endp.name = namey.to_string();
                }
                // println!("endpoint {:?}", inst.name);
            } else if o == ENDPOINT_ALIAS {
                // endpoint alias overwrites
                let namey = trim_quotes(&body);
                if namey.len() > 0 {
                    endp.name = namey.to_string();
                }
                // println!("alias {:?}", inst.name);
            } else if o == ROUTE {
                endp.body = body;
                endp.start = node.start_position().row;
                endp.end = node.end_position().row;
            } else if o == HANDLER {
                // tracing::info!("found HANDLER {:?} {:?}", body, endp.name);
                let handler_name = trim_quotes(&body);
                endp.add_handler(&handler_name);
                let p = node.start_position();
                handler_position = Some(Position::new(file, p.row as u32, p.column as u32)?);
                if let Some(graph) = graph {
                    // collect parents
                    params.parents =
                        self.lang.find_endpoint_parents(node, code, file, &|name| {
                            graph
                                .find_nodes_by_name(NodeType::Function, name)
                                .first()
                                .cloned()
                        })?;
                }
            } else if o == HANDLER_ACTIONS_ARRAY {
                // [:destroy, :index]
                params.actions_array = Some(body);
            } else if o == ENDPOINT_VERB {
                endp.add_verb(&body.to_uppercase());
            } else if o == REQUEST_CALL {
                call = Some(body);
            } else if o == ENDPOINT_GROUP {
                endp.add_group(&body);
            } else if o == COLLECTION_ITEM {
                params.item = Some(HandlerItem::new_collection(trim_quotes(&body)));
            } else if o == MEMBER_ITEM {
                params.item = Some(HandlerItem::new_member(trim_quotes(&body)));
            } else if o == RESOURCE_ITEM {
                params.item = Some(HandlerItem::new_resource_member(trim_quotes(&body)));
            }
            Ok(())
        })?;
        if endp.meta.get("verb").is_none() {
            self.lang.add_endpoint_verb(&mut endp, &call);
        }
        self.lang.update_endpoint_verb(&mut endp, &call);
        // for multi-handle endpoints with no "name:" (ENDPOINT)
        if endp.name.is_empty() {
            if let Some(handler) = endp.meta.get("handler") {
                endp.name = handler.to_string();
            }
        }
        if let Some(graph) = graph {
            if self.lang().use_handler_finder() {
                // find handler manually (not LSP)
                return Ok(self.lang().handler_finder(
                    endp,
                    &|handler, suffix| {
                        graph.find_node_by_name_and_file_end_with(
                            NodeType::Function,
                            handler,
                            suffix,
                        )
                    },
                    &|file| graph.find_nodes_by_file_ends_with(NodeType::Function, file),
                    params,
                ));
            } else {
                // here find the handler using LSP!
                if let Some(handler_name) = endp.meta.get("handler") {
                    if let Some(lsp) = lsp_tx {
                        if let Some(pos) = handler_position {
                            log_cmd(format!("=> looking for HANDLER {:?}", handler_name));
                            let res = LspCmd::GotoDefinition(pos.clone()).send(&lsp)?;
                            if let LspRes::GotoDefinition(Some(gt)) = res {
                                let target_file = gt.file.display().to_string();
                                if let Some(_t_file) = graph.find_node_by_name_in_file(
                                    NodeType::Function,
                                    &handler_name,
                                    &target_file,
                                ) {
                                    log_cmd(format!("HANDLER def, in graph: {:?}", handler_name));
                                } else {
                                    log_cmd(format!("HANDLER def, not found: {:?}", handler_name));
                                }
                                let target = NodeData::name_file(&handler_name, &target_file);
                                handler = Some(Edge::handler(&endp, &target));
                            }
                        }
                    } else {
                        // FALLBACK to find?
                        return Ok(self.lang().handler_finder(
                            endp,
                            &|handler, suffix| {
                                graph.find_node_by_name_and_file_end_with(
                                    NodeType::Function,
                                    handler,
                                    suffix,
                                )
                            },
                            &|file| graph.find_nodes_by_file_ends_with(NodeType::Function, file),
                            params,
                        ));
                    }
                }
            }
        }
        // println!("<<< endpoint >>> {:?}", endp.name);
        Ok(vec![(endp, handler)])
    }
    pub fn format_data_model(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
    ) -> Result<NodeData> {
        let mut inst = NodeData::in_file(file);
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == STRUCT_NAME {
                inst.name = trim_quotes(&body).to_string();
            } else if o == STRUCT {
                inst.body = body;
                inst.start = node.start_position().row;
                inst.end = node.end_position().row;
            }
            Ok(())
        })?;
        Ok(inst)
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
    fn format_function<G: Graph>(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Option<Function>> {
        let mut func = NodeData::in_file(file);
        let mut parent = None;
        let mut parent_type = None;
        let mut requests_within = Vec::new();
        let mut models: Vec<Edge> = Vec::new();
        let mut trait_operand = None;
        let mut name_pos = None;
        let mut return_types = Vec::new();
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == PARENT_TYPE {
                parent_type = Some(body);
            } else if o == FUNCTION_NAME {
                func.name = body;
                let p = node.start_position();
                let pos = Position::new(file, p.row as u32, p.column as u32)?;
                name_pos = Some(pos);
            } else if o == FUNCTION_DEFINITION {
                func.body = body;
                func.start = node.start_position().row;
                func.end = node.end_position().row;
                // parent
                parent = self.lang.find_function_parent(
                    node,
                    code,
                    file,
                    &func.name,
                    &|name| {
                        graph
                            .find_nodes_by_name(NodeType::Class, name)
                            .first()
                            .cloned()
                    },
                    parent_type.as_deref(),
                )?;
                if let Some(pp) = &parent {
                    func.add_operand(&pp.source.name);
                }
                // requests to endpoints
                if let Some(rq) = self.lang.request_finder() {
                    let mut cursor = QueryCursor::new();
                    let qqq = self.q(&rq, &NodeType::Request);
                    let mut matches = cursor.matches(&qqq, node, code.as_bytes());
                    while let Some(m) = matches.next() {
                        let reqs = self.format_endpoint::<G>(
                            &m,
                            code,
                            file,
                            &self.q(&rq, &NodeType::Endpoint),
                            None,
                            &None,
                        )?;
                        if !reqs.is_empty() {
                            requests_within.push(reqs[0].clone().0);
                        }
                    }
                }
                // data models
                if self.lang.use_data_model_within_finder() {
                    // do this later actually
                    // models = self.lang.data_model_within_finder(
                    //     &func,
                    //     graph,
                    // );
                } else if let Some(dmq) = self.lang.data_model_within_query() {
                    let mut cursor = QueryCursor::new();
                    let qqq = self.q(&dmq, &NodeType::DataModel);
                    let mut matches = cursor.matches(&qqq, node, code.as_bytes());
                    while let Some(m) = matches.next() {
                        let dm_node = self.format_data_model(&m, code, file, &qqq)?;
                        if models
                            .iter()
                            .any(|e| e.target.node_data.name == dm_node.name)
                        {
                            continue;
                        }
                        match graph
                            .find_nodes_by_name(NodeType::DataModel, &dm_node.name)
                            .first()
                            .cloned()
                        {
                            Some(dmr) => {
                                models.push(Edge::contains(
                                    NodeType::Function,
                                    &func,
                                    NodeType::DataModel,
                                    &dmr,
                                ));
                            }
                            None => (),
                        }
                    }
                }
            } else if o == ARGUMENTS {
                // skipping args
            } else if o == RETURN_TYPES {
                if let Some(lsp) = lsp_tx {
                    for (name, pos) in self.find_type_identifiers(node, code, file)? {
                        if is_capitalized(&name) {
                            let res = LspCmd::GotoDefinition(pos.clone()).send(&lsp)?;
                            if let LspRes::GotoDefinition(Some(gt)) = res {
                                let dfile = gt.file.display().to_string();
                                if !self.lang.is_lib_file(&dfile) {
                                    if let Some(t) =
                                        graph.find_node_at(NodeType::DataModel, &dfile, gt.line)
                                    {
                                        log_cmd(format!(
                                            "*******RETURN_TYPE found target for {:?} {} {}!!!",
                                            name, &t.file, &t.name
                                        ));
                                        return_types.push(Edge::contains(
                                            NodeType::Function,
                                            &func,
                                            NodeType::DataModel,
                                            &t,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        })?;
        if func.body.is_empty() {
            log_cmd(format!("found function but empty body {:?}", func.name));
            return Ok(None);
        }
        if let Some(pos) = name_pos {
            trait_operand = self.lang.find_trait_operand(
                pos,
                &func,
                &|row, file| graph.find_nodes_in_range(NodeType::Trait, row, file),
                lsp_tx,
            )?;
        }
        log_cmd(format!("found function {:?}", func.name));
        Ok(Some((
            func,
            parent,
            requests_within,
            models,
            trait_operand,
            return_types,
        )))
    }
    fn find_type_identifiers(
        &self,
        node: TreeNode,
        code: &str,
        file: &str,
    ) -> Result<Vec<(String, Position)>> {
        let mut results = Vec::new();
        // Check if current node matches the type identifier name
        if node.kind() == self.lang.type_identifier_node_name() {
            let type_name = node.utf8_text(code.as_bytes())?;
            let pos = node.start_position();
            let position = Position::new(file, pos.row as u32, pos.column as u32)?;
            results.push((type_name.to_string(), position));
        }
        // Recursively check all named children
        for i in 0..node.named_child_count() {
            if let Some(child) = node.named_child(i) {
                results.extend(self.find_type_identifiers(child, code, file)?);
            }
        }
        Ok(results)
    }
    fn format_test(&self, m: &QueryMatch, code: &str, file: &str, q: &Query) -> Result<NodeData> {
        let mut test = NodeData::in_file(file);
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == FUNCTION_NAME {
                test.name = trim_quotes(&body).to_string();
            } else if o == FUNCTION_DEFINITION {
                test.body = body;
                test.start = node.start_position().row;
                test.end = node.end_position().row;
            }
            Ok(())
        })?;
        Ok(test)
    }
    pub fn loop_captures<'a, F>(
        q: &Query,
        m: &QueryMatch<'a, 'a>,
        code: &str,
        mut cb: F,
    ) -> Result<()>
    where
        F: FnMut(String, TreeNode, String) -> Result<()>,
    {
        for o in q.capture_names().iter() {
            if let Some(ci) = q.capture_index_for_name(&o) {
                let mut nodes = m.nodes_for_capture_index(ci);
                if let Some(node) = nodes.next() {
                    let body = node.utf8_text(code.as_bytes())?.to_string();
                    if let Err(e) = cb(body, node, o.to_string()) {
                        println!("error in loop_captures {:?}", e);
                    }
                }
            }
        }
        Ok(())
    }
    pub fn loop_captures_multi<'a, F>(
        q: &Query,
        m: &QueryMatch<'a, 'a>,
        code: &str,
        mut cb: F,
    ) -> Result<()>
    where
        F: FnMut(String, TreeNode, String) -> Result<()>,
    {
        for o in q.capture_names().iter() {
            if let Some(ci) = q.capture_index_for_name(&o) {
                let nodes = m.nodes_for_capture_index(ci);
                for node in nodes {
                    let body = node.utf8_text(code.as_bytes())?.to_string();
                    if let Err(e) = cb(body, node, o.to_string()) {
                        println!("error in loop_captures {:?}", e);
                    }
                }
            }
        }
        Ok(())
    }
    pub fn collect_calls_in_function<'a, G: Graph>(
        &self,
        q: &Query,
        code: &str,
        file: &str,
        caller_node: TreeNode<'a>,
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
    fn format_function_call<'a, 'b, G: Graph>(
        &self,
        m: &QueryMatch<'a, 'b>,
        code: &str,
        file: &str,
        q: &Query,
        caller_name: &str,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Option<FunctionCall>> {
        let mut fc = Calls::default();
        let mut external_func = None;
        let mut class_call = None;
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == FUNCTION_NAME {
                let called = body;
                trace!("format_function_call {} {}", caller_name, called);
                if let Some(lsp) = lsp_tx {
                    let p = node.start_position();
                    log_cmd(format!("=> {} looking for {:?}", caller_name, called));
                    let pos = Position::new(file, p.row as u32, p.column as u32)?;
                    let res = LspCmd::GotoDefinition(pos.clone()).send(&lsp)?;
                    if let LspRes::GotoDefinition(None) = res {
                        log_cmd(format!("==> _ no definition found for {:?}", called));
                    }
                    if let LspRes::GotoDefinition(Some(gt)) = res {
                        let target_file = gt.file.display().to_string();
                        if let Some(t) = graph.find_node_by_name_in_file(
                            NodeType::Function,
                            &called,
                            &target_file,
                        ) {
                            log_cmd(format!(
                                "==> ! found target for {:?} {}!!!",
                                called, &t.file
                            ));
                            fc.target = NodeKeys::new(&called, &t.file, t.start);
                            // set extenal func so this is marked as USES edge rather than CALLS
                            if t.body.is_empty() && t.docs.is_some() {
                                log_cmd(format!("==> ! found target is external {:?}!!!", called));
                                external_func = Some(t);
                            }
                        } else {
                            if let Some(one_func) = func_target_file_finder(&called, &None, graph) {
                                log_cmd(format!("==> ? ONE target for {:?} {}", called, &one_func));
                                fc.target = NodeKeys::new(&called, &one_func, 0);
                            } else {
                                // println!("no target for {:?}", body);
                                log_cmd(format!(
                                    "==> ? definition, not in graph: {:?} in {}",
                                    called, &target_file
                                ));
                                if self.lang.is_lib_file(&target_file) {
                                    if !self.lang.is_component(&called) {
                                        let mut lib_func =
                                            NodeData::name_file(&called, &target_file);
                                        lib_func.start = gt.line as usize;
                                        lib_func.end = gt.line as usize;
                                        let pos2 =
                                            Position::new(&file, p.row as u32, p.column as u32)?;
                                        let hover_res = LspCmd::Hover(pos2).send(&lsp)?;
                                        if let LspRes::Hover(Some(hr)) = hover_res {
                                            lib_func.docs = Some(hr);
                                        }
                                        external_func = Some(lib_func);
                                        fc.target =
                                            NodeKeys::new(&called, &target_file, gt.line as usize);
                                    }
                                } else {
                                    // handle trait match, jump to implemenetations
                                    let res = LspCmd::GotoImplementations(pos).send(&lsp)?;
                                    if let LspRes::GotoImplementations(Some(gt2)) = res {
                                        log_cmd(format!("==> ? impls {} {:?}", called, gt2));
                                        let target_file = gt2.file.display().to_string();
                                        if let Some(t_file) = graph.find_node_by_name_in_file(
                                            NodeType::Function,
                                            &called,
                                            &target_file,
                                        ) {
                                            log_cmd(format!(
                                                "==> ! found target for impl {:?} {:?}!!!",
                                                called, &t_file
                                            ));
                                            fc.target =
                                                NodeKeys::new(&called, &t_file.file, t_file.start);
                                        }
                                    }
                                }
                                // NOTE: commented out. only add the func if its either a lib component, or in the graph already
                                // fc.target = NodeKeys::new(&called, &target_file);
                            }
                        }
                    }
                // } else if let Some(tf) = func_target_file_finder(&body, &fc.operand, graph) {
                // fc.target = NodeKeys::new(&body, &tf);
                } else {
                    // println!("no target for {:?}", body);
                    // FALLBACK to find?
                    if let Some(tf) = func_target_file_finder(&called, &None, graph) {
                        log_cmd(format!(
                            "==> ? (no lsp) ONE target for {:?} {}",
                            called, &tf
                        ));
                        fc.target = NodeKeys::new(&called, &tf, 0);
                    }
                }
            } else if o == FUNCTION_CALL {
                fc.source = NodeKeys::new(&caller_name, file, node.start_position().row);
                fc.call_start = node.start_position().row;
                fc.call_end = node.end_position().row;
            } else if o == OPERAND {
                fc.operand = Some(body.clone());
                if self.lang.direct_class_calls() {
                    let possible_classes = graph.find_nodes_by_name(NodeType::Class, &body);
                    if possible_classes.len() == 1 {
                        class_call = Some(possible_classes[0].clone())
                    }
                }
            }
            Ok(())
        })?;
        // target must be found OR class call
        if fc.target.is_empty() && class_call.is_none() {
            // NOTE should we only do the class call if there is no direct function target?
            return Ok(None);
        }
        Ok(Some((fc, external_func, class_call)))
    }
    pub fn collect_integration_test_calls<'a, G: Graph>(
        &self,
        code: &str,
        file: &str,
        caller_node: TreeNode<'a>,
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
    fn format_integration_test(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
    ) -> Result<(NodeData, NodeType)> {
        trace!("format_integration_test");
        let mut nd = NodeData::in_file(file);
        let mut e2e_test_name = None;
        let mut tt = NodeType::Test;
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == HANDLER {
                nd.name = trim_quotes(&body).to_string();
            }
            if o == INTEGRATION_TEST {
                nd.body = body.clone();
                nd.start = node.start_position().row;
                nd.end = node.end_position().row;
            }
            if o == E2E_TEST_NAME {
                e2e_test_name = Some(trim_quotes(&body).to_string());
            }
            Ok(())
        })?;
        if let Some(e2e_test_name) = e2e_test_name {
            nd.name = e2e_test_name;
            tt = NodeType::E2eTest;
            debug!("E2E_TEST_NAME {:?}", nd.name);
        }
        Ok((nd, tt))
    }
    fn format_integration_test_call<'a, 'b, G: Graph>(
        &self,
        m: &QueryMatch<'a, 'b>,
        code: &str,
        file: &str,
        q: &Query,
        caller_name: &str,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Option<Edge>> {
        trace!("format_integration_test");
        let mut fc = Calls::default();
        let mut handler_name = None;
        let mut call_position = None;
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == HANDLER {
                // println!("====> TEST HANDLER {}", body);
                // GetWorkspaceRepoByWorkspaceUuidAndRepoUuid
                fc.call_start = node.start_position().row;
                fc.call_end = node.end_position().row;
                let p = node.start_position();
                let pos = Position::new(file, p.row as u32, p.column as u32)?;
                handler_name = Some(body);
                call_position = Some(pos);
            }
            Ok(())
        })?;

        if handler_name.is_none() {
            return Ok(None);
        }
        let handler_name = handler_name.unwrap();
        if call_position.is_none() {
            return Ok(None);
        }
        let pos = call_position.unwrap();

        if lsp_tx.is_none() {
            return Ok(None);
        }
        let lsp_tx = lsp_tx.as_ref().unwrap();
        log_cmd(format!(
            "=> {} looking for integration test: {:?}",
            caller_name, handler_name
        ));
        let res = LspCmd::GotoDefinition(pos).send(&lsp_tx)?;
        if let LspRes::GotoDefinition(Some(gt)) = res {
            let target_file = gt.file.display().to_string();
            if let Some(t_file) =
                graph.find_node_by_name_in_file(NodeType::Function, &handler_name, &target_file)
            {
                log_cmd(format!(
                    "==> {} ! found integration test target for {:?} {:?}!!!",
                    caller_name, handler_name, &t_file
                ));
            } else {
                log_cmd(format!(
                    "==> {} ? integration test definition, not in graph: {:?} in {}",
                    caller_name, handler_name, &target_file
                ));
            }
            fc.target = NodeKeys::new(&handler_name, &target_file, gt.line as usize);
        }

        // target must be found
        if fc.target.is_empty() {
            return Ok(None);
        }
        let endpoint = graph.find_source_edge_by_name_and_file(
            EdgeType::Handler,
            &fc.target.name,
            &fc.target.file,
        );

        if endpoint.is_none() {
            return Ok(None);
        }
        let endpoint = endpoint.unwrap();
        let source = NodeKeys::new(&caller_name, file, 0);
        let edge = Edge::new(
            EdgeType::Calls,
            NodeRef::from(source, NodeType::Test),
            NodeRef::from(endpoint, NodeType::Endpoint),
        );
        Ok(Some(edge))
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
            let pos = Position::new(&func.file, *row, *col).unwrap();

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

            if let Some(var) =
                graph.find_node_by_name_and_file_end_with(NodeType::Var, target_name, &target_file)
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
}

fn _func_target_files_finder<G: Graph>(
    func_name: &str,
    operand: &Option<String>,
    graph: &G,
) -> Option<String> {
    log_cmd(format!("func_target_file_finder {:?}", func_name));
    let mut tf = None;
    if let Some(tf_) = find_only_one_function_file(func_name, graph) {
        tf = Some(tf_);
    } else if let Some(op) = operand {
        if let Some(tf_) = find_function_with_operand(&op, func_name, graph) {
            tf = Some(tf_);
        }
    }
    tf
}

fn func_target_file_finder<G: Graph>(
    func_name: &str,
    operand: &Option<String>,
    graph: &G,
) -> Option<String> {
    log_cmd(format!("func_target_file_finder {:?}", func_name));
    let mut tf = None;
    if let Some(tf_) = find_only_one_function_file(func_name, graph) {
        tf = Some(tf_);
    } else if let Some(op) = operand {
        if let Some(tf_) = find_function_with_operand(&op, func_name, graph) {
            tf = Some(tf_);
        }
    }
    tf
}

// FIXME: prefer funcitons in the same file?? Instead of skipping if there are 2
fn find_only_one_function_file<G: Graph>(func_name: &str, graph: &G) -> Option<String> {
    let mut target_files = Vec::new();
    let nodes = graph.find_nodes_by_name(NodeType::Function, func_name);
    for node in nodes {
        // NOT empty functions (interfaces)
        if !node.body.is_empty() {
            target_files.push(node.file.clone());
        }
    }
    if target_files.len() == 1 {
        return Some(target_files[0].clone());
    }
    // TODO: disclue "mock"
    log_cmd(format!("::: found more than one {:?}", func_name));
    target_files.retain(|x| !x.contains("mock"));
    if target_files.len() == 1 {
        log_cmd(format!("::: discluded mocks for!!! {:?}", func_name));
        return Some(target_files[0].clone());
    }
    None
}

fn _find_function_files<G: Graph>(func_name: &str, graph: &G) -> Vec<String> {
    let mut target_files = Vec::new();
    let function_nodes = graph.find_nodes_by_name(NodeType::Function, func_name);
    for node in function_nodes {
        if !node.body.is_empty() {
            target_files.push(node.file.clone());
        }
    }
    target_files
}

fn find_function_with_operand<G: Graph>(
    operand: &str,
    func_name: &str,
    graph: &G,
) -> Option<String> {
    let mut target_file = None;
    let mut instance = None;

    let operand_nodes = graph.find_nodes_by_name(NodeType::Instance, operand);
    for node in operand_nodes {
        instance = Some(node.clone());
        break;
    }
    if let Some(i) = instance {
        if let Some(dt) = &i.data_type {
            let function_nodes = graph.find_nodes_by_name(NodeType::Function, func_name);
            for node in function_nodes {
                if node.meta.get("operand") == Some(dt) {
                    target_file = Some(node.file.clone());
                    break;
                }
            }
        }
    }
    target_file
}

fn _pick_target_file_from_graph<G: Graph>(target_name: &str, graph: &G) -> Option<String> {
    let mut target_file = None;
    let function_nodes = graph.find_nodes_by_name(NodeType::Function, target_name);
    for node in function_nodes {
        target_file = Some(node.file.clone());
        break;
    }

    target_file
}

pub fn trim_quotes(value: &str) -> &str {
    let value = value.trim();
    if value.starts_with('"') && value.ends_with('"') {
        return &value[1..value.len() - 1];
    }
    if value.starts_with("'") && value.ends_with("'") {
        return &value[1..value.len() - 1];
    }
    if value.starts_with("`") && value.ends_with("`") {
        return &value[1..value.len() - 1];
    }
    if value.starts_with(":") {
        return &value[1..];
    }
    value
}

fn log_cmd(cmd: String) {
    debug!("{}", cmd);
}

fn is_capitalized(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    name.chars().next().unwrap().is_uppercase()
}
