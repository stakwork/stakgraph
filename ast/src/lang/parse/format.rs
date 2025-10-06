use std::collections::HashSet;

use crate::lang::call_finder::node_data_finder;
use crate::lang::{graphs::Graph, *};
use lsp::{Cmd as LspCmd, Position, Res as LspRes};
use shared::Result;
use streaming_iterator::StreamingIterator;
use tree_sitter::QueryMatch;

use super::super::queries::consts::FUNCTION_COMMENT;
use super::utils::{clean_class_name, find_def, is_capitalized, log_cmd, trim_quotes};

impl Lang {
    pub fn format_class_with_associations<G: Graph>(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
        graph: &G,
    ) -> Result<Option<(NodeData, Vec<Edge>)>> {
        let mut cls = NodeData::in_file(file);
        let mut associations = Vec::new();
        let mut association_type = None;
        let mut assocition_target = None;
        let mut has_impl = false;

        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == CLASS_NAME {
                cls.name = clean_class_name(&body);
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

        if self.lang.filter_by_implements() {
            if let Some(implements_query) = self.lang.implements_query() {
                let implements_q = self.lang.q(&implements_query, &NodeType::Class);
                let tree = self.lang.parse(code, &NodeType::Class)?;
                let mut cursoe = QueryCursor::new();
                let mut matches = cursoe.matches(&implements_q, tree.root_node(), code.as_bytes());
                while let Some(m) = matches.next() {
                    let (class_name, trait_name) =
                        self.format_implements(&m, code, &implements_q)?;
                    if class_name == cls.name {
                        cls.add_implements(trait_name.as_str());
                        has_impl = true;
                        break;
                    }
                }
            }
            if !has_impl {
                return Ok(None);
            }
        }
        Ok(Some((cls, associations)))
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
                    if let Some(target) = graph.find_node_by_name_in_file(
                        NodeType::Function,
                        &comp_name,
                        &target_file,
                    ) {
                        page_renders.push(Edge::renders(&pag, &target));
                        continue;
                    }
                }
            }
            // fallback
            let nodes = graph.find_nodes_by_name(NodeType::Function, &comp_name);
            // only take the first? FIXME
            let frontend_nodes = nodes
                .iter()
                .filter(|n| graph.is_frontend(&n.file))
                .collect::<Vec<_>>();
            if let Some(node) = frontend_nodes.first() {
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

    pub fn format_implements(
        &self,
        m: &QueryMatch,
        code: &str,
        q: &Query,
    ) -> Result<(String, String)> {
        let mut class_name = String::new();
        let mut trait_name = String::new();
        Self::loop_captures(q, &m, code, |body, _node, o| {
            if o == CLASS_NAME {
                class_name = body;
            } else if o == TRAIT_NAME {
                trait_name = body;
            }
            Ok(())
        })?;
        Ok((class_name, trait_name))
    }

    pub fn format_endpoint<G: Graph>(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
        graph: Option<&G>,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Vec<(NodeData, Option<Edge>)>> {
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
            } else if o == ENDPOINT_ALIAS {
                // endpoint alias overwrites
                let namey = trim_quotes(&body);
                if namey.len() > 0 {
                    endp.name = namey.to_string();
                }
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
        self.lang.update_endpoint(&mut endp, &call);
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
                                if let Some(target) = graph.find_node_by_name_in_file(
                                    NodeType::Function,
                                    &handler_name,
                                    &target_file,
                                ) {
                                    log_cmd(format!("HANDLER def, in graph: {:?}", handler_name));
                                    handler = Some(Edge::handler(&endp, &target));
                                } else {
                                    log_cmd(format!("HANDLER def, not found: {:?}", handler_name));
                                }
                            }
                        }
                    } else {
                        // FALLBACK to find?
                        return Ok(self.lang().handler_finder(
                            endp.clone(),
                            &|handler_name, _suffix| match node_data_finder(
                                handler_name,
                                graph,
                                file,
                                endp.clone().start,
                                NodeType::Endpoint,
                            ) {
                                Some(node_key) => Some(node_key.into()),
                                None => None,
                            },
                            &|file| graph.find_nodes_by_file_ends_with(NodeType::Function, file),
                            params,
                        ));
                    }
                }
            }
        }
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
    pub fn format_function<G: Graph>(
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
        let mut return_type_data_models = Vec::new();
        let mut comments = Vec::new();
        let mut raw_args: Option<String> = None;
        let mut raw_return: Option<String> = None;
        let mut def_start_byte: Option<usize> = None;
        let mut args_end_byte: Option<usize> = None;
        let mut return_end_byte: Option<usize> = None;

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
                def_start_byte = Some(node.start_byte());
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
                            let request_node = reqs[0].clone().0;
                            requests_within.push(request_node);
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
                // find variables in functions
                let vuq = self.lang.identifier_query();
                if !vuq.is_empty() {
                    let mut cursor = QueryCursor::new();
                    let qqq = self.q(&vuq, &NodeType::Var);
                    let mut matches = cursor.matches(&qqq, node, code.as_bytes());
                    let imports = graph.find_nodes_by_file_ends_with(NodeType::Import, file);
                    let import_body = imports.get(0).map(|i| i.body.clone()).unwrap_or_default();
                    let mut found_vars = HashSet::new();

                    while let Some(m) = matches.next() {
                        Self::loop_captures(&qqq, &m, code, |body, _node, o| {
                            if o == VARIABLE_NAME || o == "identifier" {
                                let candidate_vars = graph.find_nodes_by_name(NodeType::Var, &body);
                                for var in candidate_vars {
                                    found_vars.insert(var);
                                }
                            }
                            Ok(())
                        })?;
                    }

                    for var in found_vars {
                        if var.file == *file {
                            models.push(Edge::contains(
                                NodeType::Function,
                                &func,
                                NodeType::Var,
                                &var,
                            ));
                            continue;
                        }

                        if !import_body.is_empty() && import_body.contains(&var.name) {
                            models.push(Edge::contains(
                                NodeType::Function,
                                &func,
                                NodeType::Var,
                                &var,
                            ));
                            continue;
                        }

                        let func_dir = std::path::Path::new(file).parent();
                        let var_dir = std::path::Path::new(&var.file).parent();
                        if let (Some(fdir), Some(vdir)) = (func_dir, var_dir) {
                            if fdir == vdir {
                                models.push(Edge::contains(
                                    NodeType::Function,
                                    &func,
                                    NodeType::Var,
                                    &var,
                                ));
                            }
                        }
                    }
                }
            } else if o == ARGUMENTS {
                raw_args = Some(body.clone());
                args_end_byte = Some(node.end_byte());
            } else if o == RETURN_TYPES {
                raw_return = Some(body.clone());
                return_end_byte = Some(node.end_byte());
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
                                        return_type_data_models.push(t);
                                    }
                                }
                            }
                        }
                    }
                }
            } else if o == FUNCTION_COMMENT {
                comments.push(body);
            }
            Ok(())
        })?;
        if func.body.is_empty() {
            log_cmd(format!("found function but empty body {:?}", func.name));
            return Ok(None);
        }

        if !comments.is_empty() {
            func.docs = Some(self.clean_and_combine_comments(&comments));
        }

        if let Some(start) = def_start_byte {
            let end_byte = return_end_byte.or(args_end_byte);
            if let Some(end) = end_byte {
                if end > start && end <= code.len() {
                    let interface = code[start..end].trim();
                    if !interface.is_empty() {
                        func.add_interface(interface);
                    }
                }
            }
        }

        if matches!(self.kind, Language::React) {
            let titled_name = !func.name.is_empty()
                && func
                    .name
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false);
            let body = func.body.as_str();
            let has_jsx = body.contains("<>")
                || body.contains("</")
                || body.contains("/>")
                || body.contains("<Fragment")
                || body.contains("<fragment");
            let is_styled = body.contains("styled.");
            if (titled_name && has_jsx) || is_styled {
                func.add_component();
            }
        }

        let mut return_types = Vec::new();
        for t in return_type_data_models {
            return_types.push(Edge::contains(
                NodeType::Function,
                &func,
                NodeType::DataModel,
                &t,
            ));
        }

        if let Some(pos) = name_pos {
            trait_operand = self.lang.find_trait_operand(
                pos,
                &func,
                &|row, file| graph.find_node_in_range(NodeType::Trait, row, file),
                lsp_tx,
            )?;
        }
        log_cmd(format!("found function {} in file {}", func.name, file));
        Ok(Some((
            func,
            parent,
            requests_within,
            models,
            trait_operand,
            return_types,
        )))
    }
    pub fn format_test(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
    ) -> Result<NodeData> {
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
        let tt = self.lang.classify_test(&test.name, file, &test.body);
        match tt {
            NodeType::E2eTest => test.add_test_kind("e2e"),
            NodeType::IntegrationTest => test.add_test_kind("integration"),
            _ => test.add_test_kind("unit"),
        }
        Ok(test)
    }
    pub fn format_function_call<G: Graph>(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
        caller_name: &str,
        caller_start: usize,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Option<FunctionCall>> {
        let mut fc = Calls::default();
        let mut external_func = None;
        let mut class_call = None;
        let mut call_name_and_point = None;
        let mut is_variable_call = false;
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == FUNCTION_NAME {
                trace!("format_function_call {} {}", caller_name, body);
                call_name_and_point = Some((body, node.start_position()));
            } else if o == FUNCTION_CALL {
                fc.source = NodeKeys::new(&caller_name, file, caller_start);
            } else if o == OPERAND {
                fc.operand = Some(body.clone());
                if self.lang.direct_class_calls() {
                    if let Some(first_char) = body.chars().next() {
                        if first_char.is_uppercase() {
                            let possible_classes = graph.find_nodes_by_name(NodeType::Class, &body);
                            if possible_classes.len() == 1 {
                                class_call = Some(possible_classes[0].clone())
                            }
                        } else {
                            is_variable_call = true;
                        }
                    }
                }
            }
            Ok(())
        })?;

        if let None = call_name_and_point {
            return Ok(None);
        }
        let (called, call_point) = call_name_and_point.unwrap();

        if is_variable_call || self.lang.should_skip_function_call(&called, &fc.operand) {
            return Ok(None);
        }

        if called.is_empty() {
            return Ok(None);
        }

        if let Some(lsp) = lsp_tx {
            log_cmd(format!("=> {} looking for {:?}", caller_name, called));
            let pos = Position::new(file, call_point.row as u32, call_point.column as u32)?;
            let res = LspCmd::GotoDefinition(pos.clone()).send(&lsp)?;
            if let LspRes::GotoDefinition(None) = res {
                log_cmd(format!("==> _ no definition found for {:?}", called));
            }
            if let LspRes::GotoDefinition(Some(gt)) = res {
                let target_file = gt.file.display().to_string();
                if let Some(t) =
                    graph.find_node_by_name_in_file(NodeType::Function, &called, &target_file)
                {
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
                    if let Some(one_func) =
                        node_data_finder(&called, graph, file, fc.source.start, NodeType::Function)
                    {
                        log_cmd(format!(
                            "==> ? ONE target for {:?} {}",
                            called, &one_func.file
                        ));
                        fc.target = one_func.into();
                    } else {
                        log_cmd(format!(
                            "==> ? definition, not in graph: {:?} in {}",
                            called, &target_file
                        ));
                        if self.lang.is_lib_file(&target_file) {
                            if !self.lang.is_component(&called) {
                                let mut lib_func = NodeData::name_file(&called, &target_file);
                                lib_func.start = gt.line as usize;
                                lib_func.end = gt.line as usize;
                                let pos2 = Position::new(
                                    &file,
                                    call_point.row as u32,
                                    call_point.column as u32,
                                )?;
                                let hover_res = LspCmd::Hover(pos2).send(&lsp)?;
                                if let LspRes::Hover(Some(hr)) = hover_res {
                                    lib_func.docs = Some(hr);
                                }
                                external_func = Some(lib_func);
                                fc.target = NodeKeys::new(&called, &target_file, gt.line as usize);
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
                                    fc.target = NodeKeys::new(&called, &t_file.file, t_file.start);
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
            // FALLBACK to find?
            if let Some(tf) =
                node_data_finder(&called, graph, file, fc.source.start, NodeType::Function)
            {
                log_cmd(format!(
                    "==> ? (no lsp) ONE target for {:?} {}",
                    called, &tf.file
                ));
                fc.target = tf.into();
            }
        }

        if fc.target.is_empty() {
            if let Some(class_nd) = &class_call {
                if let Some(class_method) =
                    graph.find_node_by_name_in_file(NodeType::Function, &called, &class_nd.file)
                {
                    fc.target = class_method.into();
                }
            }
        }

        // target must be found OR class call
        if fc.target.is_empty() && class_call.is_none() {
            // NOTE should we only do the class call if there is no direct function target?
            return Ok(None);
        }
        Ok(Some((fc, external_func, class_call)))
    }
    pub fn format_extra<G: Graph>(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
        caller_name: &str,
        caller_start: usize,
        graph: &G,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Option<Edge>> {
        // extras
        if lsp_tx.is_none() {
            return Ok(None);
        }
        let mut pos = None;
        let mut ex = NodeData::in_file(file);
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == EXTRA_NAME {
                ex.name = trim_quotes(&body).to_string();
            } else if o == EXTRA {
                ex.body = body;
                ex.start = node.start_position().row;
                ex.end = node.end_position().row;
            } else if o == EXTRA_PROP {
                pos = Some(Position::new(
                    file,
                    node.start_position().row as u32,
                    node.start_position().column as u32,
                )?);
            }
            Ok(())
        })?;
        // unwrap is ok since we checked above
        let lsp_tx = lsp_tx.as_ref().unwrap();
        if let Some(edgy) = find_def(
            pos.clone(),
            lsp_tx,
            graph,
            &ex,
            caller_name,
            caller_start,
            NodeType::Var,
        )? {
            return Ok(Some(edgy));
        }
        if let Some(edgy) = find_def(
            pos.clone(),
            lsp_tx,
            graph,
            &ex,
            caller_name,
            caller_start,
            NodeType::DataModel,
        )? {
            return Ok(Some(edgy));
        }
        Ok(None)
    }

    pub fn format_integration_test(
        &self,
        m: &QueryMatch,
        code: &str,
        file: &str,
        q: &Query,
    ) -> Result<(NodeData, NodeType)> {
        trace!("format_integration_test");
        let mut nd = NodeData::in_file(file);
        let mut raw_name = String::new();
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == INTEGRATION_TEST || o == E2E_TEST {
                nd.body = body.clone();
                nd.start = node.start_position().row;
                nd.end = node.end_position().row;
            } else if o == TEST_NAME || o == E2E_TEST_NAME {
                raw_name = trim_quotes(&body).to_string();
            }
            Ok(())
        })?;
        nd.name = raw_name.clone();
        let tt = self.lang.classify_test(&nd.name, file, &nd.body);
        if tt == NodeType::E2eTest {
            nd.add_test_kind("e2e");
        } else if tt == NodeType::IntegrationTest {
            nd.add_test_kind("integration");
        } else {
            nd.add_test_kind("unit");
        }
        Ok((nd, tt))
    }
    pub fn format_integration_test_call<G: Graph>(
        &self,
        m: &QueryMatch,
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
            NodeRef::from(source, NodeType::IntegrationTest),
            NodeRef::from(endpoint, NodeType::Endpoint),
        );
        Ok(Some(edge))
    }

    pub fn clean_and_combine_comments(&self, comments: &[String]) -> String {
        let mut cleaned_comments = Vec::new();

        for comment in comments {
            let cleaned = self.clean_comment(comment);
            if !cleaned.is_empty() {
                cleaned_comments.push(cleaned);
            }
        }

        cleaned_comments.join("\n").trim().to_string()
    }

    fn clean_comment(&self, comment: &str) -> String {
        let lines: Vec<String> = comment
            .lines()
            .map(|line| {
                let trimmed = line.trim();
                if let Some(stripped) = trimmed.strip_prefix("///") {
                    stripped.trim().to_string()
                } else if let Some(stripped) = trimmed.strip_prefix("//") {
                    stripped.trim().to_string()
                } else if let Some(stripped) = trimmed.strip_prefix("#") {
                    stripped.trim().to_string()
                } else if let Some(stripped) = trimmed.strip_prefix("/*") {
                    let without_start = stripped.trim();
                    if let Some(without_end) = without_start.strip_suffix("*/") {
                        without_end.trim().to_string()
                    } else {
                        without_start.to_string()
                    }
                } else if let Some(stripped) = trimmed.strip_suffix("*/") {
                    stripped.trim().to_string()
                } else if let Some(stripped) = trimmed.strip_prefix("*") {
                    stripped.trim().to_string()
                } else if trimmed.starts_with("\"\"\"")
                    && trimmed.ends_with("\"\"\"")
                    && trimmed.len() > 6
                {
                    trimmed[3..trimmed.len() - 3].trim().to_string()
                } else if trimmed.starts_with("'''")
                    && trimmed.ends_with("'''")
                    && trimmed.len() > 6
                {
                    trimmed[3..trimmed.len() - 3].trim().to_string()
                } else if trimmed.starts_with("\"\"\"") || trimmed.starts_with("'''") {
                    trimmed[3..].trim().to_string()
                } else if trimmed.ends_with("\"\"\"") || trimmed.ends_with("'''") {
                    trimmed[..trimmed.len() - 3].trim().to_string()
                } else {
                    trimmed.to_string()
                }
            })
            .filter(|line| !line.is_empty())
            .collect();

        lines.join("\n")
    }
}
