use crate::lang::call_finder::func_target_file_finder;
use crate::lang::{graphs::Graph, *};
use anyhow::Result;
use lsp::{Cmd as LspCmd, Position, Res as LspRes};
use streaming_iterator::StreamingIterator;
use tracing::debug;
use tree_sitter::QueryMatch;

use super::utils::{find_def, is_capitalized, log_cmd, trim_quotes};

impl Lang {
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
                            let mut request_node = reqs[0].clone().0;
                            self.lang.update_request(&mut request_node);
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
                &|row, file| graph.find_node_in_range(NodeType::Trait, row, file),
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
        Ok(test)
    }
    pub fn format_function_call<G: Graph>(
        &self,
        m: &QueryMatch,
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
                            if let Some(one_func) =
                                func_target_file_finder(&called, &None, graph, file)
                            {
                                log_cmd(format!("==> ? ONE target for {:?} {}", called, &one_func));
                                fc.target = NodeKeys::new(&called, &one_func, 0);
                            } else {
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
                    // FALLBACK to find?
                    if let Some(tf) = func_target_file_finder(&called, &None, graph, file) {
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
}
