use crate::lang::{graphs::Graph, *};
use lsp::{Cmd as LspCmd, Position, Res as LspRes};
use shared::Result;
use tracing::debug;
use tree_sitter::{Node as TreeNode, QueryMatch};

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
    if let Some(stripped) = value.strip_prefix(':') {
        return stripped;
    }
    value
}

pub fn clean_class_name(name: &str) -> String {
    name.trim_start_matches("::").to_string()
}

pub fn log_cmd(cmd: String) {
    debug!("{}", cmd);
}

pub fn is_capitalized(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    name.chars().next().unwrap().is_uppercase()
}

// FIXME also find it its in range!!! not just on the line!!!
pub fn find_def<G: Graph>(
    pos: Option<Position>,
    lsp_tx: &CmdSender,
    graph: &G,
    ex: &NodeData,
    caller_name: &str,
    caller_start: usize,
    node_type: NodeType,
) -> Result<Option<Edge>> {
    if pos.is_none() {
        return Ok(None);
    }
    let pos = pos.unwrap();
    // unwrap is ok since we checked above
    let res = LspCmd::GotoDefinition(pos).send(lsp_tx)?;
    if let LspRes::GotoDefinition(Some(gt)) = res {
        let target_file = gt.file.display().to_string();
        let target_row = gt.line;
        if let Some(t_node) = graph.find_node_in_range(node_type.clone(), target_row, &target_file)
        {
            log_cmd(format!(
                "==> {} ! found extra target for {:?} {:?}!!!",
                caller_name, ex.name, &t_node.name
            ));
            let tt = &t_node;
            let caller_keys = NodeKeys {
                name: caller_name.to_string(),
                file: ex.file.to_string(),
                start: caller_start,
                verb: None,
            };
            return Ok(Some(Edge::new(
                EdgeType::Calls,
                NodeRef::from(caller_keys, NodeType::Function),
                NodeRef::from(tt.into(), node_type),
            )));
        }
    }

    Ok(None)
}

pub fn extract_methods_from_handler(handler_body: &str, lang: &Lang) -> Vec<String> {
    let mut methods = Vec::new();
    if let Some(query_str) = lang.lang().handler_method_query() {
        if let Ok(tree) = lang.lang().parse(handler_body, &NodeType::Function) {
            let query = lang.q(&query_str, &NodeType::Function);
            let mut cursor = tree_sitter::QueryCursor::new();
            let mut matches = cursor.matches(&query, tree.root_node(), handler_body.as_bytes());

            while let Some(m) = matches.next() {
                for capture in m.captures {
                    if let Ok(method_text) = capture.node.utf8_text(handler_body.as_bytes()) {
                        let verb = method_text.to_uppercase();
                        if !methods.contains(&verb) {
                            methods.push(verb);
                        }
                    }
                }
            }
        }
    }

    methods
}

impl Lang {
    pub fn find_strings(&self, node: TreeNode, code: &str, file: &str) -> Result<Vec<String>> {
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
    pub fn find_type_identifiers(
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
    pub fn loop_captures<F>(q: &Query, m: &QueryMatch, code: &str, mut cb: F) -> Result<()>
    where
        F: FnMut(String, TreeNode, String) -> Result<()>,
    {
        for o in q.capture_names().iter() {
            if let Some(ci) = q.capture_index_for_name(o) {
                let mut nodes = m.nodes_for_capture_index(ci);
                if let Some(node) = nodes.next() {
                    let body = node.utf8_text(code.as_bytes())?.to_string();
                    if let Err(e) = cb(body, node, o.to_string()) {
                        tracing::warn!("error in loop_captures {:?}", e);
                    }
                }
            }
        }
        Ok(())
    }
    pub fn loop_captures_multi<F>(q: &Query, m: &QueryMatch, code: &str, mut cb: F) -> Result<()>
    where
        F: FnMut(String, TreeNode, String) -> Result<()>,
    {
        for o in q.capture_names().iter() {
            if let Some(ci) = q.capture_index_for_name(o) {
                let nodes = m.nodes_for_capture_index(ci);
                for node in nodes {
                    let body = node.utf8_text(code.as_bytes())?.to_string();
                    if let Err(e) = cb(body, node, o.to_string()) {
                        tracing::warn!("error in loop_captures {:?}", e);
                    }
                }
            }
        }
        Ok(())
    }
}
