use super::{graph::Graph, *};
use crate::lang::{Function, FunctionCall, Lang};
use crate::utils::{create_node_key, create_node_key_from_ref, sanitize_string};
use lsp::Language;
use serde::Serialize;
use shared::error::Result;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, Serialize, PartialEq, Eq, Default)]
pub struct BTreeMapGraph {
    pub nodes: BTreeMap<String, Node>,
    pub edges: BTreeSet<(String, String, EdgeType)>,
    #[serde(skip)]
    allow_unverified_calls: bool,
}

impl Graph for BTreeMapGraph {
    fn new(_root: String, _lang_kind: Language) -> Self {
        BTreeMapGraph {
            nodes: BTreeMap::new(),
            edges: BTreeSet::new(),
            allow_unverified_calls: false,
        }
    }

    fn with_capacity(_nodes: usize, _edges: usize, _root: String, _lang_kind: Language) -> Self
    where
        Self: Sized,
    {
        Self::default()
    }
    fn analysis(&self) {
        for node_key in self.nodes.keys() {
            println!("Node: {}", node_key);
        }

        for (src_key, dst_key, edge_type) in &self.edges {
            // if matches!(edge_type, EdgeType::Calls) {
            println!("Edge: {} - {:?} -> {}", src_key, edge_type, dst_key);
            // }
        }
    }

    fn extend_graph(&mut self, other: Self) {
        self.nodes.extend(other.nodes);
        self.edges.extend(other.edges);
    }

    fn get_graph_size(&self) -> (u32, u32) {
        (self.nodes.len() as u32, self.edges.len() as u32)
    }
    fn add_edge(&mut self, edge: &Edge) {
        let source_key = create_node_key_from_ref(&edge.source);
        let target_key = create_node_key_from_ref(&edge.target);
        self.edges
            .insert((source_key, target_key, edge.edge.clone()));
    }
    fn add_node(&mut self, node_type: &NodeType, node_data: &NodeData) {
        let node = Node::new(node_type.clone(), node_data.clone());
        let node_key = create_node_key(&node);
        self.nodes.insert(node_key, node);
    }

    fn get_graph_keys(&self) -> (HashSet<String>, HashSet<String>) {
        let node_keys: HashSet<String> = self.nodes.keys().map(|s| s.to_lowercase()).collect();

        let edge_keys: HashSet<String> = self
            .edges
            .iter()
            .map(|(src, dst, edge_type)| {
                format!(
                    "{}-{}-{}",
                    src.to_lowercase(),
                    dst.to_lowercase(),
                    edge_type
                )
            })
            .collect();
        (node_keys, edge_keys)
    }

    fn find_nodes_by_name(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        let prefix = format!(
            "{}-{}",
            sanitize_string(&node_type.to_string()),
            sanitize_string(name)
        );

        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .filter(|(_, node)| node.node_data.name == name)
            .map(|(_, node)| node.node_data.clone())
            .collect()
    }
    fn find_node_by_name_in_file(
        &self,
        node_type: NodeType,
        name: &str,
        file: &str,
    ) -> Option<NodeData> {
        let prefix = format!(
            "{}-{}-{}",
            sanitize_string(&node_type.to_string()),
            sanitize_string(name),
            sanitize_string(file)
        );

        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .map(|(_, node)| node.node_data.clone())
            .next()
    }

    fn add_node_with_parent(
        &mut self,
        node_type: &NodeType,
        node_data: &NodeData,
        parent_type: &NodeType,
        parent_file: &str,
    ) {
        self.add_node(node_type, node_data);

        let prefix = format!("{:?}-", parent_type).to_lowercase();
        if let Some((_parent_key, parent_node)) = self
            .nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .find(|(_, n)| n.node_data.file == parent_file)
        {
            let edge = Edge::contains(
                parent_type.clone(),
                &parent_node.node_data,
                node_type.clone(),
                node_data,
            );
            self.add_edge(&edge);
        }
    }

    fn create_filtered_graph(self, final_filter: &[String], lang_kind: Language) -> Self {
        let mut filtered = Self::new(String::new(), lang_kind);

        for (key, node) in &self.nodes {
            if node.node_type == NodeType::Repository || final_filter.contains(&node.node_data.file)
            {
                filtered.nodes.insert(key.clone(), node.clone());
            }
        }

        for (src, dst, edge_type) in &self.edges {
            if let (Some(src_node), Some(dst_node)) = (self.nodes.get(src), self.nodes.get(dst)) {
                if final_filter.contains(&src_node.node_data.file)
                    || final_filter.contains(&dst_node.node_data.file)
                {
                    filtered
                        .edges
                        .insert((src.clone(), dst.clone(), edge_type.clone()));
                }
            }
        }

        filtered
    }

    fn find_node_in_range(&self, node_type: NodeType, row: u32, file: &str) -> Option<NodeData> {
        let prefix = format!("{:?}-", node_type).to_lowercase();

        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .find(|(_, node)| {
                node.node_data.file == file
                    && (node.node_data.start as u32) <= row
                    && (node.node_data.end as u32) >= row
            })
            .map(|(_, node)| node.node_data.clone())
    }

    fn find_node_at(&self, node_type: NodeType, file: &str, line: u32) -> Option<NodeData> {
        let prefix = format!("{:?}-", node_type).to_lowercase();

        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .find(|(_, node)| node.node_data.file == file && node.node_data.start == line as usize)
            .map(|(_, node)| node.node_data.clone())
    }

    fn find_node_by_name_and_file_end_with(
        &self,
        node_type: NodeType,
        name: &str,
        suffix: &str,
    ) -> Option<NodeData> {
        let prefix = format!(
            "{}-{}-",
            sanitize_string(&node_type.to_string()),
            sanitize_string(name)
        );
        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .find(|(_, node)| node.node_data.file.ends_with(suffix))
            .map(|(_, node)| node.node_data.clone())
    }

    fn find_nodes_by_file_ends_with(&self, node_type: NodeType, file: &str) -> Vec<NodeData> {
        let prefix = format!("{:?}-", node_type).to_lowercase();
        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .filter(|(_, node)| node.node_data.file.ends_with(file))
            .map(|(_, node)| node.node_data.clone())
            .collect()
    }
    fn find_source_edge_by_name_and_file(
        &self,
        edge_type: EdgeType,
        target_name: &str,
        target_file: &str,
    ) -> Option<NodeKeys> {
        for (src_key, dst_key, edge) in &self.edges {
            if edge == &edge_type {
                if let (Some(src_node), Some(dst_node)) =
                    (self.nodes.get(src_key), self.nodes.get(dst_key))
                {
                    if dst_node.node_data.name == target_name
                        && dst_node.node_data.file == target_file
                    {
                        return Some(NodeKeys::from(&src_node.node_data));
                    }
                }
            }
        }
        None
    }
    fn add_instances(&mut self, instances: &[NodeData]) {
        for inst in instances {
            if let Some(of) = &inst.data_type {
                if let Some(class_node_data) = self.find_nodes_by_name(NodeType::Class, of).first()
                {
                    self.add_node_with_parent(
                        &NodeType::Instance,
                        inst,
                        &NodeType::File,
                        &inst.file,
                    );

                    let edge = Edge::of(&inst, class_node_data);
                    self.add_edge(&edge);
                }
            }
        }
    }

    fn add_functions(&mut self, functions: &[Function]) {
        for (func_node_data, method_of, reqs, dms, trait_operand, return_types, nested_in) in
            functions
        {
            let func_clone = func_node_data.clone();
            self.add_node(&NodeType::Function, func_node_data);

            if let Some(file_name) = std::path::Path::new(&func_clone.file)
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
            {
                if let Some(file_node_data) = self
                    .find_nodes_by_name(NodeType::File, &file_name)
                    .first()
                    .cloned()
                {
                    let contains_edge = Edge::contains(
                        NodeType::File,
                        &file_node_data,
                        NodeType::Function,
                        &func_clone,
                    );
                    self.add_edge(&contains_edge);
                }
            }

            if let Some(p) = method_of {
                let edge = p.clone().into();
                self.add_edge(&edge);
            }
            if let Some(to) = trait_operand {
                self.add_edge(&to);
            }
            for rt in return_types {
                self.add_edge(rt);
            }
            for ne in nested_in {
                self.add_edge(ne);
            }

            for req in reqs {
                let req_clone = req.clone();
                self.add_node(&NodeType::Request, req);
                let calls_edge = Edge::calls(
                    NodeType::Function,
                    &func_clone,
                    NodeType::Request,
                    &req_clone,
                );
                self.add_edge(&calls_edge);
            }

            for dm_edge in dms {
                self.add_edge(dm_edge);
            }
        }
    }

    fn add_page(&mut self, page: (NodeData, Option<Edge>)) {
        let (page_data, edge_opt) = page;
        self.add_node(&NodeType::Page, &page_data);

        if let Some(edge) = edge_opt {
            self.add_edge(&edge);
        }
    }

    fn add_pages(&mut self, pages: &[(NodeData, Vec<Edge>)]) {
        for (page_data, edges) in pages {
            self.add_node(&NodeType::Page, page_data);

            for edge in edges {
                self.add_edge(edge);
            }
        }
    }

    fn find_endpoint(&self, name: &str, file: &str, verb: &str) -> Option<NodeData> {
        let prefix = format!(
            "{}-{}-{}",
            &sanitize_string(&format!("{:?}", NodeType::Endpoint)),
            sanitize_string(name),
            sanitize_string(file)
        );
        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .find(|(_, node)| node.node_data.meta.get("verb") == Some(&verb.to_string()))
            .map(|(_, node)| node.node_data.clone())
    }

    fn add_endpoints(&mut self, endpoints: &[(NodeData, Option<Edge>)]) {
        for (endpoint_data, handler_edge) in endpoints {
            if endpoint_data.meta.contains_key("handler") {
                let default_verb = "".to_string();
                let verb = endpoint_data.meta.get("verb").unwrap_or(&default_verb);

                if self
                    .find_endpoint(&endpoint_data.name, &endpoint_data.file, verb)
                    .is_some()
                {
                    continue;
                }

                self.add_node(&NodeType::Endpoint, endpoint_data);

                if let Some(edge) = handler_edge {
                    self.add_edge(edge);
                }
            }
        }
    }

    fn add_tests(&mut self, tests: &[TestRecord]) {
        for tr in tests {
            let file = tr.node.file.clone();
            self.add_node_with_parent(&tr.kind, &tr.node, &NodeType::File, &file);
            for e in &tr.edges {
                self.add_edge(e);
            }
        }
    }
    // Add calls only between function definitions not between function calls
    fn add_calls(
        &mut self,
        calls: (&[FunctionCall], &[FunctionCall], &[Edge], &[Edge]),
        lang: &Lang,
    ) {
        let (funcs, tests, int_tests, extras) = calls;
        // Diagnostic flag (see array_graph.rs for rationale)
        let disable_test_class_edges = std::env::var("DISABLE_TEST_CLASS_CALLS").is_ok();
        let mut unique_edges: HashSet<(String, String, usize, String, String, usize)> =
            HashSet::new();

        for (fc, ext_func, class_call) in funcs {
            if let Some(class_call) = &class_call {
                let edge = Edge::new(
                    EdgeType::Calls,
                    NodeRef::from(fc.source.clone(), NodeType::Function),
                    NodeRef::from(class_call.into(), NodeType::Class),
                );
                self.add_edge(&edge);
            }
            if fc.target.is_empty() {
                continue;
            }

            if let Some(ext_nd) = ext_func {
                let edge_key = (
                    fc.source.name.clone(),
                    fc.source.file.clone(),
                    fc.source.start,
                    ext_nd.name.clone(),
                    ext_nd.file.clone(),
                    ext_nd.start,
                );

                if !unique_edges.contains(&edge_key) {
                    unique_edges.insert(edge_key);

                    let ext_node = Node::new(NodeType::Function, ext_nd.clone());
                    let ext_key = create_node_key(&ext_node);
                    self.nodes.entry(ext_key).or_insert(ext_node);

                    // Use CALLS edge for unverified stub nodes, USES for external/library functions
                    let edge = if ext_nd.file == "<unverified>" {
                        fc.clone().into()
                    } else {
                        Edge::uses(fc.source.clone(), &ext_nd)
                    };
                    self.add_edge(&edge);
                }
            } else {
                let edge_key = (
                    fc.source.name.clone(),
                    fc.source.file.clone(),
                    fc.source.start,
                    fc.target.name.clone(),
                    fc.target.file.clone(),
                    fc.target.start,
                );

                if !unique_edges.contains(&edge_key) {
                    unique_edges.insert(edge_key);
                    let edge = fc.clone().into();
                    self.add_edge(&edge);
                }
            }
        }

        for (tc, ext_func, class_call) in tests {
            if let Some(class_nd) = class_call {
                if !disable_test_class_edges {
                    let class_edge_key = (
                        tc.source.name.clone(),
                        tc.source.file.clone(),
                        tc.source.start,
                        class_nd.name.clone(),
                        class_nd.file.clone(),
                        class_nd.start,
                    );

                    if !unique_edges.contains(&class_edge_key) {
                        unique_edges.insert(class_edge_key);

                        let edge = Edge::from_test_class_call(&tc, &class_nd, lang, self);
                        self.add_edge(&edge);

                        // Ensure class node exists in graph
                        let class_node = Node::new(NodeType::Class, class_nd.clone());
                        let class_key = create_node_key(&class_node);
                        self.nodes.entry(class_key).or_insert(class_node);
                    }
                }
            }
            if let Some(ext_nd) = ext_func {
                let edge_key = (
                    tc.source.name.clone(),
                    tc.source.file.clone(),
                    tc.source.start,
                    ext_nd.name.clone(),
                    ext_nd.file.clone(),
                    ext_nd.start,
                );

                if !unique_edges.contains(&edge_key) {
                    unique_edges.insert(edge_key);

                    let edge = Edge::uses(tc.source.clone(), &ext_nd);
                    self.add_edge(&edge);
                    let ext_node = Node::new(NodeType::Function, ext_nd.clone());
                    let ext_key = create_node_key(&ext_node);
                    self.nodes.entry(ext_key).or_insert(ext_node);
                }
            } else if tc.target.is_empty() {
                continue;
            } else {
                let edge_key = (
                    tc.source.name.clone(),
                    tc.source.file.clone(),
                    tc.source.start,
                    tc.target.name.clone(),
                    tc.target.file.clone(),
                    tc.target.start,
                );

                if !unique_edges.contains(&edge_key) {
                    unique_edges.insert(edge_key);
                    let edge = Edge::from_test_call(&tc, lang, self);
                    self.add_edge(&edge);
                }
            }
        }

        for edge in int_tests {
            self.add_edge(edge);
        }

        for extra in extras {
            self.add_edge(extra);
        }
    }
    fn process_endpoint_groups(&mut self, eg: &[NodeData], lang: &Lang) -> Result<()> {
        let endpoints: Vec<NodeData> = self
            .nodes
            .values()
            .filter(|node| node.node_type == NodeType::Endpoint)
            .map(|node| node.node_data.clone())
            .collect();

        let find_import_node = |file: &str| -> Option<NodeData> {
            self.nodes
                .values()
                .find(|node| node.node_type == NodeType::Import && node.node_data.file == file)
                .map(|node| node.node_data.clone())
        };

        let matches = lang
            .lang()
            .match_endpoint_groups(&eg, &endpoints, &find_import_node);

        let mut best_matches: HashMap<(String, String, usize, String), (NodeData, String)> = HashMap::new();
        for (endpoint, prefix) in matches {
            let endpoint_verb = endpoint.meta.get("verb").cloned().unwrap_or_default();
            let key = (
                endpoint.name.clone(),
                endpoint.file.clone(),
                endpoint.start,
                endpoint_verb,
            );

            match best_matches.get(&key) {
                Some((_existing_ep, existing_prefix)) if prefix.len() > existing_prefix.len() => {
                    best_matches.insert(key, (endpoint, prefix));
                }
                None => {
                    best_matches.insert(key, (endpoint, prefix));
                }
                _ => {
                }
            }
        }

        let mut updates = Vec::new();
        for ((_, _, _, _), (endpoint, prefix)) in best_matches {
            if let Some((key, node)) = self.nodes.iter().find(|(_, n)| {
                n.node_type == NodeType::Endpoint
                    && n.node_data.name == endpoint.name
                    && n.node_data.file == endpoint.file
                    && n.node_data.start == endpoint.start
                    && n.node_data.meta.get("verb") == endpoint.meta.get("verb")
            }) {
                let full_path = format!("{}{}", prefix, endpoint.name);
                let mut updated_node = node.clone();
                updated_node.node_data.name = full_path;

                let edges_to_update: Vec<_> = self
                    .edges
                    .iter()
                    .filter(|(src, _, _)| src == key)
                    .map(|(_, dst, edge)| (dst.clone(), edge.clone()))
                    .collect();

                updates.push((key.clone(), updated_node, edges_to_update));
            }
        }

        for (old_key, updated_node, edges) in updates {
            let new_key = create_node_key(&updated_node);

            self.nodes.remove(&old_key);
            self.nodes.insert(new_key.clone(), updated_node);

            for (dst, edge) in edges {
                self.edges
                    .remove(&(old_key.clone(), dst.clone(), edge.clone()));
                self.edges.insert((new_key.clone(), dst, edge));
            }
        }

        Ok(())
    }
    fn class_includes(&mut self) {
        let class_nodes: Vec<_> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.node_type == NodeType::Class)
            .map(|(k, n)| (k.clone(), n.clone()))
            .collect();

        for (_, node) in class_nodes {
            if let Some(includes) = node.node_data.meta.get("includes") {
                let modules = includes.split(',').map(|m| m.trim());
                for module in modules {
                    if let Some(module_node) =
                        self.find_nodes_by_name(NodeType::Class, module).first()
                    {
                        let edge = Edge::class_imports(&node.node_data, module_node);
                        self.add_edge(&edge);
                    }
                }
            }
        }
    }
    fn class_inherits(&mut self) {
        let class_nodes: Vec<_> = self
            .nodes
            .iter()
            .filter(|(_, n)| n.node_type == NodeType::Class)
            .map(|(k, n)| (k.clone(), n.clone()))
            .collect();

        for (_, node) in class_nodes {
            if let Some(parent) = node.node_data.meta.get("parent") {
                if let Some(parent_node) = self.find_nodes_by_name(NodeType::Class, parent).first()
                {
                    let edge = Edge::parent_of(parent_node, &node.node_data);
                    self.add_edge(&edge);
                }
            }
        }
    }
    fn get_data_models_within(&mut self, lang: &Lang) {
        let prefix = format!("{:?}-", NodeType::DataModel).to_lowercase();

        let data_model_nodes: Vec<NodeData> = self
            .nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .map(|(_, node)| node.node_data.clone())
            .collect();

        for data_model in data_model_nodes {
            let edges = lang.lang().data_model_within_finder(&data_model, &|file| {
                self.find_nodes_by_file_ends_with(NodeType::Function, file)
            });

            for edge in edges {
                self.add_edge(&edge);
            }
        }
    }

    fn filter_out_nodes_without_children(
        &mut self,
        parent_type: NodeType,
        child_type: NodeType,
        child_meta_key: &str,
    ) {
        let mut has_children: BTreeMap<String, bool> = BTreeMap::new();

        let parent_prefix = format!("{:?}-", parent_type).to_lowercase();
        for (_, node) in self
            .nodes
            .range(parent_prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&parent_prefix))
        {
            has_children.insert(node.node_data.name.clone(), false);
        }

        let child_prefix = format!("{:?}-", child_type).to_lowercase();
        for (_, node) in self
            .nodes
            .range(child_prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&child_prefix))
        {
            if let Some(parent_name) = node.node_data.meta.get(child_meta_key) {
                if let Some(entry) = has_children.get_mut(parent_name) {
                    *entry = true;
                }
            }
        }
        let parent_prefix = format!("{:?}-", parent_type).to_lowercase();
        let nodes_to_remove: Vec<_> = self
            .nodes
            .range(parent_prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&parent_prefix))
            .filter(|(_, node)| !has_children.get(&node.node_data.name).unwrap_or(&true))
            .map(|(k, _)| k.clone())
            .collect();

        for key in &nodes_to_remove {
            self.nodes.remove(key);
            self.edges.retain(|(src, dst, _)| src != key && dst != key);
        }
    }

    fn remove_node(&mut self, node_type: NodeType, node_data: &NodeData) {
        let node = Node::new(node_type, node_data.clone());
        let node_key = create_node_key(&node);

        self.nodes.remove(&node_key);

        self.edges
            .retain(|(src, dst, _)| src != &node_key && dst != &node_key);
    }

    fn deduplicate_nodes(&mut self, remove_type: NodeType, keep_type: NodeType, _operation: &str) {
        let remove_prefix = format!("{:?}-", remove_type).to_lowercase();
        let nodes_to_check: Vec<NodeData> = self
            .nodes
            .range(remove_prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&remove_prefix))
            .map(|(_, node)| node.node_data.clone())
            .collect();

        let keep_prefix = format!("{:?}-", keep_type).to_lowercase();
        let mut keep_nodes_map: HashMap<(String, String), String> = HashMap::new();
        for (k, node) in self
            .nodes
            .range(keep_prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&keep_prefix))
        {
            let key = (node.node_data.name.clone(), node.node_data.file.clone());
            keep_nodes_map.insert(key, k.clone());
        }

        let mut keys_with_methods: HashSet<String> = HashSet::new();
        for (src, _, edge_type) in &self.edges {
            if edge_type == &EdgeType::Operand {
                keys_with_methods.insert(src.clone());
            }
        }

        for remove_node in nodes_to_check {
            let lookup_key = (remove_node.name.clone(), remove_node.file.clone());

            if let Some(keep_key) = keep_nodes_map.get(&lookup_key) {
                if keys_with_methods.contains(keep_key) {
                    self.remove_node(remove_type.clone(), &remove_node);
                }
            }
        }
    }

    fn find_nodes_by_name_contains(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        let prefix = format!("{:?}-", node_type).to_lowercase();
        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, n)| k.starts_with(&prefix) && n.node_data.name.contains(name))
            .map(|(_, node)| node.node_data.clone())
            .collect()
    }

    fn find_resource_nodes(&self, node_type: NodeType, verb: &str, path: &str) -> Vec<NodeData> {
        let prefix = format!("{:?}-", node_type).to_lowercase();

        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .filter(|(_, node)| {
                let node_data = &node.node_data;

                // Check if path matches
                let path_matches = node_data.name.contains(path);

                // Check if verb matches (if present in metadata)
                let verb_matches = match node_data.meta.get("verb") {
                    Some(node_verb) => node_verb.to_uppercase() == verb.to_uppercase(),
                    None => true, // If no verb in metadata, don't filter on it
                };

                path_matches && verb_matches
            })
            .map(|(_, node)| node.node_data.clone())
            .collect()
    }

    fn find_handlers_for_endpoint(&self, endpoint: &NodeData) -> Vec<NodeData> {
        let endpoint = Node::new(NodeType::Endpoint, endpoint.clone());
        let endpoint_key = create_node_key(&endpoint);

        let mut handlers = Vec::new();

        for (src, dst, edge_type) in &self.edges {
            if *edge_type == EdgeType::Handler && src == &endpoint_key {
                if let Some(node) = self.nodes.get(dst) {
                    handlers.push(node.node_data.clone());
                }
            }
        }

        handlers
    }

    fn check_direct_data_model_usage(&self, function_name: &str, data_model: &str) -> bool {
        for (src_key, dst_key, edge_type) in &self.edges {
            if *edge_type == EdgeType::Contains {
                if let (Some(src_node), Some(dst_node)) =
                    (self.nodes.get(src_key), self.nodes.get(dst_key))
                {
                    if src_node.node_data.name == function_name
                        && dst_node.node_data.name.contains(data_model)
                    {
                        return true;
                    }
                }
            }
        }

        false
    }

    fn find_functions_called_by(&self, function: &NodeData) -> Vec<NodeData> {
        let function_prefix = format!(
            "{:?}-{}-{}",
            NodeType::Function,
            sanitize_string(&function.name),
            sanitize_string(&function.file)
        )
        .to_lowercase();
        let mut called_functions = Vec::new();

        for (src, dst, edge_type) in &self.edges {
            if let EdgeType::Calls = edge_type {
                if src.starts_with(&function_prefix) {
                    if let Some(node) = self.nodes.get(dst) {
                        called_functions.push(node.node_data.clone());
                    }
                }
            }
        }

        called_functions
    }

    fn find_nodes_by_type(&self, node_type: NodeType) -> Vec<NodeData> {
        let prefix = format!("{:?}-", node_type).to_lowercase();
        self.nodes
            .range(prefix.clone()..)
            .take_while(|(k, _)| k.starts_with(&prefix))
            .map(|(_, node)| node.node_data.clone())
            .collect()
    }

    fn find_nodes_with_edge_type(
        &self,
        source_type: NodeType,
        target_type: NodeType,
        edge_type: EdgeType,
    ) -> Vec<(NodeData, NodeData)> {
        let mut result = Vec::new();
        let source_prefix = format!("{:?}-", source_type).to_lowercase();
        let target_prefix = format!("{:?}-", target_type).to_lowercase();
        for (src_key, dst_key, edge) in &self.edges {
            if *edge == edge_type
                && src_key.starts_with(&source_prefix)
                && dst_key.starts_with(&target_prefix)
            {
                if let (Some(src_node), Some(dst_node)) =
                    (self.nodes.get(src_key), self.nodes.get(dst_key))
                {
                    result.push((src_node.node_data.clone(), dst_node.node_data.clone()));
                }
            }
        }

        result
    }
    fn count_edges_of_type(&self, edge_type: EdgeType) -> usize {
        self.edges
            .iter()
            .filter(|(_, _, edge)| match (edge, &edge_type) {
                (EdgeType::Calls, EdgeType::Calls) => true,
                _ => *edge == edge_type,
            })
            .count()
    }
    fn has_edge(&self, source: &Node, target: &Node, edge_type: EdgeType) -> bool {
        let source_key = create_node_key(source);
        let target_key = create_node_key(target);

        self.find_edge_by_keys(&source_key, &target_key, &edge_type)
            .is_some()
    }
    fn get_edge_keys(&self) -> BTreeSet<(String, String, EdgeType)> {
        self.edges.clone()
    }

    fn get_edges_vec(&self) -> Vec<Edge> {
        self.edges
            .iter()
            .filter_map(|(src_key, dst_key, edge_type)| {
                let src_node = self.nodes.get(src_key)?;
                let dst_node = self.nodes.get(dst_key)?;
                Some(Edge::new(
                    edge_type.clone(),
                    NodeRef::from((&src_node.node_data).into(), src_node.node_type.clone()),
                    NodeRef::from((&dst_node.node_data).into(), dst_node.node_type.clone()),
                ))
            })
            .collect()
    }

    fn set_allow_unverified_calls(&mut self, allow: bool) {
        self.allow_unverified_calls = allow;
    }

    fn get_allow_unverified_calls(&self) -> bool {
        self.allow_unverified_calls
    }

    fn iter_all_nodes(&self) -> Box<dyn Iterator<Item = (&NodeType, &NodeData)> + '_> {
        Box::new(
            self.nodes
                .values()
                .map(|node| (&node.node_type, &node.node_data)),
        )
    }
}
impl BTreeMapGraph {
    pub fn iter_all_edges(&self) -> impl Iterator<Item = (&String, &String, &EdgeType)> {
        self.edges.iter().map(|(src, dst, edge)| (src, dst, edge))
    }

    pub fn find_edge_by_keys(
        &self,
        src_key: &str,
        dst_key: &str,
        edge_type: &EdgeType,
    ) -> Option<Edge> {
        if let (Some(src_node), Some(dst_node)) = (self.nodes.get(src_key), self.nodes.get(dst_key))
        {
            Some(Edge::new(
                edge_type.clone(),
                NodeRef::from((&src_node.node_data).into(), src_node.node_type.clone()),
                NodeRef::from((&dst_node.node_data).into(), dst_node.node_type.clone()),
            ))
        } else {
            None
        }
    }
}
impl PartialOrd for BTreeMapGraph {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BTreeMapGraph {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.nodes.cmp(&other.nodes) {
            std::cmp::Ordering::Equal => self.edges.cmp(&other.edges),
            other_ordering => other_ordering,
        }
    }
}
