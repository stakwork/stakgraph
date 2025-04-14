use super::{graph::Graph, *};
use crate::lang::linker::normalize_backend_path;
use crate::lang::{Function, FunctionCall, Lang};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tracing::debug;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArrayGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub errors: Vec<String>,
}

impl Graph for ArrayGraph {
    fn new() -> Self {
        ArrayGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            errors: Vec::new(),
        }
    }
    fn with_capacity(_nodes: usize, _edges: usize) -> Self
    where
        Self: Sized,
    {
        Self::default()
    }
    fn analysis(&self) {
        for node in &self.nodes {
            println!(
                "Node Name : {:?} for Node: {:?}",
                node.node_data.name, node.node_data
            );
        }
    }
    fn create_filtered_graph(&self, final_filter: &[String]) -> Self {
        let mut new_graph = Self::new();

        for node in &self.nodes {
            if node.node_type == NodeType::Repository {
                new_graph.nodes.push(node.clone());
                continue;
            }
            if final_filter.contains(&node.node_data.file) {
                new_graph.nodes.push(node.clone());
            }
        }

        for edge in &self.edges {
            if final_filter.contains(&edge.source.node_data.file)
                || final_filter.contains(&edge.target.node_data.file)
            {
                new_graph.edges.push(edge.clone());
            }
        }

        new_graph
    }

    fn extend_graph(&mut self, other: Self) {
        self.nodes.extend(other.nodes);
        self.edges.extend(other.edges);
        self.errors.extend(other.errors);
    }

    fn get_graph_size(&self) -> (u32, u32) {
        ((self.nodes.len() as u32), (self.edges.len() as u32))
    }
    fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    fn find_nodes_by_name(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        self.nodes
            .iter()
            .filter(|node| node.node_type == node_type && node.node_data.name == name)
            .map(|node| node.node_data.clone())
            .collect()
    }

    fn find_nodes_in_range(&self, node_type: NodeType, row: u32, file: &str) -> Option<NodeData> {
        self.nodes.iter().find_map(|node| {
            if node.node_type == node_type
                && node.node_data.file == file
                && node.node_data.start as u32 <= row
                && node.node_data.end as u32 >= row
            {
                Some(node.node_data.clone())
            } else {
                None
            }
        })
    }
    fn find_node_at(&self, node_type: NodeType, file: &str, line: u32) -> Option<NodeData> {
        self.nodes.iter().find_map(|node| {
            if node.node_type == node_type
                && node.node_data.file == file
                && node.node_data.start == line as usize
            {
                Some(node.node_data.clone())
            } else {
                None
            }
        })
    }

    fn add_node_with_parent(
        &mut self,
        node_type: NodeType,
        node_data: NodeData,
        parent_type: NodeType,
        parent_file: &str,
    ) {
        let _edge = if let Some(parent) = self
            .nodes
            .iter()
            .find(|n| n.node_type == parent_type && n.node_data.file == parent_file)
            .map(|n| n.node_data.clone())
        {
            let edge = Edge::contains(parent_type, &parent, node_type.clone(), &node_data);
            self.nodes.push(Node::new(node_type, node_data));
            self.add_edge(edge);
        } else {
            self.nodes.push(Node::new(node_type, node_data));
        };
    }
    // NOTE does this need to be per lang on the trait?
    fn process_endpoint_groups(&mut self, eg: Vec<NodeData>, lang: &Lang) -> Result<()> {
        // the group "name" needs to be added to the beginning of the names of the endpoints in the group
        for group in eg {
            // group name (like TribesHandlers)
            if let Some(g) = group.meta.get("group") {
                // function (handler) for the group
                if let Some(gf) = self.find_nodes_by_name(NodeType::Function, &g).first() {
                    // each individual endpoint in the group code
                    for q in lang.lang().endpoint_finders() {
                        let endpoints_in_group = lang.get_query_opt::<Self>(
                            Some(q),
                            &gf.body,
                            &gf.file,
                            NodeType::Endpoint,
                        )?;
                        // find the endpoint in the graph
                        for end in endpoints_in_group {
                            if let Some(idx) =
                                self.find_index_by_name(NodeType::Endpoint, &end.name)
                            {
                                let end_node = self.nodes.get_mut(idx).unwrap();
                                if end_node.node_type == NodeType::Endpoint {
                                    let new_endpoint =
                                        format!("{}{}", group.name, end_node.node_data.name);
                                    end_node.node_data.name = new_endpoint.clone();
                                    if let Some(ei) =
                                        self.find_edge_index_by_src(&end.name, &end.file)
                                    {
                                        let edge = self.edges.get_mut(ei).unwrap();
                                        edge.source.node_data.name = new_endpoint;
                                    } else {
                                        println!("missing edge for endpoint: {:?}", end);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn class_inherits(&mut self) {
        for n in self.nodes.iter() {
            if n.node_type == NodeType::Class {
                if let Some(parent) = n.node_data.meta.get("parent") {
                    if let Some(parent_node) =
                        self.find_nodes_by_name(NodeType::Class, parent).first()
                    {
                        let edge = Edge::parent_of(&parent_node, &n.node_data);
                        self.edges.push(edge);
                    }
                }
            }
        }
    }
    fn class_includes(&mut self) {
        for n in self.nodes.iter() {
            if n.node_type == NodeType::Class {
                if let Some(includes) = n.node_data.meta.get("includes") {
                    let modules = includes.split(",").map(|m| m.trim()).collect::<Vec<&str>>();
                    for m in modules {
                        if let Some(m_node) = self.find_nodes_by_name(NodeType::Class, m).first() {
                            let edge = Edge::class_imports(&n.node_data, &m_node);
                            self.edges.push(edge);
                        }
                    }
                }
            }
        }
    }

    fn add_instances(&mut self, instances: Vec<NodeData>) {
        for inst in instances {
            if let Some(of) = &inst.data_type {
                if let Some(cl) = self.find_nodes_by_name(NodeType::Class, &of).first() {
                    self.add_node_with_parent(
                        NodeType::Instance,
                        inst.clone(),
                        NodeType::File,
                        &inst.file,
                    );
                    let of_edge = Edge::of(&inst, &cl);
                    self.edges.push(of_edge);
                }
            }
        }
    }
    fn add_functions(&mut self, functions: Vec<Function>) {
        for f in functions {
            // HERE return_types
            let (node, method_of, reqs, dms, trait_operand, return_types) = f;
            if let Some(ff) = self.file_data(&node.file) {
                let edge = Edge::contains(NodeType::File, &ff, NodeType::Function, &node);
                self.add_edge(edge);
            }
            self.nodes.push(Node::new(NodeType::Function, node.clone()));
            if let Some(p) = method_of {
                self.add_edge(p.into());
            }
            if let Some(to) = trait_operand {
                self.add_edge(to.into());
            }
            for rt in return_types {
                self.add_edge(rt);
            }
            for r in reqs {
                // FIXME add operand on calls (axios, api, etc)
                self.edges.push(Edge::calls(
                    NodeType::Function,
                    &node,
                    NodeType::Request,
                    &r,
                    CallsMeta {
                        call_start: r.start,
                        call_end: r.end,
                        operand: None,
                    },
                ));
                self.nodes.push(Node::new(NodeType::Request, r));
            }
            for dm in dms {
                self.edges.push(dm);
            }
        }
    }
    fn add_page(&mut self, page: (NodeData, Option<Edge>)) {
        let (p, e) = page;
        self.nodes.push(Node::new(NodeType::Page, p));
        if let Some(edge) = e {
            self.edges.push(edge);
        }
    }
    fn add_pages(&mut self, pages: Vec<(NodeData, Vec<Edge>)>) {
        for (p, e) in pages {
            self.nodes.push(Node::new(NodeType::Page, p));
            for edge in e {
                self.add_edge(edge);
            }
        }
    }
    fn find_endpoint(&self, name: &str, file: &str, verb: &str) -> Option<NodeData> {
        self.nodes.iter().find_map(|n| {
            if n.node_type == NodeType::Endpoint
                && n.node_data.name == name
                && n.node_data.file == file
                && n.node_data.meta.get("verb") == Some(&verb.to_string())
            {
                Some(n.node_data.clone())
            } else {
                None
            }
        })
    }
    // one endpoint can have multiple handlers like in Ruby on Rails (resources)
    fn add_endpoints(&mut self, endpoints: Vec<(NodeData, Option<Edge>)>) {
        for (e, h) in endpoints {
            if let Some(_handler) = e.meta.get("handler") {
                let default_verb = "".to_string();
                let verb = e.meta.get("verb").unwrap_or(&default_verb);

                if self.find_endpoint(&e.name, &e.file, verb).is_some() {
                    continue;
                }
                self.nodes.push(Node::new(NodeType::Endpoint, e));
                if let Some(edge) = h {
                    self.edges.push(edge);
                }
            } else {
                debug!("err missing handler on endpoint!");
            }
        }
    }
    fn add_test_node(&mut self, test_data: NodeData, test_type: NodeType, test_edge: Option<Edge>) {
        self.add_node_with_parent(
            test_type,
            test_data.clone(),
            NodeType::File,
            &test_data.file,
        );

        if let Some(edge) = test_edge {
            self.edges.push(edge);
        }
    }
    // funcs, tests, integration tests
    fn add_calls(
        &mut self,
        (funcs, tests, int_tests): (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>),
    ) {
        // add lib funcs first
        for (fc, ext_func) in funcs {
            if let Some(ext_nd) = ext_func {
                self.add_edge(Edge::uses(fc.source, &ext_nd));
                // don't add if it's already in the graph
                if let None =
                    self.find_node_by_name_in_file(NodeType::Function, &ext_nd.name, &ext_nd.file)
                {
                    self.nodes.push(Node::new(NodeType::Function, ext_nd));
                }
            } else {
                self.add_edge(fc.into())
            }
        }
        for (tc, ext_func) in tests {
            if let Some(ext_nd) = ext_func {
                self.add_edge(Edge::uses(tc.source, &ext_nd));

                if let None =
                    self.find_node_by_name_in_file(NodeType::Function, &ext_nd.name, &ext_nd.file)
                {
                    self.nodes.push(Node::new(NodeType::Function, ext_nd));
                }
            } else {
                self.add_edge(Edge::new_test_call(tc));
            }
        }
        for edge in int_tests {
            self.add_edge(edge);
        }
    }
    fn find_node_by_name_in_file(
        &self,
        node_type: NodeType,
        name: &str,
        file: &str,
    ) -> Option<NodeData> {
        self.nodes.iter().find_map(|node| {
            if node.node_type == node_type
                && node.node_data.name == name
                && node.node_data.file == file
            {
                Some(node.node_data.clone())
            } else {
                None
            }
        })
    }
    fn find_node_by_name_and_file_end_with(
        &self,
        node_type: NodeType,
        name: &str,
        suffix: &str,
    ) -> Option<NodeData> {
        self.nodes.iter().find_map(|node| {
            if node.node_type == node_type
                && node.node_data.name == name
                && node.node_data.file.ends_with(suffix)
            {
                Some(node.node_data.clone())
            } else {
                None
            }
        })
    }
    fn find_nodes_by_file_ends_with(&self, node_type: NodeType, file: &str) -> Vec<NodeData> {
        self.nodes
            .iter()
            .filter(|node| node.node_type == node_type && node.node_data.file.ends_with(file))
            .map(|node| node.node_data.clone())
            .collect()
    }
    fn find_source_edge_by_name_and_file(
        &self,
        edge_type: EdgeType,
        target_name: &str,
        target_file: &str,
    ) -> Option<NodeKeys> {
        self.edges
            .iter()
            .find(|edge| {
                edge.edge == edge_type
                    && edge.target.node_data.name == target_name
                    && edge.target.node_data.file == target_file
            })
            .map(|edge| edge.source.node_data.clone())
    }
    fn filter_out_nodes_without_children(
        &mut self,
        parent_type: NodeType,
        child_type: NodeType,
        child_meta_key: &str,
    ) {
        let mut has_children: BTreeMap<String, bool> = BTreeMap::new();

        //all parents have no children
        for node in &self.nodes {
            if node.node_type == parent_type {
                has_children.insert(node.node_data.name.clone(), false);
            }
        }

        //nodes that have children
        for node in &self.nodes {
            if node.node_type == child_type {
                if let Some(parent_name) = node.node_data.meta.get(child_meta_key) {
                    if let Some(entry) = has_children.get_mut(parent_name) {
                        *entry = true;
                    }
                }
            }
        }

        //now remove nodes without children
        self.nodes.retain(|node| {
            node.node_type != parent_type
                || *has_children.get(&node.node_data.name).unwrap_or(&true)
        });
    }
    fn get_data_models_within(&mut self, lang: &Lang) {
        let data_model_nodes: Vec<NodeData> = self
            .nodes
            .iter()
            .filter(|n| n.node_type == NodeType::DataModel)
            .map(|n| n.node_data.clone())
            .collect();
        for data_model in data_model_nodes {
            let edges = lang.lang().data_model_within_finder(&data_model, &|file| {
                self.find_nodes_by_file_ends_with(NodeType::Function, file)
            });

            self.edges.extend(edges);
        }
    }
    fn prefix_paths(&mut self, root: &str) {
        for node in &mut self.nodes {
            node.add_root(root);
        }

        for edge in &mut self.edges {
            edge.add_root(root);
        }
    }
    fn find_nodes_by_name_contains(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        self.nodes
            .iter()
            .filter(|n| n.node_type == node_type && n.node_data.name.contains(name))
            .map(|n| n.node_data.clone())
            .collect()
    }

    fn find_resource_nodes(&self, node_type: NodeType, verb: &str, path: &str) -> Vec<NodeData> {
        self.nodes
            .iter()
            .filter(|node| {
                if node.node_type != node_type {
                    return false;
                }

                let node_data = &node.node_data;
                let normalized_path = normalize_backend_path(&node_data.name);

                let path_matches = normalized_path.map_or(false, |p| p.contains(path))
                    || node_data.name.contains(path);

                let verb_matches = match node_data.meta.get("verb") {
                    Some(node_verb) => node_verb.to_uppercase() == verb.to_uppercase(),
                    None => true,
                };

                path_matches && verb_matches
            })
            .map(|node| node.node_data.clone())
            .collect()
    }
    fn find_handlers_for_endpoint(&self, endpoint: &NodeData) -> Vec<NodeData> {
        // double check if the endpoint is in the graph
        let endpoint_node = self.nodes.iter().find(|n| {
            n.node_type == NodeType::Endpoint
                && n.node_data.name == endpoint.name
                && n.node_data.file == endpoint.file
        });

        if let Some(_endpoint_node) = endpoint_node {
            self.edges
                .iter()
                .filter(|edge| {
                    edge.edge == EdgeType::Handler
                        && edge.source.node_type == NodeType::Endpoint
                        && edge.source.node_data.name == endpoint.name
                        && edge.source.node_data.file == endpoint.file
                })
                .filter_map(|edge| {
                    let handler_nodes = self.find_nodes_by_name(
                        edge.target.node_type.clone(),
                        &edge.target.node_data.name,
                    );
                    handler_nodes.first().cloned()
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    fn check_direct_data_model_usage(&self, function_name: &str, data_model: &str) -> bool {
        self.edges.iter().any(|edge| {
            edge.edge == EdgeType::Contains
                && edge.source.node_data.name == function_name
                && edge.target.node_data.name.contains(data_model)
        })
    }

    fn find_functions_called_by(&self, function: &NodeData) -> Vec<NodeData> {
        let mut result = Vec::new();
        for edge in &self.edges {
            if let EdgeType::Calls(_) = edge.edge {
                if edge.source.node_data.name == function.name
                    && edge.source.node_data.file == function.file
                {
                    if let Some(target_function) = self.find_node_by_name_in_file(
                        edge.target.node_type.clone(),
                        &edge.target.node_data.name,
                        &edge.target.node_data.file,
                    ) {
                        result.push(target_function);
                    }
                }
            }
        }

        result
    }

    fn find_nodes_with_edge_type(
        &self,
        source_type: NodeType,
        target_type: NodeType,
        edge_type: EdgeType,
    ) -> Vec<(NodeData, NodeData)> {
        self.edges
            .iter()
            .filter(|edge| {
                edge.edge == edge_type
                    && edge.source.node_type == source_type
                    && edge.target.node_type == target_type
            })
            .filter_map(|edge| {
                let source_nodes = self
                    .find_nodes_by_name(edge.source.node_type.clone(), &edge.source.node_data.name);
                let target_nodes = self
                    .find_nodes_by_name(edge.target.node_type.clone(), &edge.target.node_data.name);

                if let (Some(source), Some(target)) = (source_nodes.first(), target_nodes.first()) {
                    Some((source.clone(), target.clone()))
                } else {
                    None
                }
            })
            .collect::<Vec<(NodeData, NodeData)>>()
    }
    fn count_edges_of_type(&self, edge_type: EdgeType) -> usize {
        self.edges
            .iter()
            .filter(|edge| match (&edge.edge, &edge_type) {
                (EdgeType::Calls(_), EdgeType::Calls(_)) => true,
                _ => edge.edge == edge_type,
            })
            .count()
    }

    fn find_nodes_by_type(&self, node_type: NodeType) -> Vec<NodeData> {
        self.nodes
            .iter()
            .filter(|node| node.node_type == node_type)
            .map(|node| node.node_data.clone())
            .collect()
    }
}

impl ArrayGraph {
    pub fn new() -> Self {
        ArrayGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            errors: Vec::new(),
        }
    }
    pub fn file_data(&self, filename: &str) -> Option<NodeData> {
        self.nodes.iter().find_map(|n| {
            if n.node_type == NodeType::File && n.node_data.file == filename {
                Some(n.node_data.clone())
            } else {
                None
            }
        })
    }

    pub fn find_index_by_name(&self, nt: NodeType, name: &str) -> Option<usize> {
        self.nodes
            .iter()
            .position(|n| n.node_type == nt && n.node_data.name == name)
    }

    pub fn find_edge_index_by_src(&self, name: &str, file: &str) -> Option<usize> {
        for (i, n) in self.edges.iter().enumerate() {
            if n.source.node_data.name == name && n.source.node_data.file == file {
                return Some(i);
            }
        }
        None
    }
}

impl Default for ArrayGraph {
    fn default() -> Self {
        ArrayGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            errors: Vec::new(),
        }
    }
}
