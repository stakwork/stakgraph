use super::{linker::normalize_backend_path, *};
use crate::lang::graph_trait::*;
use serde::{Deserialize, Serialize};
use tracing::debug;

#[derive(Clone, Debug, Serialize, Deserialize,Default)]
pub struct ArrayGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
pub enum NodeType {
    Repository,
    Language,
    Directory,
    File,
    Import,
    Module,
    Library,
    Class,
    Trait,
    Instance,
    Function,
    Test,
    #[serde(rename = "E2etest")]
    E2eTest,
    Arg,
    Endpoint,
    Request,
    #[serde(rename = "Datamodel")]
    DataModel,
    Feature,
    Page,
}

// pub enum TestType {
//     Unit,
//     Integration,
//     E2e,
// }

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "node_type", content = "node_data")]
pub enum Node {
    Repository(NodeData),
    Language(NodeData),
    Directory(NodeData),
    File(NodeData),
    Import(NodeData),
    Class(NodeData),
    Trait(NodeData),
    Library(NodeData),
    Instance(NodeData),
    Function(NodeData),
    Test(NodeData),
    #[serde(rename = "E2etest")]
    E2eTest(NodeData),
    Endpoint(NodeData),
    Request(NodeData),
    #[serde(rename = "Datamodel")]
    DataModel(NodeData),
    Arg(Arg),
    Module(Module),
    Feature(NodeData),
    Page(NodeData),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub edge: EdgeType,
    pub source: NodeRef,
    pub target: NodeRef,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, Eq, PartialEq)]
pub struct CallsMeta {
    pub call_start: usize,
    pub call_end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operand: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(tag = "edge_type", content = "edge_data")]
#[serde(rename_all = "UPPERCASE")]
pub enum EdgeType {
    Calls(CallsMeta), // Function -> Function
    Uses,             // like Calls but for libraries
    Operand,          // Class -> Function
    ArgOf,            // Function -> Arg
    Contains,         // Module -> Function/Class/Module OR File -> Function/Class/Module
    Imports,          // File -> Module
    Of,               // Instance -> Class
    Handler,          // Endpoint -> Function
    Includes,         // Feature -> Function/Class/Module/Endpoint/Request/DataModel/Test
    Renders,          // Page -> Component
    #[serde(rename = "PARENT_OF")]
    ParentOf, // Class -> Class
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeRef {
    pub node_type: NodeType,
    pub node_data: NodeKeys,
}

impl NodeRef {
    pub fn from(node_data: NodeKeys, node_type: NodeType) -> Self {
        Self {
            node_type,
            node_data,
        }
    }
}

impl ArrayGraph {
  

    //Common node operations for this Graph Implementation
    fn add_node_with_file_edge(&mut self, node: Node, file: &str) {
        if let Some(ff) = self.file_data(file) {
            let edge = Edge::contains(NodeType::File, &ff, node.to_node_type(), &node.into_data());
            self.edges.push(edge);
        }
        self.nodes.push(node);
    }
    fn add_node_with_edges(&mut self, node: Node, edges: Vec<Edge>) {
        self.nodes.push(node);
        self.edges.extend(edges);
    }

    fn add_node_with_parent(&mut self, node: Node, parent_type: NodeType, parent: &NodeData) {
        let edge = Edge::contains(parent_type, parent, node.to_node_type(), &node.into_data());
        self.edges.push(edge);
        self.nodes.push(node);
    }
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

    fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    fn edges(&self) -> Vec<Edge> {
        self.edges.clone()
    }
    fn errors(&self) -> &[String] {
        &self.errors
    }
    fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }
    fn errors_mut(&mut self) -> &mut Vec<String> {
        &mut self.errors
    }
    fn get_errors(&self) -> Vec<String> {
        self.errors.clone()
    }
    
    fn add_node(&mut self, node: NodeData) {
        self.nodes.push(Node::File(node));
    }
    fn add_node_type(&mut self, node: Node) {
        self.nodes.push(node);
    }

    fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    fn get_nodes(&self) -> Vec<NodeData> {
        self.nodes
            .iter()
            .map(|n| n.into_data())
            .collect::<Vec<NodeData>>()
    }
    fn get_edges(&self) -> Vec<Edge> {
        self.edges.clone()
    }
    fn nodes_mut(&mut self) -> &mut Vec<Node> {
        &mut self.nodes
    }

    fn edges_mut(&mut self) -> &mut Vec<Edge> {
        &mut self.edges
    }
     fn find_node<F>(&self, predicate: F) -> Option<&Node>
    where
        F: Fn(&Node) -> bool,
    {
        self.nodes.iter().find(|n| predicate(n))
    }

    fn find_nodes<F>(&self, predicate: F) -> Vec<&Node>
    where
        F: Fn(&Node) -> bool,
    {
        self.nodes.iter().filter(|n| predicate(n)).collect()
    }
     fn remove_node(&mut self, index: usize) -> Option<Node> {
        if index < self.nodes.len() {
            // Remove the node
            let node = self.nodes.remove(index);
            
            // Remove any associated edges
            self.edges.retain(|edge| {
                edge.source.node_data.name != node.into_data().name ||
                edge.target.node_data.name != node.into_data().name
            });
            
            Some(node)
        } else {
            None
        }
    }
    fn remove_node_by_predicate<F>(&mut self, predicate: F) -> Vec<Node>
    where
        F: Fn(&Node) -> bool,
    {
        let mut removed = Vec::new();
        
        // Collect indices to remove (in reverse order to maintain validity)
        let indices: Vec<_> = self.nodes.iter()
            .enumerate()
            .filter(|(_, node)| predicate(node))
            .map(|(i, _)| i)
            .rev()
            .collect();
        
        // Remove nodes and collect them
        for index in indices {
            if let Some(node) = self.remove_node(index) {
                removed.push(node);
            }
        }
        
        removed
    }

    fn add_repository(&mut self, url: &str, org: &str, name: &str, hash: &str) {
        let mut repo = NodeData {
            name: format!("{}/{}", org, name),
            // FIXME find main file or repo
            file: format!("main"),
            hash: Some(hash.to_string()),
            ..Default::default()
        };
        repo.add_source_link(url);
        self.nodes.push(Node::Repository(repo));
    }

    fn add_language(&mut self, lang: &str) {
        let l = NodeData {
            name: lang.to_string(),
            file: "".to_string(),
            ..Default::default()
        };
        let repo = self.get_repository();
        let edge = Edge::contains(NodeType::Repository, &repo, NodeType::Language, &l);
        self.edges.push(edge);
        self.nodes.push(Node::Language(l));
    }

    fn add_directory(&mut self, path: &str) {
        // "file" is actually the path
        let mut d = NodeData::in_file(path);
        d.name = path.to_string();
        self.add_node_with_parent(Node::Directory(d.clone()), NodeType::Directory, &d);
    }
    fn add_file(&mut self, path: &str, code: &str) {
        if self.file_data(path).is_some() {
            return;
        }
        let mut f = NodeData::in_file(path);
        f.name = path.to_string();
        let skip_file_content = std::env::var("DEV_SKIP_FILE_CONTENT").is_ok();
        if !skip_file_content {
            f.body = code.to_string();
        }
        f.hash = Some(sha256::digest(&f.body));
        self.add_node_with_parent(Node::File(f.clone()), NodeType::File, &f);
    }

    fn add_classes(&mut self, classes: Vec<NodeData>) {
        for c in classes {
            self.add_node_with_file_edge(Node::Class(c.clone()), &c.file);
        }
    }
    fn add_imports(&mut self, imports: Vec<NodeData>) {
        for i in imports {
            self.add_node_with_file_edge(Node::Import(i.clone()), &i.file);
        }
    }
    fn add_traits(&mut self, traits: Vec<NodeData>) {
        for t in traits {
            self.add_node_with_file_edge(Node::Trait(t.clone()), &t.file);
        }
    }
    fn add_libs(&mut self, libs: Vec<NodeData>) {
        for l in libs {
            self.add_node_with_file_edge(Node::Library(l.clone()), &l.file);
        }
    }
    fn add_page(&mut self, page: (NodeData, Option<Edge>)) {
        let (p, e) = page;
        self.add_node_with_edges(Node::Page(p), e.into_iter().collect());
    }
    fn add_pages(&mut self, pages: Vec<(NodeData, Vec<Edge>)>) {
        for (p, e) in pages {
            self.add_node_with_edges(Node::Page(p), e);
        }
    }
    fn add_instances(&mut self, instances: Vec<NodeData>) {
        for inst in instances {
            if let Some(of) = &inst.data_type {
                if let Some(cl) = self.find_by_name(NodeType::Class, &of) {
                    if let Some(ff) = self.file_data(&inst.file) {
                        let edge = Edge::contains(NodeType::File, &ff, NodeType::Instance, &inst);
                        self.edges.push(edge);
                    }
                    let of_edge = Edge::of(&inst, &cl);
                    self.edges.push(of_edge);
                    self.nodes.push(Node::Instance(inst));
                }
            }
        }
    }
    fn add_structs(&mut self, structs: Vec<NodeData>) {
        for s in structs {
            self.add_node_with_file_edge(Node::DataModel(s.clone()), &s.file);
        }
    }
    fn add_functions(&mut self, functions: Vec<Function>) {
        for f in functions {
            // HERE return_types
            let (node, method_of, args, reqs, dms, trait_operand, return_types) = f;
            if let Some(ff) = self.file_data(&node.file) {
                let edge = Edge::contains(NodeType::File, &ff, NodeType::Function, &node);
                self.edges.push(edge);
            }
            self.nodes.push(Node::Function(node.clone()));
            if let Some(p) = method_of {
                self.edges.push(p.into());
            }
            if let Some(to) = trait_operand {
                self.edges.push(to.into());
            }
            for a in args {
                let n = node.clone();
                self.nodes.push(Node::Arg(a.clone()));
                self.edges.push(ArgOf::new(&n, &a).into());
            }
            for rt in return_types {
                self.edges.push(rt);
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
                self.nodes.push(Node::Request(r));
            }
            for dm in dms {
                self.edges.push(dm);
            }
        }
    }
    fn add_tests(&mut self, tests: Vec<Function>) {
        for t in tests {
            if let Some(ff) = self.file_data(&t.0.file) {
                let edge = Edge::contains(NodeType::File, &ff, NodeType::Test, &t.0);
                self.edges.push(edge);
            }
            self.nodes.push(Node::Test(t.0));
        }
    }
    fn add_calls(
        &mut self,
        (funcs, tests, int_tests): (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>),
    ) {
        // add lib funcs first
        for (fc, _, ext_func) in funcs {
            if let Some(ext_nd) = ext_func {
                self.edges.push(Edge::uses(fc.source, &ext_nd));
                // don't add if it's already in the graph
                if let None = self.find_exact_func(&ext_nd.name, &ext_nd.file) {
                    self.nodes.push(Node::Function(ext_nd));
                }
            } else {
                self.edges.push(fc.into())
            }
        }
        for (tc, _, ext_func) in tests {
            if let Some(ext_nd) = ext_func {
                self.edges.push(Edge::uses(tc.source, &ext_nd));
                // don't add if it's already in the graph
                if let None = self.find_exact_func(&ext_nd.name, &ext_nd.file) {
                    self.nodes.push(Node::Function(ext_nd));
                }
            } else {
                self.edges.push(Edge::new_test_call(tc));
            }
        }
        for edg in int_tests {
            self.edges.push(edg);
        }
    }
    // one endpoint can have multiple handlers like in Ruby on Rails (resources)
    fn add_endpoints(&mut self, endpoints: Vec<(NodeData, Option<Edge>)>) {
        for (e, h) in endpoints {
            if let Some(_handler) = e.meta.get("handler") {
                if self
                    .find_exact_endpoint(&e.name, &e.file, e.meta.get("verb"))
                    .is_some()
                {
                    continue;
                }
                self.nodes.push(Node::Endpoint(e));
                if let Some(edge) = h {
                    self.edges.push(edge);
                }
            } else {
                debug!("err missing handler on endpoint!");
            }
        }
    }
    fn add_integration_test(&mut self, t: NodeData, tt: NodeType, e: Option<Edge>) {
        if let Some(ff) = self.file_data(&t.file) {
            let edge = Edge::contains(NodeType::File, &ff, tt.clone(), &t);
            self.edges.push(edge);
        }
        let node = match tt {
            NodeType::Test => Node::Test(t),
            NodeType::E2eTest => Node::E2eTest(t),
            _ => Node::Test(t),
        };
        self.nodes.push(node);
        if let Some(e) = e {
            self.edges.push(e);
        }
    }

    fn class_inherits(&mut self) {
        for n in self.nodes.iter() {
            match n {
                Node::Class(c) => {
                    if let Some(parent) = c.meta.get("parent") {
                        if let Some(parent_node) = self.find_by_name(NodeType::Class, parent) {
                            let edge = Edge::parent_of(&parent_node, &c);
                            self.edges.push(edge);
                        }
                    }
                }
                _ => (),
            }
        }
    }
    fn class_includes(&mut self) {
        for n in self.nodes.iter() {
            match n {
                Node::Class(c) => {
                    if let Some(includes) = c.meta.get("includes") {
                        let modules = includes.split(",").map(|m| m.trim()).collect::<Vec<&str>>();
                        for m in modules {
                            if let Some(m_node) = self.find_by_name(NodeType::Class, m) {
                                let edge = Edge::class_imports(&c, &m_node);
                                self.edges.push(edge);
                            }
                        }
                    }
                }
                _ => (),
            }
        }
    }
    fn get_repository(&self) -> NodeData {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                Node::Repository(r) => Some(r.clone()),
                _ => None,
            })
            .next()
            .unwrap()
    }
    fn repo_data(&self, filename: &str) -> Option<NodeData> {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                Node::Repository(r) => Some(r.clone()),
                _ => None,
            })
            .find(|r| r.name == filename)
    }
    fn file_data(&self, filename: &str) -> Option<NodeData> {
        let mut f = None;
        for n in self.nodes.iter() {
            if let Node::File(ff) = n {
                if ff.file == filename {
                    f = Some(ff.clone());
                    break;
                }
            }
        }
        f
    }
    fn parent_edge(&self, path: &str, nd: &mut NodeData, nt: NodeType) -> Edge {
        if path.contains("/") {
            let mut paths = path.split("/").collect::<Vec<&str>>();
            let file_name = paths.pop().unwrap();
            nd.name = file_name.to_string();
            let parent_name = paths.get(paths.len() - 1).unwrap();
            let mut parent_node = NodeData::in_file(&paths.join("/"));
            parent_node.name = parent_name.to_string();
            Edge::contains(NodeType::Directory, &parent_node, nt, nd)
        } else {
            let repo = self.get_repository();
            Edge::contains(NodeType::Repository, &repo, nt, nd)
        }
    }
    fn filter_functions(&self) -> Vec<NodeData> {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                Node::Function(f) => Some(f.clone()),
                _ => None,
            })
            .collect()
    }

    fn find_by_name(&self, nt: NodeType, name: &str) -> Option<NodeData> {
        self.find_node(|n| n.to_node_type() == nt && n.into_data().name == name)
            .map(|n| n.into_data())
    }
    fn find_exact_func(&self, name: &str, file: &str) -> Option<NodeData> {
        self.find_node(|n| matches!(n, Node::Function(f) if f.name == name && f.file == file))
            .map(|n| n.into_data())
    }
    fn find_exact_endpoint(&self, name: &str, file: &str, verb: Option<&String>) -> Option<NodeData> {
        self.find_node(|n| {
            matches!(n, Node::Endpoint(e) if e.name == name && 
                                           e.file == file && 
                                           e.meta.get("verb") == verb)
        }).map(|n| n.into_data())
    }

fn find_index_by_name(&self, nt: NodeType, name: &str) -> Option<usize> {
        self.nodes.iter().enumerate().find_map(|(i, node)| {
            match node {
                Node::Class(n) | Node::Function(n) | Node::Endpoint(n) |
                Node::Instance(n) | Node::Request(n) | Node::DataModel(n)
                    if n.name == name && node.to_node_type() == nt => Some(i),
                _ => None
            }
        })
    }

    fn find_trait_range(&self, row: u32, file: &str) -> Option<NodeData> {
        for n in self.nodes.iter() {
            if let Node::Trait(t) = n {
                if t.file == file && t.start as u32 <= row && t.end as u32 >= row {
                    return Some(t.clone());
                }
            }
        }
        None
    }
    fn find_edge_index_by_src(&self, name: &str, file: &str) -> Option<usize> {
        for (i, n) in self.edges.iter().enumerate() {
            if n.source.node_data.name == name && n.source.node_data.file == file {
                return Some(i);
            }
        }
        None
    }

    fn find_func_by<F>(&self, predicate: F) -> Option<NodeData>
    where
        F: Fn(&NodeData) -> bool,
    {
        let mut f = None;
        for n in self.nodes.iter() {
            match n {
                Node::Function(ff) => {
                    if predicate(&ff) {
                        f = Some(ff.clone());
                        break;
                    }
                }
                _ => (),
            }
        }
        f
    }

fn find_funcs_by<F>(&self, predicate: F) -> Vec<NodeData>
    where
        F: Fn(&NodeData) -> bool,
    {
        self.nodes.iter()
            .filter_map(|n| match n {
                Node::Function(f) if predicate(f) => Some(f.clone()),
                _ => None,
            })
            .collect()
    }

    fn find_edges_by<F>(&self, predicate: F) -> Vec<Edge>
    where
        F: Fn(&Edge) -> bool,
    {
        let mut es = Vec::new();
        for n in self.edges.iter() {
            if predicate(&n) {
                es.push(n.clone());
            }
        }
        es
    }

    fn find_data_model_by<F>(&self, predicate: F) -> Option<NodeData>
    where
        F: Fn(&NodeData) -> bool,
    {
        let mut f = None;
        for n in self.nodes.iter() {
            match n {
                Node::DataModel(dm) => {
                    if predicate(&dm) {
                        f = Some(dm.clone());
                        break;
                    }
                }
                _ => (),
            }
        }
        f
    }

    fn find_class_by<F>(&self, predicate: F) -> Option<NodeData>
    where
        F: Fn(&NodeData) -> bool,
    {
        let mut f = None;
        for n in self.nodes.iter() {
            match n {
                Node::Class(ff) => {
                    if predicate(&ff) {
                        f = Some(ff.clone());
                        break;
                    }
                }
                _ => (),
            }
        }
        f
    }
    fn find_data_model_at(&self, file: &str, line: u32) -> Option<NodeData> {
        for n in self.nodes.iter() {
            match n {
                Node::DataModel(dm) => {
                    if dm.file == file && dm.start == line as usize {
                        return Some(dm.clone());
                    }
                }
                _ => (),
            }
        }
        None
    }

    fn find_languages(&self) -> Vec<Node> {
        self.find_nodes(|n| matches!(n, Node::Language(_)))
            .into_iter()
            .cloned()
            .collect()
    }

    fn find_specific_endpoints(&self, verb: &str, path: &str) -> Option<Node> {
        let endpoints_nodes = self
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Endpoint(_)))
            .cloned()
            .collect::<Vec<_>>();

        endpoints_nodes
            .iter()
            .find(|node| {
                if let Node::Endpoint(data) = node {
                    let normalized_actual_path =
                        normalize_backend_path(&data.name).unwrap_or_default();

                    let actual_verb = match data.meta.get("verb") {
                        Some(v) => v.trim_matches('\''),
                        None => "",
                    };

                    normalized_actual_path == path
                        && actual_verb.to_uppercase() == verb.to_uppercase()
                } else {
                    false
                }
            })
            .cloned()
    }

    fn find_target_by_edge_type(&self, source: &Node, edge_type: EdgeType) -> Option<Node> {
        let source_data = source.into_data();

        for edge in &self.edges {
            if edge.edge == edge_type
                && source_data.name == edge.source.node_data.name
                && source_data.file == edge.source.node_data.file
            {
                for node in &self.nodes {
                    let node_data = node.into_data();
                    if node_data.name == edge.target.node_data.name
                        && node_data.file == edge.target.node_data.file
                        && node.to_node_type() == edge.target.node_type
                    {
                        return Some(node.clone());
                    }
                }
            }
        }

        None
    }
    fn find_functions_called_by_handler(&self, handler: &Node) -> Vec<Node> {
        let handler_data = handler.into_data();
        let mut called_functions = Vec::new();

        for edge in &self.edges {
            if let EdgeType::Calls(_) = &edge.edge {
                let source_data = &handler_data;
                if edge.source.node_data.name == source_data.name
                    && edge.source.node_data.file == source_data.file
                {
                    for node in &self.nodes {
                        let node_data = node.into_data();
                        if node_data.name == edge.target.node_data.name
                            && node_data.file == edge.target.node_data.file
                        {
                            called_functions.push(node.clone());
                        }
                    }
                }
            }
        }

        called_functions
    }

    // NOTE does this need to be per lang on the trait?
    fn process_endpoint_groups(&mut self, eg: Vec<NodeData>, lang: &Lang) -> Result<()> {
        // the group "name" needs to be added to the beginning of the names of the endpoints in the group
        for group in eg {
            // group name (like TribesHandlers)
            if let Some(g) = group.meta.get("group") {
                // function (handler) for the group
                if let Some(gf) = self.find_by_name(NodeType::Function, &g) {
                    // each individual endpoint in the group code
                    for q in lang.lang().endpoint_finders() {
                        let endpoints_in_group =
                            lang.get_query_opt::<ArrayGraph>(Some(q), &gf.body, &gf.file, NodeType::Endpoint)?;
                        // find the endpoint in the graph
                        for end in endpoints_in_group {
                            if let Some(idx) =
                                self.find_index_by_name(NodeType::Endpoint, &end.name)
                            {
                                let end_node = self.nodes.get_mut(idx).unwrap();
                                if let Node::Endpoint(e) = end_node {
                                    let new_endpoint = format!("{}{}", group.name, e.name);
                                    e.name = new_endpoint.clone();
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
}

impl Edge {
    pub fn new(edge: EdgeType, source: NodeRef, target: NodeRef) -> Self {
        Self {
            edge,
            source,
            target,
        }
    }
    fn new_test_call(m: Calls) -> Edge {
        Edge::new(
            EdgeType::Calls(CallsMeta {
                call_start: m.call_start,
                call_end: m.call_end,
                operand: m.operand,
            }),
            NodeRef::from(m.source, NodeType::Test),
            NodeRef::from(m.target, NodeType::Function),
        )
    }
    pub fn linked_e2e_test_call(source: &NodeData, target: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Calls(CallsMeta {
                call_start: source.start,
                call_end: source.end,
                operand: None,
            }),
            NodeRef::from(source.into(), NodeType::E2eTest),
            NodeRef::from(target.into(), NodeType::Function),
        )
    }
    pub fn contains(nt1: NodeType, f: &NodeData, nt2: NodeType, c: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Contains,
            NodeRef::from(f.into(), nt1),
            NodeRef::from(c.into(), nt2),
        )
    }
    pub fn calls(nt1: NodeType, f: &NodeData, nt2: NodeType, c: &NodeData, cm: CallsMeta) -> Edge {
        Edge::new(
            EdgeType::Calls(cm),
            NodeRef::from(f.into(), nt1),
            NodeRef::from(c.into(), nt2),
        )
    }
    pub fn uses(f: NodeKeys, c: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Uses,
            NodeRef::from(f, NodeType::Function),
            NodeRef::from(c.into(), NodeType::Function),
        )
    }
    fn of(f: &NodeData, c: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Of,
            NodeRef::from(f.into(), NodeType::Instance),
            NodeRef::from(c.into(), NodeType::Class),
        )
    }
    pub fn handler(e: &NodeData, f: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Handler,
            NodeRef::from(e.into(), NodeType::Endpoint),
            NodeRef::from(f.into(), NodeType::Function),
        )
    }
    pub fn renders(e: &NodeData, f: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Renders,
            NodeRef::from(e.into(), NodeType::Page),
            NodeRef::from(f.into(), NodeType::Function),
        )
    }
    pub fn trait_operand(t: &NodeData, f: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Operand,
            NodeRef::from(t.into(), NodeType::Trait),
            NodeRef::from(f.into(), NodeType::Function),
        )
    }
    pub fn parent_of(c: &NodeData, p: &NodeData) -> Edge {
        Edge::new(
            EdgeType::ParentOf,
            NodeRef::from(c.into(), NodeType::Class),
            NodeRef::from(p.into(), NodeType::Class),
        )
    }
    pub fn class_imports(c: &NodeData, m: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Imports,
            NodeRef::from(c.into(), NodeType::Class),
            NodeRef::from(m.into(), NodeType::Class),
        )
    }
    pub fn add_root(&mut self, root: &str) {
        self.source.node_data.file = format!("{}/{}", root, self.source.node_data.file);
        self.target.node_data.file = format!("{}/{}", root, self.target.node_data.file);
    }
}

impl From<Operand> for Edge {
    fn from(m: Operand) -> Self {
        Edge::new(
            EdgeType::Operand,
            NodeRef::from(m.source, NodeType::Class),
            NodeRef::from(m.target, NodeType::Function),
        )
    }
}
impl From<ArgOf> for Edge {
    fn from(m: ArgOf) -> Self {
        Edge::new(
            EdgeType::ArgOf,
            NodeRef::from(m.source, NodeType::Function),
            NodeRef::from(m.target, NodeType::Arg),
        )
    }
}
impl From<Calls> for Edge {
    fn from(m: Calls) -> Self {
        Edge::new(
            EdgeType::Calls(CallsMeta {
                call_start: m.call_start,
                call_end: m.call_end,
                operand: m.operand,
            }),
            NodeRef::from(m.source, NodeType::Function),
            NodeRef::from(m.target, NodeType::Function),
        )
    }
}

impl Node {
    pub fn into_data(&self) -> NodeData {
        match self {
            Node::Import(i) => i.clone(),
            Node::Class(c) => c.clone(),
            Node::Instance(i) => i.clone(),
            Node::Function(f) => f.clone(),
            Node::Test(t) => t.clone(),
            Node::File(f) => f.clone(),
            Node::Repository(r) => r.clone(),
            Node::Endpoint(e) => e.clone(),
            Node::Request(r) => r.clone(),
            Node::DataModel(d) => d.clone(),
            Node::Feature(f) => f.clone(),
            Node::Page(p) => p.clone(),
            Node::Language(l) => l.clone(),
            Node::Directory(d) => d.clone(),
            Node::Library(l) => l.clone(),
            Node::E2eTest(t) => t.clone(),
            Node::Trait(t) => t.clone(),
            Node::Module(_m) => Default::default(),
            Node::Arg(_a) => Default::default(),
        }
    }
    pub fn to_node_type(&self) -> NodeType {
        match self {
            Node::Class(_) => NodeType::Class,
            Node::Trait(_) => NodeType::Trait,
            Node::Instance(_) => NodeType::Instance,
            Node::Function(_) => NodeType::Function,
            Node::Test(_) => NodeType::Test,
            Node::File(_) => NodeType::File,
            Node::Repository(_) => NodeType::Repository,
            Node::Endpoint(_) => NodeType::Endpoint,
            Node::Request(_) => NodeType::Request,
            Node::DataModel(_) => NodeType::DataModel,
            Node::Feature(_) => NodeType::Feature,
            Node::Page(_) => NodeType::Page,
            Node::Arg(_) => NodeType::Arg,
            Node::Library(_) => NodeType::Library,
            Node::E2eTest(_) => NodeType::E2eTest,
            Node::Language(_) => NodeType::Language,
            Node::Directory(_) => NodeType::Directory,
            Node::Import(_) => NodeType::Import,
            Node::Module(_) => NodeType::Module,
        }
    }

    pub fn add_root(&mut self, root: &str) {
        match self {
            Node::File(f) => form(root, f),
            Node::Endpoint(e) => form(root, e),
            Node::Request(r) => form(root, r),
            Node::DataModel(d) => form(root, d),
            Node::Feature(f) => form(root, f),
            Node::Page(p) => form(root, p),
            Node::Import(i) => form(root, i),
            Node::Class(c) => form(root, c),
            Node::Trait(t) => form(root, t),
            Node::Instance(i) => form(root, i),
            Node::Language(l) => form(root, l),
            Node::Directory(d) => form(root, d),
            Node::Repository(r) => form(root, r),
            Node::Library(l) => form(root, l),
            Node::E2eTest(t) => form(root, t),
            Node::Test(t) => form(root, t),
            Node::Function(f) => form(root, f),
            Node::Module(_m) => (),
            Node::Arg(a) => a.file = format!("{}/{}", root, a.file),
        }
    }
}

pub fn form(root: &str, nd: &mut NodeData) {
    if nd.file.starts_with("/") {
        return;
    }
    nd.file = format!("{}/{}", root, nd.file);
}
