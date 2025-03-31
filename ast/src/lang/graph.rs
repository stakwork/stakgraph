use super::{linker::normalize_backend_path, *};
use serde::{Deserialize, Serialize};
use tracing::debug;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
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

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
            errors: Vec::new(),
        }
    }
    pub fn add_repository(&mut self, url: &str, org: &str, name: &str, hash: &str) {
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
    pub fn add_language(&mut self, lang: &str) {
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
    pub fn get_repository(&self) -> NodeData {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                Node::Repository(r) => Some(r.clone()),
                _ => None,
            })
            .next()
            .unwrap()
    }
    pub fn add_directory(&mut self, path: &str) {
        // "file" is actually the path
        let mut d = NodeData::in_file(path);
        d.name = path.to_string();
        let edge = self.parent_edge(path, &mut d, NodeType::Directory);
        self.nodes.push(Node::Directory(d));
        self.edges.push(edge);
    }
    pub fn add_file(&mut self, path: &str, code: &str) {
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
        let edge = self.parent_edge(path, &mut f, NodeType::File);
        self.edges.push(edge);
        self.nodes.push(Node::File(f));
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
    pub fn repo_data(&self, filename: &str) -> Option<NodeData> {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                Node::Repository(r) => Some(r.clone()),
                _ => None,
            })
            .find(|r| r.name == filename)
    }
    pub fn file_data(&self, filename: &str) -> Option<NodeData> {
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
    pub fn add_classes(&mut self, classes: Vec<NodeData>) {
        for c in classes {
            if let Some(ff) = self.file_data(&c.file) {
                let edge = Edge::contains(NodeType::File, &ff, NodeType::Class, &c);
                self.edges.push(edge);
            }
            self.nodes.push(Node::Class(c));
        }
    }
    pub fn add_imports(&mut self, imports: Vec<NodeData>) {
        for i in imports {
            if let Some(ff) = self.file_data(&i.file) {
                let edge = Edge::contains(NodeType::File, &ff, NodeType::Import, &i);
                self.edges.push(edge);
            }
            self.nodes.push(Node::Import(i));
        }
    }
    pub fn add_traits(&mut self, traits: Vec<NodeData>) {
        for t in traits {
            if let Some(ff) = self.file_data(&t.file) {
                let edge = Edge::contains(NodeType::File, &ff, NodeType::Trait, &t);
                self.edges.push(edge);
            }
            self.nodes.push(Node::Trait(t));
        }
    }
    pub fn add_libs(&mut self, libs: Vec<NodeData>) {
        for l in libs {
            if let Some(ff) = self.file_data(&l.file) {
                let edge = Edge::contains(NodeType::File, &ff, NodeType::Library, &l);
                self.edges.push(edge);
            }
            self.nodes.push(Node::Library(l));
        }
    }
    pub fn add_page(&mut self, page: (NodeData, Option<Edge>)) {
        let (p, e) = page;
        self.nodes.push(Node::Page(p));
        if let Some(edge) = e {
            self.edges.push(edge);
        }
    }
    pub fn add_pages(&mut self, pages: Vec<(NodeData, Vec<Edge>)>) {
        for (p, e) in pages {
            self.nodes.push(Node::Page(p));
            for edge in e {
                self.edges.push(edge);
            }
        }
    }
    pub fn add_instances(&mut self, instances: Vec<NodeData>) {
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
    pub fn add_structs(&mut self, structs: Vec<NodeData>) {
        for s in structs {
            if let Some(ff) = self.file_data(&s.file) {
                let edge = Edge::contains(NodeType::File, &ff, NodeType::DataModel, &s);
                self.edges.push(edge);
            }
            self.nodes.push(Node::DataModel(s));
        }
    }
    pub fn add_functions(&mut self, functions: Vec<Function>) {
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
    pub fn add_tests(&mut self, tests: Vec<Function>) {
        for t in tests {
            if let Some(ff) = self.file_data(&t.0.file) {
                let edge = Edge::contains(NodeType::File, &ff, NodeType::Test, &t.0);
                self.edges.push(edge);
            }
            self.nodes.push(Node::Test(t.0));
        }
    }
    pub fn filter_functions(&self) -> Vec<NodeData> {
        self.nodes
            .iter()
            .filter_map(|n| match n {
                Node::Function(f) => Some(f.clone()),
                _ => None,
            })
            .collect()
    }
    // funcs, tests, integration tests
    pub fn add_calls(
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
    pub fn add_endpoints(&mut self, endpoints: Vec<(NodeData, Option<Edge>)>) {
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
    pub fn add_integration_test(&mut self, t: NodeData, tt: NodeType, e: Option<Edge>) {
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
    pub fn class_inherits(&mut self) {
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
    pub fn class_includes(&mut self) {
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
    // NOTE does this need to be per lang on the trait?
    pub fn process_endpoint_groups(&mut self, eg: Vec<NodeData>, lang: &Lang) -> Result<()> {
        // the group "name" needs to be added to the beginning of the names of the endpoints in the group
        for group in eg {
            // group name (like TribesHandlers)
            if let Some(g) = group.meta.get("group") {
                // function (handler) for the group
                if let Some(gf) = self.find_by_name(NodeType::Function, &g) {
                    // each individual endpoint in the group code
                    for q in lang.lang().endpoint_finders() {
                        let endpoints_in_group =
                            lang.get_query_opt(Some(q), &gf.body, &gf.file, NodeType::Endpoint)?;
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
    pub fn find_by_name(&self, nt: NodeType, name: &str) -> Option<NodeData> {
        match self.find_index_by_name(nt, name) {
            Some(idx) => Some(self.nodes[idx].into_data()),
            None => None,
        }
    }
    pub fn find_exact_func(&self, name: &str, file: &str) -> Option<NodeData> {
        let mut f = None;
        for n in self.nodes.iter() {
            match n {
                Node::Function(ff) => {
                    if ff.name == name && ff.file == file {
                        f = Some(ff.clone());
                        break;
                    }
                }
                _ => (),
            }
        }
        f
    }
    pub fn find_exact_endpoint(
        &self,
        name: &str,
        file: &str,
        verb: Option<&String>,
    ) -> Option<NodeData> {
        let mut f = None;
        for n in self.nodes.iter() {
            match n {
                Node::Endpoint(ff) => {
                    if ff.name == name && ff.file == file && ff.meta.get("verb") == verb {
                        f = Some(ff.clone());
                        break;
                    }
                }
                _ => (),
            }
        }
        f
    }
    pub fn find_index_by_name(&self, nt: NodeType, name: &str) -> Option<usize> {
        let mut f = None;
        for (i, n) in self.nodes.iter().enumerate() {
            match n {
                Node::Class(ff) => {
                    if ff.name == name && nt == NodeType::Class {
                        f = Some(i);
                        break;
                    }
                }
                Node::Function(ff) => {
                    if ff.name == name && nt == NodeType::Function {
                        f = Some(i);
                        break;
                    }
                }
                Node::Endpoint(ff) => {
                    if ff.name == name && nt == NodeType::Endpoint {
                        f = Some(i);
                        break;
                    }
                }
                Node::Instance(ff) => {
                    if ff.name == name && nt == NodeType::Instance {
                        f = Some(i);
                        break;
                    }
                }
                Node::Request(ff) => {
                    if ff.name == name && nt == NodeType::Request {
                        f = Some(i);
                        break;
                    }
                }
                Node::DataModel(dm) => {
                    if dm.name == name && nt == NodeType::DataModel {
                        f = Some(i);
                        break;
                    }
                }
                _ => (),
            }
        }
        f
    }
    pub fn find_trait_range(&self, row: u32, file: &str) -> Option<NodeData> {
        for n in self.nodes.iter() {
            if let Node::Trait(t) = n {
                if t.file == file && t.start as u32 <= row && t.end as u32 >= row {
                    return Some(t.clone());
                }
            }
        }
        None
    }
    pub fn find_edge_index_by_src(&self, name: &str, file: &str) -> Option<usize> {
        for (i, n) in self.edges.iter().enumerate() {
            if n.source.node_data.name == name && n.source.node_data.file == file {
                return Some(i);
            }
        }
        None
    }
    pub fn find_func_by<F>(&self, predicate: F) -> Option<NodeData>
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
    pub fn find_funcs_by<F>(&self, predicate: F) -> Vec<NodeData>
    where
        F: Fn(&NodeData) -> bool,
    {
        let mut fs = Vec::new();
        for n in self.nodes.iter() {
            match n {
                Node::Function(ff) => {
                    if predicate(&ff) {
                        fs.push(ff.clone());
                    }
                }
                _ => (),
            }
        }
        fs
    }
    pub fn find_edges_by<F>(&self, predicate: F) -> Vec<Edge>
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
    pub fn find_class_by<F>(&self, predicate: F) -> Option<NodeData>
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
    pub fn find_data_model_by<F>(&self, predicate: F) -> Option<NodeData>
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
    pub fn find_data_model_at(&self, file: &str, line: u32) -> Option<NodeData> {
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

    pub fn find_languages(&self) -> Vec<Node> {
        self.nodes
            .iter()
            .filter(|n| matches!(n, Node::Language(_)))
            .cloned()
            .collect::<Vec<_>>()
    }

    pub fn find_specific_endpoints(&self, verb: &str, path: &str) -> Option<Node> {
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

    pub fn find_target_by_edge_type(&self, source: &Node, edge_type: EdgeType) -> Option<Node> {
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

    pub fn find_functions_called_by_handler(&self, handler: &Node) -> Vec<Node> {
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
