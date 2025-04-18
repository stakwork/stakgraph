pub mod array_graph;
pub mod btreemap_graph;
pub mod graph;

pub use array_graph::*;
pub use btreemap_graph::*;
pub use graph::*;

use crate::lang::asg::*;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord)]
pub enum NodeType {
    Repository,
    Language,
    Directory,
    File,
    Import,
    Library,
    Class,
    Trait,
    Instance,
    Function,
    Test,
    #[serde(rename = "E2etest")]
    E2eTest,
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node {
    pub node_type: NodeType,
    pub node_data: NodeData,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub edge: EdgeType,
    pub source: NodeRef,
    pub target: NodeRef,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, Eq, PartialEq, PartialOrd, Ord)]
pub struct CallsMeta {
    pub call_start: usize,
    pub call_end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operand: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord)]
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
    pub fn new(node_type: NodeType, node_data: NodeData) -> Self {
        Self {
            node_type,
            node_data,
        }
    }
    pub fn into_data(&self) -> NodeData {
        self.node_data.clone()
    }
    pub fn to_node_type(&self) -> NodeType {
        self.node_type.clone()
    }

    pub fn add_root(&mut self, root: &str) {
        form(root, &mut self.node_data)
    }
}

pub fn form(root: &str, nd: &mut NodeData) {
    if nd.file.starts_with("/") {
        return;
    }
    nd.file = format!("{}/{}", root, nd.file);
}
