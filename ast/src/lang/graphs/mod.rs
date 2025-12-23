pub mod array_graph;
pub mod btreemap_graph;
pub mod graph;
pub mod utils;

#[cfg(feature = "neo4j")]
pub mod coverage;

#[cfg(feature = "neo4j")]
pub mod graph_ops;

#[cfg(feature = "neo4j")]
pub mod neo4j;

use std::str::FromStr;

pub use array_graph::*;
pub use btreemap_graph::*;
pub use graph::*;

#[cfg(feature = "neo4j")]
pub use neo4j::*;

use shared::Error;

use crate::lang::asg::*;
use crate::lang::Lang;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord)]
pub enum NodeType {
    Repository,
    Package,
    Language,
    Directory,
    File,
    Import,
    Library,
    Class,
    Trait,
    Instance,
    Function,
    Endpoint,
    Request,
    #[serde(rename = "Datamodel")]
    DataModel,
    Feature,
    Page,
    Var,
    UnitTest,
    IntegrationTest,
    #[serde(rename = "E2etest")]
    E2eTest,
    Mock,
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

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord)]
pub struct Edge {
    pub edge: EdgeType,
    pub source: NodeRef,
    pub target: NodeRef,
    pub ref_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[serde(tag = "edge_type", content = "edge_data")]
#[serde(rename_all = "UPPERCASE")]
pub enum EdgeType {
    Calls,    // Function -> Function
    Uses,     // like Calls but for libraries
    Operand,  // Class -> Function
    ArgOf,    // Function -> Arg
    Contains, // Module -> Function/Class/Module OR File -> Function/Class/Module
    Imports,  // File -> Module
    Of,       // Instance -> Class
    Handler,  // Endpoint -> Function
    Includes, // Feature -> Function/Class/Module/Endpoint/Request/DataModel/Test
    Renders,  // Page -> Component
    #[serde(rename = "PARENT_OF")]
    ParentOf, // Class -> Class
    Implements, // Class -> Trait
    #[serde(rename = "NESTED_IN")]
    NestedIn, // Function -> Function (child to parent)
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, PartialOrd, Ord)]
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

#[derive(Debug, Clone, Default)]
pub struct TestFilters {
    pub unit_regexes: Vec<String>,
    pub integration_regexes: Vec<String>,
    pub e2e_regexes: Vec<String>,
    pub target_regex: Option<String>,
    pub ignore_dirs: Vec<String>,
}

impl Edge {
    pub fn new(edge: EdgeType, source: NodeRef, target: NodeRef) -> Self {
        Self {
            edge,
            source,
            target,
            ref_id: Uuid::new_v4().to_string(),
        }
    }
    pub fn from_test_call(call: &Calls, lang: &Lang, graph: &impl Graph) -> Edge {
        // Look up test node to get body for language-specific classification
        let test_node = graph.find_test_node(&call.source.name, &call.source.file);
        let body = test_node.as_ref().map(|n| n.body.as_str()).unwrap_or("");
        let tt = lang.lang().classify_test(&call.source.name, &call.source.file, body);
        
        let mut src_nd = NodeData::name_file(&call.source.name, &call.source.file);
        src_nd.start = call.source.start;
        let mut tgt_nd = NodeData::name_file(&call.target.name, &call.target.file);
        tgt_nd.start = call.target.start;
        Edge::test_calls(tt, &src_nd, NodeType::Function, &tgt_nd)
    }

    pub fn from_test_class_call(call: &Calls, class_target: &NodeData, lang: &Lang, graph: &impl Graph) -> Edge {
        // Look up test node to get body for language-specific classification
        let test_node = graph.find_test_node(&call.source.name, &call.source.file);
        let body = test_node.as_ref().map(|n| n.body.as_str()).unwrap_or("");
        let test_type = lang.lang().classify_test(&call.source.name, &call.source.file, body);
        
        let mut src_nd = NodeData::name_file(&call.source.name, &call.source.file);
        src_nd.start = call.source.start;
        Edge::test_calls(test_type, &src_nd, NodeType::Class, class_target)
    }

    pub fn linked_e2e_test_call(source: &NodeData, target: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Calls,
            NodeRef::from(source.into(), NodeType::E2eTest),
            NodeRef::from(target.into(), NodeType::Function),
        )
    }
    pub fn test_calls(
        test_type: NodeType,
        source: &NodeData,
        target_type: NodeType,
        target: &NodeData,
    ) -> Edge {
        let tt = match test_type {
            NodeType::UnitTest | NodeType::IntegrationTest | NodeType::E2eTest => test_type,
            _ => NodeType::UnitTest,
        };
        Edge::new(
            EdgeType::Calls,
            NodeRef::from(source.into(), tt),
            NodeRef::from(target.into(), target_type),
        )
    }
    pub fn linked_integration_test_call(source: &NodeData, target: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Calls,
            NodeRef::from(source.into(), NodeType::IntegrationTest),
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
    pub fn calls(nt1: NodeType, f: &NodeData, nt2: NodeType, c: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Calls,
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
    pub fn render_from_class(class: &NodeData, page: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Renders,
            NodeRef::from(class.into(), NodeType::Class),
            NodeRef::from(page.into(), NodeType::Page),
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
    pub fn file_imports(file: &NodeData, target_type: NodeType, target: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Imports,
            NodeRef::from(file.into(), NodeType::File),
            NodeRef::from(target.into(), target_type),
        )
    }
    pub fn implements(class: &NodeData, tr: &NodeData) -> Edge {
        Edge::new(
            EdgeType::Implements,
            NodeRef::from(class.into(), NodeType::Class),
            NodeRef::from(tr.into(), NodeType::Trait),
        )
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
            EdgeType::Calls,
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
}

pub fn form(root: &str, nd: &mut NodeData) {
    if nd.file.starts_with("/") {
        return;
    }
    nd.file = format!("{}/{}", root, nd.file);
}

impl ToString for EdgeType {
    fn to_string(&self) -> String {
        match self {
            EdgeType::ArgOf => "ARG_OF".to_string(),
            EdgeType::Contains => "CONTAINS".to_string(),
            EdgeType::Handler => "HANDLER".to_string(),
            EdgeType::Imports => "IMPORTS".to_string(),
            EdgeType::Of => "OF".to_string(),
            EdgeType::Operand => "OPERAND".to_string(),
            EdgeType::ParentOf => "PARENT_OF".to_string(),
            EdgeType::Renders => "RENDERS".to_string(),
            EdgeType::Uses => "USES".to_string(),
            EdgeType::Includes => "INCLUDES".to_string(),
            EdgeType::Calls => "CALLS".to_string(),
            EdgeType::Implements => "IMPLEMENTS".to_string(),
            EdgeType::NestedIn => "NESTED_IN".to_string(),
        }
    }
}

impl FromStr for EdgeType {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "CALLS" => Ok(EdgeType::Calls),
            "USES" => Ok(EdgeType::Uses),
            "OPERAND" => Ok(EdgeType::Operand),
            "ARG_OF" => Ok(EdgeType::ArgOf),
            "CONTAINS" => Ok(EdgeType::Contains),
            "IMPORTS" => Ok(EdgeType::Imports),
            "OF" => Ok(EdgeType::Of),
            "HANDLER" => Ok(EdgeType::Handler),
            "RENDERS" => Ok(EdgeType::Renders),
            "PARENT_OF" => Ok(EdgeType::ParentOf),
            "IMPLEMENTS" => Ok(EdgeType::Implements),
            "NESTED_IN" => Ok(EdgeType::NestedIn),
            _ => Err(Error::Custom(format!("Invalid EdgeType: {}", s))),
        }
    }
}
