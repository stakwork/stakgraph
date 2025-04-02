use crate::lang::asg::*;
use crate::lang::graph::*;
use crate::lang::{Function, FunctionCall, Lang};
use anyhow::Result;
use std::fmt::Debug;

pub trait Graph: Default + Debug {
    fn new() -> Self
    where
        Self: Sized,
    {
        Self::default()
    }
    fn with_capacity(_nodes: usize, _edges: usize) -> Self
    where
        Self: Sized,
    {
        Self::default()
    }
    fn nodes(&self) -> &[Node];
    fn edges(&self) -> Vec<Edge>;
    fn errors(&self) -> &[String];
    fn add_error(&mut self, error: String);
    fn errors_mut(&mut self) -> &mut Vec<String>;
    fn add_node(&mut self, node: NodeData);
    fn add_node_type(&mut self, node: Node);
    fn add_edge(&mut self, edge: Edge);
    fn get_nodes(&self) -> Vec<NodeData>;
    fn get_edges(&self) -> Vec<Edge>;
    fn get_errors(&self) -> Vec<String>;
    fn nodes_mut(&mut self) -> &mut Vec<Node>;
    fn edges_mut(&mut self) -> &mut Vec<Edge>;
    fn find_node<F>(&self, predicate: F) -> Option<&Node>
    where
        F: Fn(&Node) -> bool;
    fn find_nodes<F>(&self, predicate: F) -> Vec<&Node>
    where
        F: Fn(&Node) -> bool;
    fn remove_node(&mut self, index: usize) -> Option<Node>;
    fn remove_node_by_predicate<F>(&mut self, predicate: F) -> Vec<Node>
    where
        F: Fn(&Node) -> bool;
    fn add_repository(&mut self, url: &str, org: &str, name: &str, hash: &str);
    fn add_language(&mut self, lang: &str);
    fn add_directory(&mut self, path: &str);
    fn add_file(&mut self, path: &str, code: &str);
    fn add_classes(&mut self, classes: Vec<NodeData>);
    fn add_traits(&mut self, traits: Vec<NodeData>);
    fn add_functions(&mut self, functions: Vec<Function>);
    fn add_instances(&mut self, instances: Vec<NodeData>);
    fn add_tests(&mut self, tests: Vec<Function>);
    fn add_integration_test(&mut self, t: NodeData, tt: NodeType, e: Option<Edge>);
    fn add_structs(&mut self, structs: Vec<NodeData>);
    fn add_pages(&mut self, pages: Vec<(NodeData, Vec<Edge>)>);
    fn add_page(&mut self, page: (NodeData, Option<Edge>));
    fn add_libs(&mut self, libs: Vec<NodeData>);
    fn add_imports(&mut self, imports: Vec<NodeData>);
    fn add_endpoints(&mut self, endpoints: Vec<(NodeData, Option<Edge>)>);
    fn add_calls(&mut self, calls: (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>));
    fn class_inherits(&mut self);
    fn class_includes(&mut self);
    fn file_data(&self, filename: &str) -> Option<NodeData>;
    fn repo_data(&self, filename: &str) -> Option<NodeData>;
    fn get_repository(&self) -> NodeData;
    fn parent_edge(&self, path: &str, nd: &mut NodeData, nt: NodeType) -> Edge;
    fn filter_functions(&self) -> Vec<NodeData>;
    fn find_by_name(&self, nt: NodeType, name: &str) -> Option<NodeData>;
    fn find_exact_func(&self, name: &str, file: &str) -> Option<NodeData>;
    fn find_exact_endpoint(
        &self,
        name: &str,
        file: &str,
        verb: Option<&String>,
    ) -> Option<NodeData>;
    fn find_index_by_name(&self, nt: NodeType, name: &str) -> Option<usize>;
    fn find_trait_range(&self, row: u32, file: &str) -> Option<NodeData>;
    fn find_edge_index_by_src(&self, name: &str, file: &str) -> Option<usize>;
    fn find_func_by<F>(&self, predicate: F) -> Option<NodeData>
    where
        F: Fn(&NodeData) -> bool;
    fn find_funcs_by<F>(&self, predicate: F) -> Vec<NodeData>
    where
        F: Fn(&NodeData) -> bool;
    fn find_edges_by<F>(&self, predicate: F) -> Vec<Edge>
    where
        F: Fn(&Edge) -> bool;
    fn find_class_by<F>(&self, predicate: F) -> Option<NodeData>
    where
        F: Fn(&NodeData) -> bool;
    fn find_data_model_by<F>(&self, predicate: F) -> Option<NodeData>
    where
        F: Fn(&NodeData) -> bool;
    fn find_data_model_at(&self, file: &str, line: u32) -> Option<NodeData>;
    fn find_languages(&self) -> Vec<Node>;
    fn find_specific_endpoints(&self, verb: &str, path: &str) -> Option<Node>;
    fn find_target_by_edge_type(&self, source: &Node, edge_type: EdgeType) -> Option<Node>;
    fn find_functions_called_by_handler(&self, handler: &Node) -> Vec<Node>;
    fn process_endpoint_groups(&mut self, eg: Vec<NodeData>, lang: &Lang) -> Result<()>;
}
