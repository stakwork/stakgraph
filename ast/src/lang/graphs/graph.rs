use crate::lang::{Edge, Lang, NodeType};
use crate::lang::{Function, FunctionCall};
use anyhow::Result;
use std::collections::HashSet;
use std::fmt::Debug;

use super::{EdgeType, NodeData, NodeKeys};

#[async_trait::async_trait]
pub trait Graph: Default + Debug + Send + Sync {
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
    async fn analysis(&self);
    async fn create_filtered_graph(&self, final_filter: &[String]) -> Self
    where
        Self: Sized;

    async fn extend_graph(&mut self, other: Self)
    where
        Self: Sized;

    async fn get_graph_size(&self) -> (u32, u32);

    async fn find_nodes_by_name(&self, node_type: NodeType, name: &str) -> Vec<NodeData>;
    async fn add_node_with_parent(
        &mut self,
        node_type: NodeType,
        node_data: NodeData,
        parent_type: NodeType,
        parent_file: &str,
    );
    async fn add_edge(&mut self, edge: Edge);
    async fn add_node(&mut self, node_type: NodeType, node_data: NodeData);
    async fn get_graph_keys(&self) -> (HashSet<&str>, HashSet<&str>);

    async fn find_source_edge_by_name_and_file(
        &self,
        edge_type: EdgeType,
        target_name: &str,
        target_file: &str,
    ) -> Option<NodeKeys>;

    //Special cases
    async fn process_endpoint_groups(&mut self, eg: Vec<NodeData>, lang: &Lang) -> Result<()>;
    async fn class_inherits(&mut self);
    async fn class_includes(&mut self);
    async fn add_instances(&mut self, nodes: Vec<NodeData>);
    async fn add_functions(&mut self, functions: Vec<Function>);
    async fn add_page(&mut self, page: (NodeData, Option<Edge>));
    async fn add_pages(&mut self, pages: Vec<(NodeData, Vec<Edge>)>);
    async fn add_endpoints(&mut self, endpoints: Vec<(NodeData, Option<Edge>)>);
    async fn add_test_node(&mut self, test_data: NodeData, test_type: NodeType, test_edge: Option<Edge>);
    async fn add_calls(&mut self, calls: (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>));
    async fn filter_out_nodes_without_children(
        &mut self,
        parent_type: NodeType,
        child_type: NodeType,
        child_meta_key: &str,
    );
    async fn get_data_models_within(&mut self, lang: &Lang);
    async fn prefix_paths(&mut self, root: &str);

    //Specific
    async fn find_endpoint(&self, name: &str, file: &str, verb: &str) -> Option<NodeData>;

    async fn find_resource_nodes(&self, node_type: NodeType, verb: &str, path: &str) -> Vec<NodeData>;
    async fn find_handlers_for_endpoint(&self, endpoint: &NodeData) -> Vec<NodeData>;
    async fn check_direct_data_model_usage(&self, function_name: &str, data_model: &str) -> bool;
    async fn find_functions_called_by(&self, function: &NodeData) -> Vec<NodeData>;
    async fn find_nodes_by_type(&self, node_type: NodeType) -> Vec<NodeData>;
    async fn find_nodes_with_edge_type(
        &self,
        source_type: NodeType,
        target_type: NodeType,
        edge_type: EdgeType,
    ) -> Vec<(NodeData, NodeData)>;
    async fn count_edges_of_type(&self, edge_type: EdgeType) -> usize;

    //Default implementations
    async fn find_nodes_by_name_contains(&self, node_type: NodeType, name: &str) -> Vec<NodeData> {
        self.find_nodes_by_type(node_type).await
            .into_iter()
            .filter(|node| node.name.contains(name))
            .collect()
    }

    async fn find_node_by_name_in_file(
        &self,
        node_type: NodeType,
        name: &str,
        file: &str,
    ) -> Option<NodeData> {
        self.find_nodes_by_name(node_type, name).await
            .into_iter()
            .find(|node| node.file == file)
    }

    async fn find_nodes_by_file_ends_with(&self, node_type: NodeType, file: &str) -> Vec<NodeData> {
        self.find_nodes_by_type(node_type).await
            .into_iter()
            .filter(|node| node.file.ends_with(file))
            .collect()
    }

    async fn find_node_by_name_and_file_end_with(
        &self,
        node_type: NodeType,
        name: &str,
        suffix: &str,
    ) -> Option<NodeData> {
        self.find_nodes_by_name(node_type, name).await
            .into_iter()
            .find(|node| node.file.ends_with(suffix))
    }

    async fn find_nodes_in_range(&self, node_type: NodeType, row: u32, file: &str) -> Option<NodeData> {
        self.find_nodes_by_type(node_type).await
            .into_iter()
            .find(|node| node.file == file && node.start as u32 <= row && node.end as u32 >= row)
    }

    async fn find_node_at(&self, node_type: NodeType, file: &str, line: u32) -> Option<NodeData> {
        self.find_nodes_by_type(node_type).await
            .into_iter()
            .find(|node| node.file == file && node.start == line as usize)
    }
}
