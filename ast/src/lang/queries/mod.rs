pub mod angular;
pub mod bash;
pub mod consts;
pub mod erb;
pub mod go;
pub mod haml;
pub mod java;
pub mod kotlin;
pub mod python;
mod rails_routes;
pub mod react;
pub mod ruby;
pub mod rust;
pub mod svelte;
pub mod swift;
pub mod toml;
pub mod typescript;

use crate::lang::asg::Operand;
use crate::lang::graph::Edge;
use crate::lang::{Function, NodeData, NodeType};
use anyhow::Result;
use lsp::Language as LspLanguage;
use lsp::{CmdSender, Position};
use tree_sitter::{Node as TreeNode, Query, Tree};

#[derive(Default, Debug)]
pub enum HandlerItemType {
    Collection,
    Member,
    ResourceMember,
    #[default]
    Namespace,
}

#[derive(Default, Debug)]
pub struct HandlerItem {
    pub item_type: HandlerItemType,
    pub name: String,
}

#[derive(Default, Debug)]
pub struct HandlerParams {
    pub actions_array: Option<String>,
    pub item: Option<HandlerItem>,
    pub parents: Vec<HandlerItem>, // nested resources OR namespaces in RoR
}

pub trait Stack {
    fn q(&self, q: &str, nt: &NodeType) -> Query;
    // use different parser for pkg files
    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree>;
    fn module_query(&self) -> Option<String> {
        None
    }
    fn lib_query(&self) -> Option<String> {
        None
    }
    fn is_lib_file(&self, file_name: &str) -> bool {
        // default: absolute path is library func
        file_name.starts_with("/")
    }
    fn is_component(&self, _func_name: &str) -> bool {
        false
    }
    // hack for now: imports are all concatenated into one section
    // so must be ONLY at the beginning of the file, with no other elements
    // only empty lines will be added between imports
    fn imports_query(&self) -> Option<String> {
        None
    }
    fn trait_query(&self) -> Option<String> {
        None
    }
    fn class_definition_query(&self) -> String;
    fn instance_definition_query(&self) -> Option<String> {
        None
    }
    fn function_definition_query(&self) -> String;
    fn test_query(&self) -> Option<String> {
        None
    }
    // fn method_definition_query(&self) -> Option<String>;
    fn function_call_query(&self) -> String;
    // this is optional if the one above captures both
    // fn method_call_query(&self) -> Option<String>;
    fn identifier_query(&self) -> String {
        format!("(identifier) @identifier")
    }
    fn type_identifier_node_name(&self) -> String {
        "type_identifier".to_string()
    }
    fn string_node_name(&self) -> String {
        "string".to_string()
    }
    // data model definitions
    fn data_model_query(&self) -> Option<String> {
        None
    }
    // data model CONTAINS edge within a function
    fn data_model_within_query(&self) -> Option<String> {
        None
    }
    fn data_model_path_filter(&self) -> Option<String> {
        None
    }
    fn use_data_model_within_finder(&self) -> bool {
        false
    }
    fn data_model_within_finder(
        &self,
        _dm: &NodeData,
        _find_fns_in: &dyn Fn(&str) -> Vec<NodeData>,
    ) -> Vec<Edge> {
        Vec::new()
    }
    fn data_model_name(&self, dm_name: &str) -> String {
        dm_name.to_string()
    }
    fn find_function_parent(
        &self,
        _node: TreeNode,
        _code: &str,
        _file: &str,
        _func_name: &str,
        _callback: &dyn Fn(&str) -> Option<NodeData>,
        _parent_type: Option<&str>,
    ) -> Result<Option<Operand>> {
        Ok(None)
    }
    fn find_trait_operand(
        &self,
        _pos: Position,
        _nd: &NodeData,
        _callback: &dyn Fn(u32, &str) -> Option<NodeData>,
        _lsp_tx: &Option<CmdSender>,
    ) -> Result<Option<Edge>> {
        Ok(None)
    }
    // not used:
    // fn endpoint_handler_queries(&self) -> Vec<String> {
    //     Vec::new()
    // }
    fn endpoint_finders(&self) -> Vec<String> {
        Vec::new()
    }
    fn find_endpoint_parents(
        &self,
        _node: TreeNode,
        _code: &str,
        _file: &str,
        _callback: &dyn Fn(&str) -> Option<NodeData>,
    ) -> Result<Vec<HandlerItem>> {
        Ok(Vec::new())
    }
    fn endpoint_group_find(&self) -> Option<String> {
        None
    }
    fn endpoint_path_filter(&self) -> Option<String> {
        None
    }
    fn request_finder(&self) -> Option<String> {
        None
    }
    fn is_test(&self, _func_name: &str, _func_file: &str) -> bool {
        false
    }
    fn is_test_file(&self, _filename: &str) -> bool {
        false
    }
    fn add_endpoint_verb(&self, _nd: &mut NodeData, _call: &Option<String>) {}
    fn update_endpoint_verb(&self, _nd: &mut NodeData, _call: &Option<String>) {}
    // this one should be the same for all langs?
    fn filter_tests(&self, funcs: Vec<Function>) -> (Vec<Function>, Vec<Function>) {
        let mut fs = Vec::new();
        let mut ts = Vec::new();
        for func in funcs {
            if self.is_test(&func.0.name, &func.0.file) {
                ts.push(func);
            } else {
                fs.push(func);
            }
        }
        (fs, ts)
    }
    fn e2e_test_id_finder_string(&self) -> Option<String> {
        None
    }
    fn use_handler_finder(&self) -> bool {
        false
    }
    fn handler_finder(
        &self,
        endpoint: NodeData,
        find_fn: &dyn Fn(&str, &str) -> Option<NodeData>,
        _find_fns_in: &dyn Fn(&str) -> Vec<NodeData>,
        _handler_params: HandlerParams,
    ) -> Vec<(NodeData, Option<Edge>)> {
        if let Some(handler) = endpoint.meta.get("handler") {
            if let Some(nd) = find_fn(handler, &endpoint.file) {
                //make sure you pass the suffix of the file
                let edge = Edge::handler(&endpoint, &nd);
                return vec![(endpoint, Some(edge))];
            }
        }
        Vec::new()
    }
    fn integration_test_query(&self) -> Option<String> {
        None
    }
    fn use_integration_test_finder(&self) -> bool {
        false
    }
    fn integration_test_edge_finder(
        &self,
        _nd: &NodeData,
        _callback: &dyn Fn(&str) -> Option<NodeData>,
        _tt: NodeType,
    ) -> Option<Edge> {
        None
    }
    fn is_router_file(&self, _file_name: &str, _code: &str) -> bool {
        false
    }
    fn page_query(&self) -> Option<String> {
        None
    }
    fn use_extra_page_finder(&self) -> bool {
        false
    }
    fn is_extra_page(&self, _file_name: &str) -> bool {
        false
    }
    fn extra_page_finder(
        &self,
        _file_name: &str,
        _callback: &dyn Fn(&str, &str) -> Option<NodeData>,
    ) -> Option<Edge> {
        None
    }

    fn clean_graph(&mut self, _callback: &dyn Fn(NodeType, NodeType, &str)) -> bool {
        false
    }
}

pub fn treesitter_from_lsp_language(ll: LspLanguage) -> tree_sitter::Language {
    match ll {
        LspLanguage::Bash => tree_sitter_bash::LANGUAGE.into(),
        LspLanguage::Go => tree_sitter_go::LANGUAGE.into(),
        LspLanguage::Python => tree_sitter_python::LANGUAGE.into(),
        LspLanguage::Ruby => tree_sitter_ruby::LANGUAGE.into(),
        LspLanguage::Toml => tree_sitter_toml_ng::LANGUAGE.into(),
        LspLanguage::Kotlin => tree_sitter_kotlin_sg::LANGUAGE.into(),
        LspLanguage::Swift => tree_sitter_swift::LANGUAGE.into(),
        LspLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
        LspLanguage::Java => tree_sitter_java::LANGUAGE.into(),
        LspLanguage::Typescript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        LspLanguage::React => tree_sitter_typescript::LANGUAGE_TSX.into(),
        LspLanguage::Svelte => tree_sitter_svelte_ng::LANGUAGE.into(),
        LspLanguage::Angular => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        // _ => tree_sitter_bash::LANGUAGE.into(),
    }
}

impl HandlerItem {
    pub fn new_member(name: &str) -> Self {
        HandlerItem {
            item_type: HandlerItemType::Member,
            name: name.to_string(),
        }
    }
    pub fn new_collection(name: &str) -> Self {
        HandlerItem {
            item_type: HandlerItemType::Collection,
            name: name.to_string(),
        }
    }
    pub fn new_resource_member(name: &str) -> Self {
        HandlerItem {
            item_type: HandlerItemType::ResourceMember,
            name: name.to_string(),
        }
    }
}

use std::collections::BTreeMap;
