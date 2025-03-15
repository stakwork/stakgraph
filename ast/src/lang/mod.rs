pub mod asg;
pub mod graph;
pub mod linker;
pub mod parse;
pub mod queries;

use anyhow::{Context, Result};
use asg::*;
use consts::*;
use graph::*;
pub use graph::{Graph, NodeType};
use lsp::{CmdSender, Language};
use queries::*;
use std::fmt;
use std::str::FromStr;
use streaming_iterator::{IntoStreamingIterator, StreamingIterator};
use tracing::trace;
use tree_sitter::{Node as TreeNode, Query, QueryCursor, QueryMatch};

pub struct Lang {
    pub kind: Language,
    lang: Box<dyn Stack>,
}

impl fmt::Display for Lang {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lang Kind: {:?}", self.kind)
    }
}

// function, operand, args, requests within, data models within, trait operand, return types
pub type Function = (
    NodeData,
    Option<Operand>,
    Vec<Arg>,
    Vec<NodeData>,
    Vec<Edge>,
    Option<Edge>,
    Vec<Edge>,
);
// Calls, args, external function (from library or std)
pub type FunctionCall = (Calls, Vec<Arg>, Option<NodeData>);

impl Lang {
    pub fn new_python() -> Self {
        Self {
            kind: Language::Python,
            lang: Box::new(python::Python::new()),
        }
    }
    pub fn new_go() -> Self {
        Self {
            kind: Language::Go,
            lang: Box::new(go::Go::new()),
        }
    }
    pub fn new_rust() -> Self {
        Self {
            kind: Language::Rust,
            lang: Box::new(rust::Rust::new()),
        }
    }
    pub fn new_react_ts() -> Self {
        Self {
            kind: Language::Typescript,
            lang: Box::new(react_ts::ReactTs::new()),
        }
    }
    pub fn new_ruby() -> Self {
        Self {
            kind: Language::Ruby,
            lang: Box::new(ruby::Ruby::new()),
        }
    }
    pub fn lang(&self) -> &dyn Stack {
        self.lang.as_ref()
    }
    pub fn q(&self, q: &str, nt: &NodeType) -> Query {
        self.lang.q(q, nt)
    }
    pub fn get_libs(&self, code: &str, file: &str) -> Result<Vec<NodeData>> {
        if let Some(qo) = self.lang.lib_query() {
            let qo = self.q(&qo, &NodeType::Library);
            Ok(self.collect(&qo, code, file, NodeType::Library)?)
        } else {
            Ok(Vec::new())
        }
    }
    pub fn get_classes(&self, code: &str, file: &str) -> Result<Vec<NodeData>> {
        let qo = self.q(&self.lang.class_definition_query(), &NodeType::Class);
        Ok(self.collect(&qo, code, file, NodeType::Class)?)
    }
    pub fn get_traits(&self, code: &str, file: &str) -> Result<Vec<NodeData>> {
        if let Some(qo) = self.lang.trait_query() {
            let qo = self.q(&qo, &NodeType::Trait);
            Ok(self.collect(&qo, code, file, NodeType::Trait)?)
        } else {
            Ok(Vec::new())
        }
    }
    pub fn get_imports(&self, code: &str, file: &str) -> Result<Vec<NodeData>> {
        if let Some(qo) = self.lang.imports_query() {
            let qo = self.q(&qo, &NodeType::Import);
            Ok(self.collect(&qo, code, file, NodeType::Import)?)
        } else {
            Ok(Vec::new())
        }
    }
    pub fn get_pages(
        &self,
        code: &str,
        file: &str,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<Vec<(NodeData, Option<Edge>)>> {
        if let Some(qo) = self.lang.page_query() {
            let qo = self.q(&qo, &NodeType::Page);
            Ok(self.collect_pages(&qo, code, file, lsp_tx)?)
        } else {
            Ok(Vec::new())
        }
    }
    pub fn get_identifier_for_node(&self, node: TreeNode, code: &str) -> Result<Option<String>> {
        let query = self.q(&self.lang.identifier_query(), &NodeType::Function);
        let ident = Self::get_identifier_for_query(query, node, code)?;
        Ok(ident)
    }
    pub fn get_identifier_for_query(
        query: Query,
        node: TreeNode,
        code: &str,
    ) -> Result<Option<String>> {
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&query, node, code.as_bytes());
        let first = matches.next();
        if first.is_none() {
            return Ok(None);
        }
        let mut cs = first.unwrap().captures.iter().into_streaming_iter_ref();
        let name_node = cs.next().context("no name_node")?;
        let name = name_node.node.utf8_text(code.as_bytes())?;
        Ok(Some(name.to_string()))
    }
    // returns (Vec<Function>, Vec<Test>)
    pub fn get_functions_and_tests(
        &self,
        code: &str,
        file: &str,
        graph: &Graph,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<(Vec<Function>, Vec<Function>)> {
        let qo = self.q(&self.lang.function_definition_query(), &NodeType::Function);
        let funcs1 = self.collect_functions(&qo, code, file, graph, lsp_tx)?;
        let (funcs, mut tests) = self.lang.filter_tests(funcs1);
        if let Some(tq) = self.lang.test_query() {
            let qo2 = self.q(&tq, &NodeType::Test);
            let more_tests = self.collect_tests(&qo2, code, file)?;
            tests.extend(more_tests);
        }
        Ok((funcs, tests))
    }
    pub fn get_query_opt(
        &self,
        q: Option<String>,
        code: &str,
        file: &str,
        fmtr: NodeType,
    ) -> Result<Vec<NodeData>> {
        if let Some(qo) = q {
            let insts = self.collect(&self.q(&qo, &fmtr), code, file, fmtr)?;
            Ok(insts)
        } else {
            Ok(Vec::new())
        }
    }
    // returns (Vec<CallsFromFunctions>, Vec<CallsFromTests>)
    pub async fn get_function_calls(
        &self,
        code: &str,
        file: &str,
        graph: &Graph,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<(Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>)> {
        trace!("get_function_calls");
        let tree = self.lang.parse(&code, &NodeType::Function)?;
        // get each function
        let qo1 = self.q(&self.lang.function_definition_query(), &NodeType::Function);
        let mut cursor = QueryCursor::new();
        let mut matches = cursor.matches(&qo1, tree.root_node(), code.as_bytes());
        // calls from functions, calls from tests
        let mut res = (Vec::new(), Vec::new(), Vec::new());
        // get each function call within that function
        while let Some(m) = matches.next() {
            // FIXME can we only pass in the node code here? Need to sum line nums
            self.add_calls_for_function(&mut res, &m, &qo1, code, file, graph, lsp_tx)
                .await?;
        }
        Ok(res)
    }
    pub async fn add_calls_for_function<'a, 'b>(
        &self,
        res: &mut (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>), // funcs, tests, integration tests
        m: &QueryMatch<'a, 'b>,
        q: &Query,
        code: &str,
        file: &str,
        graph: &Graph,
        lsp_tx: &Option<CmdSender>,
    ) -> Result<()> {
        trace!("add_calls_for_function");
        let mut caller_name = "".to_string();
        Self::loop_captures(q, &m, code, |body, node, o| {
            if o == FUNCTION_NAME {
                caller_name = body;
            } else if o == FUNCTION_DEFINITION {
                // NOTE this should always be the last one
                let q2 = self.q(&self.lang.function_call_query(), &NodeType::Function);
                let calls = self.collect_calls_in_function(
                    &q2,
                    code,
                    file,
                    node,
                    &caller_name,
                    graph,
                    lsp_tx,
                )?;
                self.add_calls_inside(res, &caller_name, file, calls);
                if self.lang.is_test(&caller_name, file) {
                    let int_calls = self.collect_integration_test_calls(
                        code,
                        file,
                        node,
                        &caller_name,
                        graph,
                        lsp_tx,
                    )?;
                    res.2.extend(int_calls);
                }
            }
            Ok(())
        })?;
        Ok(())
    }
    fn add_calls_inside(
        &self,
        res: &mut (Vec<FunctionCall>, Vec<FunctionCall>, Vec<Edge>),
        caller_name: &str,
        caller_file: &str,
        calls: Vec<FunctionCall>,
    ) {
        if self.lang.is_test(&caller_name, caller_file) {
            res.1.extend_from_slice(&calls);
        } else {
            res.0.extend_from_slice(&calls);
        }
    }
}

impl Lang {
    pub fn from_language(l: Language) -> Lang {
        match l {
            Language::Rust => Lang::new_rust(),
            Language::Python => Lang::new_python(),
            Language::Go => Lang::new_go(),
            Language::Typescript => Lang::new_react_ts(),
            Language::Ruby => Lang::new_ruby(),
            Language::Bash => unimplemented!(),
            Language::Toml => unimplemented!(),
        }
    }
}
impl FromStr for Lang {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let ss = Language::from_str(s)?;
        Ok(Lang::from_language(ss))
    }
}

pub fn vecy(args: &[&str]) -> Vec<String> {
    args.iter().map(|s| s.to_string()).collect()
}

pub fn query_to_ident(query: Query, node: TreeNode, code: &str) -> Result<Option<String>> {
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&query, node, code.as_bytes());
    let first = matches.next();
    if first.is_none() {
        return Ok(None);
    }
    let mut cs = first.unwrap().captures.iter().into_streaming_iter_ref();
    let name_node = cs.next().context("no name_node")?;
    let name = name_node.node.utf8_text(code.as_bytes())?;
    Ok(Some(name.to_string()))
}
