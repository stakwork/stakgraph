use super::super::*;
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Erb(Language);

impl Default for Erb {
    fn default() -> Self {
        Self::new()
    }
}

impl Erb {
    pub fn new() -> Self {
        Erb(tree_sitter_embedded_template::LANGUAGE.into())
    }
}

impl Stack for Erb {
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        Query::new(&self.0, q).unwrap()
    }
    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.0)?;
        parser.parse(code, None).context("failed to parse")
    }
    fn class_definition_query(&self) -> String {
        "".to_string()
    }
    fn function_definition_query(&self) -> String {
        "".to_string()
    }
    fn function_call_query(&self) -> String {
        "".to_string()
    }
}
