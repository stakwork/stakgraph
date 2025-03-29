use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Prisma(Language);

impl Prisma {
    pub fn new() -> Self {
        Prisma(tree_sitter_prisma_io::language())
    }
}

impl Stack for Prisma {
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        Query::new(&self.0, q).unwrap()
    }

    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.0)?;
        Ok(parser
            .parse(code, None)
            .context("Failed to parse prisma schema")?)
    }
    fn class_definition_query(&self) -> String {
        "".to_string()
    }
    fn function_call_query(&self) -> String {
        "".to_string()
    }
    fn function_definition_query(&self) -> String {
        "".to_string()
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (model_declaration
                (identifier) @{STRUCT_NAME}
            )@{STRUCT}
            "#
        ))
    }
    fn data_model_path_filter(&self) -> Option<String> {
        Some("prisma/schema.prisma".to_string())
    }
}
