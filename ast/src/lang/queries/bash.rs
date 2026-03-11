use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

#[allow(dead_code)]
pub struct Bash(Language);

impl Default for Bash {
    fn default() -> Self {
        Self::new()
    }
}

impl Bash {
    pub fn new() -> Self {
        Bash(tree_sitter_bash::LANGUAGE.into())
    }
}

impl Stack for Bash {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            match Query::new(&tree_sitter_bash::LANGUAGE.into(), q) {
                Ok(query) => query,
                Err(err) => panic!("Failed to compile Bash library query '{}': {}", q, err),
            }
        } else {
            match Query::new(&self.0, q) {
                Ok(query) => query,
                Err(err) => panic!("Failed to compile Bash query '{}': {}", q, err),
            }
        }
    }

    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_bash::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }
        parser.parse(code, None).context("failed to parse Bash")
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"(function_definition
                name: (word) @{CLASS_NAME}
                (#eq? @{CLASS_NAME} "__stakgraph_no_class__")
            ) @{CLASS_DEFINITION}"#
        )
    }

    fn identifier_query(&self) -> String {
        "(word) @identifier".to_string()
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"(function_definition
                name: (word) @{FUNCTION_NAME}
            ) @{FUNCTION_DEFINITION}"#
        )
    }

    fn function_call_query(&self) -> String {
        format!(
            r#"(command
                name: [
                    (command_name (word) @{FUNCTION_NAME})
                    (word) @{FUNCTION_NAME}
                ]
                argument: (word)* @{ARGUMENTS}
            ) @{FUNCTION_CALL}"#
        )
    }
}
