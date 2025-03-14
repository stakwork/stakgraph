use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Toml(Language);

impl Toml {
    pub fn new() -> Self {
        Toml(tree_sitter_toml_ng::LANGUAGE.into())
    }
}

impl Stack for Toml {
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        Query::new(&self.0, q).unwrap()
    }

    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.0)?;
        Ok(parser.parse(code, None).context("failed to parse TOML")?)
    }

    fn lib_query(&self) -> Option<String> {
        Some(format!(
            r#"(document
          (table 
            (bare_key) @section (#eq? @section "dependencies")
            (pair 
              (bare_key) @{LIBRARY_NAME}
              [
                ; Simple version string: package = "1.0.0"
                (string) @{LIBRARY_VERSION}
                
                ; Table with version: package = {{ version = "1.0.0", ... }}
                (inline_table
                  (pair
                    (bare_key) @version_key (#eq? @version_key "version")
                    (string) @{LIBRARY_VERSION}
                  )
                )
              ]
            )*
          )
        ) @{LIBRARY}"#
        ))
    }
    fn class_definition_query(&self) -> String {
        format!(
            r#"(table
              (pair
                (bare_key) @class_name
                (inline_table
                  (pair
                    (bare_key) @class_key (#eq? @class_key "type")
                    (string) @class_type
                  )
                )
              )
            ) @{CLASS_DEFINITION}"#
        )
    }
    fn function_definition_query(&self) -> String {
        format!(
            r#"(table
              (pair
                (bare_key) @function_name
                (inline_table
                  (pair
                    (bare_key) @function_key (#eq? @function_key "type")
                    (string) @function_type
                  )
                )
              )
            ) @{FUNCTION_DEFINITION}"#
        )
    }
    fn function_call_query(&self) -> String {
        format!(
            r#"(pair
              (bare_key) @caller
              (array 
                (string) @function_name
              )
            ) @{FUNCTION_CALL}"#
        )
    }
}
