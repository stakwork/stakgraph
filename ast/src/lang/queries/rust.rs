use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use toml::Toml;
use tree_sitter::{Language, Parser, Query, Tree};
pub struct Rust(Language);

impl Rust {
    pub fn new() -> Self {
        Rust(tree_sitter_rust::LANGUAGE.into())
    }
}

impl Stack for Rust {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            Query::new(&tree_sitter_toml_ng::LANGUAGE.into(), q).unwrap()
        } else {
            Query::new(&self.0, q).unwrap()
        }
    }

    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();

        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_toml_ng::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }

        Ok(parser.parse(code, None).context("failed to parse")?)
    }

    fn lib_query(&self) -> Option<String> {
        Toml::new().lib_query()
    }

    fn imports_query(&self) -> Option<String> {
        Some(
            r#"
            (use_declaration) @import
            (use_list) @import_list
            (scoped_use_list) @scoped_import
            (extern_crate_declaration) @extern_crate
        "#
            .to_string(),
        )
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
            (struct_item
                name: (type_identifier) @{STRUCT_NAME}
                body: (field_declaration_list)? @struct.body) @{STRUCT}
                
            (trait_item
                name: (type_identifier) @{TRAIT_NAME}
                body: (declaration_list)? @trait.body) @{TRAIT}
                
            (impl_item
                trait: (type_identifier)? @impl.trait
                type: (type_identifier) @{PARENT_TYPE}
                body: (declaration_list)? @impl.body) @impl
            "#
        )
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"
            (function_item
              name: (identifier) @{FUNCTION_NAME}
              parameters: (parameters) @{ARGUMENTS}
              return_type: (type_identifier)? @{RETURN_TYPES}
              body: (block)? @function.body) @{FUNCTION_DEFINITION}
              
            (function_signature_item
              name: (identifier) @{FUNCTION_NAME}
              parameters: (parameters) @{ARGUMENTS}
              return_type: (type_identifier)? @{RETURN_TYPES}) @{FUNCTION_DEFINITION}
            
            (impl_item
              type: (_) @{PARENT_TYPE}
              body: (declaration_list
                (function_item
                  name: (identifier) @{FUNCTION_NAME}
                  parameters: (parameters) @{ARGUMENTS}
                  return_type: (type_identifier)? @{RETURN_TYPES}
                  body: (block)? @method.body) @method)) @impl
            "#
        )
    }
    fn function_call_query(&self) -> String {
        format!(
            r#"
            (call_expression
              function: (identifier) @{FUNCTION_NAME}
              arguments: (argument_list) @{ARGUMENTS}) @{FUNCTION_CALL}
            "#
        )
    }

    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            // Pattern 1: Router methods with handler - router.route("/path", get(handler))
            r#"(call_expression
  function: (field_expression
    (field_identifier) @route_method (#eq? @route_method "route")
  )
  (argument_list
    (string_literal) @route
    (call_expression
      function: (identifier) @verb (#match? @verb "^(get|post|put|delete|patch|options|head)$")
      (argument_list
        (identifier) @handler
      )
    )
  )
)"#
            .to_string(),
            // Pattern 2: Direct HTTP verb methods - router.get("/path", handler)
            r#"(call_expression
  function: (field_expression
    (field_identifier) @verb (#match? @verb "^(get|post|put|delete|patch|options|head)$")
  )
  (argument_list
    (string_literal) @route
    (identifier) @handler
  )
)"#
            .to_string(),
            // Pattern 3: Rocket-style attribute macros
            r#"(attribute_item
  (meta_item
    (identifier) @verb (#match? @verb "^(get|post|put|delete|patch|options|head)$")
    (token_tree
      (string_literal) @route
    )
  )
)
(function_item
  name: (identifier) @handler
)"#
            .to_string(),
        ]
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (struct_item
                    name: (type_identifier) @{STRUCT_NAME}
                    body: (field_declaration_list)? @struct.body) @{STRUCT}
                    
                (trait_item
                    name: (type_identifier) @{TRAIT_NAME}
                    body: (declaration_list)? @trait.body) @{TRAIT}
                    
                (impl_item
                    trait: (type_identifier)? @impl.trait
                    type: (type_identifier) @{PARENT_TYPE}
                    body: (declaration_list)? @impl.body) @impl
                "#
        ))
    }
    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"
                (struct_item
                    name: (type_identifier) @{STRUCT_NAME}
                    body: (field_declaration_list)? @struct.body) @{STRUCT}
                    
                (trait_item
                    name: (type_identifier) @{TRAIT_NAME}
                    body: (declaration_list)? @trait.body) @{TRAIT}
                    
                (impl_item
                    trait: (type_identifier)? @impl.trait
                    type: (type_identifier) @{PARENT_TYPE}
                    body: (declaration_list)? @impl.body) @impl
                "#
        ))
    }
}
