use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Swift(Language);

impl Swift {
    pub fn new() -> Self {
        Swift(tree_sitter_swift::LANGUAGE.into())
    }
}

impl Stack for Swift {
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        Query::new(&self.0, q).unwrap()
    }
    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();

        parser.set_language(&self.0)?;
        Ok(parser.parse(code, None).context("failed to parse")?)
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (import_declaration
                (identifier) @{IMPORTS}
            )
            "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
            (class_declaration
                (type_identifier) @{CLASS_NAME}
            ) @{CLASS_DEFINITION}
            "#
        )
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"
        (function_declaration
            (simple_identifier) @{FUNCTION_NAME}
        ) @{FUNCTION_DEFINITION}
        "#
        )
    }

    fn function_call_query(&self) -> String {
        format!(
            r#"
            (call_expression
                 (simple_identifier) @{ARGUMENTS}
            ) @{FUNCTION_CALL}
            "#
        )
    }

    fn find_function_parent(
        &self,
        node: TreeNode,
        code: &str,
        file: &str,
        func_name: &str,
        _callback: &dyn Fn(&str) -> Option<NodeData>,
        _parent_type: Option<&str>,
    ) -> Result<Option<Operand>> {
        let mut parent = node.parent();
        while parent.is_some() {
            if parent.unwrap().kind().to_string() == "class_declaration" {
                // found it!
                break;
            }
            parent = parent.unwrap().parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q("(type_identifier) @class-name", &NodeType::Class);
                match query_to_ident(query, p, code)? {
                    Some(parent_name) => Some(Operand {
                        source: NodeKeys::new(&parent_name, file, p.start_position().row),
                        target: NodeKeys::new(func_name, file, node.start_position().row),
                    }),
                    None => None,
                }
            }
            None => None,
        };
        Ok(parent_of)
    }

    fn request_finder(&self) -> Option<String> {
        Some(format!(
            r#"
            (call_expression
                (simple_identifier) @{REQUEST_CALL} (#match? @{REQUEST_CALL} "^createRequest$")
            ) @{ROUTE}
        "#
        ))
    }
    fn add_endpoint_verb(&self, inst: &mut NodeData, _call: &Option<String>) {
        if inst.meta.get("verb").is_none() {
            if inst.body.contains("method: \"GET\"") || inst.body.contains("bodyParams: nil") {
                inst.add_verb("GET");
            } else if inst.body.contains("method: \"POST\"") {
                inst.add_verb("POST");
            } else if inst.body.contains("method: \"PUT\"") {
                inst.add_verb("PUT");
            } else if inst.body.contains("method: \"DELETE\"") {
                inst.add_verb("DELETE");
            }

            if inst.meta.get("verb").is_none() {
                inst.add_verb("GET"); // Default
            }
        }
        if inst.name.is_empty() {
            let url_start = inst.body.find("url:");
            if let Some(start_pos) = url_start {
                if let Some(quote_start) = inst.body[start_pos..].find("\"") {
                    let start_idx = start_pos + quote_start + 1;
                    if let Some(quote_end) = inst.body[start_idx..].find("\"") {
                        let url_section = &inst.body[start_idx..start_idx + quote_end];
                        if let Some(path_start) = url_section.rfind("/") {
                            let path = &url_section[path_start..];
                            if !path.is_empty() {
                                inst.name = path.to_string();
                            }
                        }
                    }
                }
            }
        }
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (class_declaration
                (type_identifier) @{STRUCT_NAME}
                (_)*
            ) @{STRUCT}
        "#
        ))
    }

    fn data_model_path_filter(&self) -> Option<String> {
        Some("CoreData".to_string())
    }

    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"[
                (identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")

                (call_expression
                    (simple_identifier) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                )

                ]@{STRUCT}
            "#
        ))
    }

    fn is_test(&self, func_name: &str, _func_file: &str) -> bool {
        func_name.starts_with("test")
    }
}
