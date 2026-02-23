use super::super::*;
use super::consts::*;
use shared::error::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Svelte(Language);

impl Default for Svelte {
    fn default() -> Self {
        Self::new()
    }
}

impl Svelte {
    pub fn new() -> Self {
        Svelte(tree_sitter_svelte_ng::LANGUAGE.into())
    }
}

impl Stack for Svelte {
    fn should_skip_function_call(&self, called: &str, operand: &Option<String>) -> bool {
        super::skips::svelte::should_skip(called, operand)
    }
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            Query::new(&tree_sitter_svelte_ng::LANGUAGE.into(), q).unwrap()
        } else {
            Query::new(&self.0, q).unwrap()
        }
    }

    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_svelte_ng::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }
        parser.parse(code, None).context("failed to parse")
    }

    fn imports_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (document
                (_) @{IMPORTS}
            )
        "#
        ))
    }

    fn class_definition_query(&self) -> String {
        format!(
            r#"
            (script_element
                (_) @{CLASS_NAME}
            )
            "#
        )
    }

    fn function_definition_query(&self) -> String {
        format!(
            r#"
            (
                attribute
                (expression
                    (_) @{FUNCTION_NAME}
                ) @{FUNCTION_DEFINITION}
            )
            "#
        )
    }
    fn comment_query(&self) -> Option<String> {
        Some(format!(r#"(comment) @{FUNCTION_COMMENT}"#))
    }

    fn function_call_query(&self) -> String {
        r#"
            (expression
                (_) @args
            ) @FUNCTION_CALL
            "#.to_string()
    }

    fn find_function_parent(
        &self,
        node: TreeNode,
        code: &str,
        file: &str,
        func_name: &str,
        _callback: &dyn Fn(&str) -> Option<(NodeData, NodeType)>,
        _parent_type: Option<&str>,
    ) -> Result<Option<Operand>> {
        let mut parent = node.parent();
        while parent.is_some() {
            if parent.unwrap().kind() == "class_declaration" {
                // found it!
                break;
            }
            parent = parent.unwrap().parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q("(type_identifier) @class_name", &NodeType::Class);
                query_to_ident(query, p, code)?.map(|parent_name| Operand {
                        source: NodeKeys::new(&parent_name, file, p.start_position().row),
                        target: NodeKeys::new(func_name, file, node.start_position().row),
                        source_type: NodeType::Class,
                    })
            }
            None => None,
        };
        Ok(parent_of)
    }

    fn request_finder(&self) -> Option<String> {
        Some(format!(
            r#"
            (_
                (_) @{ENDPOINT}
                (#match? @{ENDPOINT} "fetch|get|post|put|delete")
            ) @{REQUEST_CALL}
            "#
        ))
    }

    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"
            (document
                (_
                    (_) + @{STRUCT_NAME}
                )
            ) @{STRUCT}
            "#
        ))
    }
    fn identifier_query(&self) -> String {
        r#"(tag_name) @identifier"#.to_string()
    }

    fn data_model_within_query(&self) -> Option<String> {
        Some(format!(
            r#"
            [
                    (_) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                (expression
                     (_) @{STRUCT_NAME} (#match? @{STRUCT_NAME} "^[A-Z].*")
                )
            ]
            "#
        ))
    }

    fn is_test(&self, func_name: &str, _func_file: &str, _func_body: &str) -> bool {
        func_name.starts_with("test")
    }
}
