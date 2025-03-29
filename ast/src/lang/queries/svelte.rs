
use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Svelte(Language);

impl Svelte {
    pub fn new() -> Self {
        Svelte(tree_sitter_svelte_ng::LANGUAGE.into())
    }
}


impl Stack for Svelte {
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
        Ok(parser.parse(code, None).context("failed to parse")?)
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
                    (_) @args
                ) @{FUNCTION_DEFINITION}
            )
            "#
        )
    }



    fn function_call_query(&self) -> String {
        format!(
            r#"
            (expression
                (_) @args
            ) @FUNCTION_CALL
            "#
        )
        }


    fn find_function_parent(
        &self,
        node: TreeNode,
        code: &str,
        file: &str,
        func_name: &str,
        _graph: &Graph,
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
                let query = self.q("(type_identifier) @class_name", &NodeType::Class);
                match query_to_ident(query, p, code)? {
                    Some(parent_name) => Some(Operand {
                        source: NodeKeys::new(&parent_name, file),
                        target: NodeKeys::new(func_name, file),
                    }),
                    None => None,
                }
            }
            None => None,
        };
        Ok(parent_of)
    }


    fn endpoint_finders(&self) -> Vec<String> {
        vec![
            format!(
                r#"
                (script_element
                    (_) @{ENDPOINT}
                ) @{ENDPOINT}
                (script_element
                    (_) @{ENDPOINT}
                    (#match? @endpoint "GET|POST|SET|PUT|DELETE")
                ) @{ENDPOINT}
                "#
            ),
        ]
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

    fn is_test(&self, func_name: &str, _func_file: &str) -> bool {
        func_name.starts_with("test")
    }
}
