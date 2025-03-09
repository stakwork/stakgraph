use super::super::*;
use super::consts::*;
use anyhow::{Context, Result};
use tree_sitter::{Language, Node as TreeNode, Parser, Query, Tree};

pub struct Python(Language);

impl Python {
    pub fn new() -> Self {
        Python(tree_sitter_python::LANGUAGE.into())
    }
}

impl Stack for Python {
    fn q(&self, q: &str, nt: &NodeType) -> Query {
        if matches!(nt, NodeType::Library) {
            Query::new(&tree_sitter_bash::LANGUAGE.into(), q).unwrap()
        } else {
            Query::new(&self.0, q).unwrap()
        }
    }
    fn parse(&self, code: &str, nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        if matches!(nt, NodeType::Library) {
            parser.set_language(&tree_sitter_bash::LANGUAGE.into())?;
        } else {
            parser.set_language(&self.0)?;
        }
        Ok(parser.parse(code, None).context("failed to parse")?)
    }
    fn lib_query(&self) -> Option<String> {
        Some(format!(
            "(variable_assignment
	name: (variable_name) @{LIBRARY_NAME}
    value: (word) @{LIBRARY_VERSION}
) @{LIBRARY}"
        ))
    }
    fn imports_query(&self) -> Option<String> {
        Some(format!(
            "(module
    [(import_statement)+ (import_from_statement)+] @{IMPORTS}
)"
        ))
    }
    fn class_definition_query(&self) -> String {
        format!(
            "(class_definition
    name: (identifier) @{CLASS_NAME}
) @{CLASS_DEFINITION}"
        )
    }
    // this captures both
    fn function_definition_query(&self) -> String {
        format!(
            "(function_definition
    name: (identifier) @{FUNCTION_NAME}
    parameters: (parameters) @{ARGUMENTS}
) @{FUNCTION_DEFINITION}"
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
        while parent.is_some() && parent.unwrap().kind().to_string() != "class_definition" {
            parent = parent.unwrap().parent();
        }
        let parent_of = match parent {
            Some(p) => {
                let query = self.q(&self.identifier_query(), &NodeType::Class);
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
    fn function_call_query(&self) -> String {
        format!(
            "(call
    function: [
        (identifier)
        (attribute object: (identifier) @{OPERAND}
            attribute: (identifier) @{FUNCTION_NAME}
        )
    ] @{FUNCTION_NAME}
) @{FUNCTION_CALL}"
        )
    }


    fn endpoint_finders(&self) -> Vec<String> {
        vec![format!(
          r#"(decorated_definition
    (decorator
        (call
            function: (attribute
                attribute: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$")
            )
            arguments: (argument_list
                (string) @{ENDPOINT}
            )
       )
    ) @{ROUTE}
    definition: (function_definition
        name: (identifier) @{HANDLER}
    )
)"#
        )]
    }

 fn data_model_query(&self) -> Option<String> {
        Some(format!(
            "(class_definition
    name: (identifier) @{STRUCT_NAME}
) @{STRUCT}"
        ))
    }

     fn is_test(&self, func_name: &str, _func_file: &str) -> bool {
        func_name.starts_with("test_")
    }


}
