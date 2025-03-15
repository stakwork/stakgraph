use std::collections::BTreeMap;

use super::super::*;
use super::consts::*;
use crate::builder::get_page_name;
use crate::lang::parse::trim_quotes;
use crate::lang::queries::rails_routes;
use anyhow::{Context, Result};
use std::path::Path;
use tracing::debug;
use tree_sitter::{Language, Parser, Query, Tree};

pub struct Ruby(Language);

impl Ruby {
    pub fn new() -> Self {
        Ruby(tree_sitter_ruby::LANGUAGE.into())
    }
}

impl Stack for Ruby {
    fn q(&self, q: &str, _nt: &NodeType) -> Query {
        Query::new(&self.0, q).unwrap()
    }
    fn parse(&self, code: &str, _nt: &NodeType) -> Result<Tree> {
        let mut parser = Parser::new();
        parser.set_language(&self.0)?;
        Ok(parser.parse(code, None).context("failed to parse")?)
    }
    fn lib_query(&self) -> Option<String> {
        Some(format!(
            r#"(call
    method: (identifier) @gem (#eq? @gem "gem")
    arguments: (argument_list
        . (string) @{LIBRARY_NAME}
        (string)? @{LIBRARY_VERSION}
    )
) @{LIBRARY}"#
        ))
    }
    fn class_definition_query(&self) -> String {
        format!(
            r#"[
    (class
        name: [
            (constant)
            (scope_resolution)
        ] @{CLASS_NAME}
        (superclass
            (constant) @{CLASS_PARENT}
        )?
        (body_statement
            (call
                method: (identifier) @call (#eq? @call "include")
                arguments: (argument_list) @{INCLUDED_MODULES}
            )
        )?
    )
    (module
        name: [
            (constant)
            (scope_resolution)
        ] @{CLASS_NAME}
    )
] @{CLASS_DEFINITION}"#
        )
    }
    fn function_definition_query(&self) -> String {
        format!(
            "[
    (method
        name: (identifier) @{FUNCTION_NAME}
        parameters: (method_parameters)? @{ARGUMENTS}
    )
    (singleton_method
        name: (identifier) @{FUNCTION_NAME}
        parameters: (method_parameters)? @{ARGUMENTS}
    )
] @{FUNCTION_DEFINITION}"
        )
    }
    fn function_call_query(&self) -> String {
        format!(
            "(call
    receiver: [
        (identifier)
        (constant)
        (call)
    ] @{OPERAND}
    method: (identifier) @{FUNCTION_NAME}
    arguments: (argument_list) @{ARGUMENTS}
) @{FUNCTION_CALL}"
        )
    }
    fn endpoint_finders(&self) -> Vec<String> {
        super::rails_routes::ruby_endpoint_finders_func()
    }
    fn endpoint_path_filter(&self) -> Option<String> {
        Some("routes.rb".to_string())
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
        while parent.is_some() && parent.unwrap().kind().to_string() != "class" {
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
    fn identifier_query(&self) -> String {
        format!("name: [(constant) (scope_resolution)] @identifier")
    }
    fn data_model_query(&self) -> Option<String> {
        Some(format!(
            r#"(call
    receiver: [
        (element_reference
            object: (scope_resolution
                scope: (constant) @scope (#eq? @scope "ActiveRecord")
                name: (constant) @name (#eq? @name "Schema")
            )
        )
        (scope_resolution
            scope: (constant) @scope (#eq? @scope "ActiveRecord")
            name: (constant) @name (#eq? @name "Schema")
        )
    ]
    block: (do_block
        body: (body_statement
            (call
                method: (identifier) @create (#eq? @create "create_table")
                arguments: (argument_list
                    (string) @{STRUCT_NAME}
                )
            ) @{STRUCT}
        )
    )
)"#
        ))
    }
    fn data_model_path_filter(&self) -> Option<String> {
        Some("db/schema.rb".to_string())
    }
    fn use_data_model_within_finder(&self) -> bool {
        true
    }
    fn data_model_within_finder(&self, data_model: &NodeData, graph: &Graph) -> Vec<Edge> {
        // file: app/controllers/api/advisor_groups_controller.rb
        let mut models = Vec::new();
        let funcs = graph.find_funcs_by(|f| is_controller(&f, &data_model.name));
        for func in funcs {
            models.push(Edge::contains(
                NodeType::Function,
                &func,
                NodeType::DataModel,
                data_model,
            ));
        }
        // without: Returning Graph with 12726 nodes and 13283 edges
        // if edge:Handler with source.node_data.name == name, then the target -> Contains this data model
        // "advisor_groups"
        models
    }
    fn is_test(&self, _func_name: &str, func_file: &str) -> bool {
        self.is_test_file(func_file)
    }
    fn is_test_file(&self, filename: &str) -> bool {
        filename.ends_with("_spec.rb")
    }
    fn e2e_test_id_finder_string(&self) -> Option<String> {
        Some("get_by_test_id".to_string())
    }
    fn use_handler_finder(&self) -> bool {
        true
    }
    fn handler_finder(
        &self,
        endpoint: NodeData,
        graph: &Graph,
        params: HandlerParams,
    ) -> Vec<(NodeData, Option<Edge>)> {
        if endpoint.meta.get("handler").is_none() {
            return Vec::new();
        }
        let handler_string = endpoint.meta.get("handler").unwrap();
        // tracing::info!("handler_finder: {} {:?}", handler_string, params);
        let mut explicit_path = false;
        // intermediate nodes (src/target)
        let mut inter = Vec::new();
        // let mut targets = Vec::new();
        if let Some(item) = &params.item {
            if let Some(nd) = graph.find_func_by(|nd: &NodeData| {
                is_controller(nd, handler_string) && nd.name == item.name
            }) {
                inter.push((endpoint, nd));
            }
        } else if handler_string.contains("#") {
            // put 'request_center/:id', to: 'request_center#update'
            let arr = handler_string.split("#").collect::<Vec<&str>>();
            if arr.len() != 2 {
                return Vec::new();
            }
            let controller = arr[0];
            let name = arr[1];
            // debug!("controller: {}, name: {}", controller, name);
            if let Some(nd) =
                graph.find_func_by(|nd: &NodeData| nd.name == name && is_controller(nd, controller))
            {
                inter.push((endpoint, nd));
                explicit_path = true;
            }
        } else {
            // https://guides.rubyonrails.org/routing.html  section 2.2 CRUD, Verbs, and Actions
            let ror_actions = vec![
                "index", "show", "new", "create", "edit", "update", "destroy",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            let verb_mapping = vec![
                ("GET", "index"),
                ("GET", "show"),
                ("GET", "new"),
                ("POST", "create"),
                ("GET", "edit"),
                ("PUT", "update"),
                ("DELETE", "destroy"),
            ];
            let mut verb_map = BTreeMap::new();
            for (verb, action) in verb_mapping {
                verb_map.insert(action.to_string(), verb.to_string());
            }
            let actions = match &params.actions_array {
                Some(aa) => {
                    let aaa = trim_array_string(aa);
                    // split on commas, and trim_quotes
                    aaa.split(",")
                        .map(|s| trim_quotes(s).to_string())
                        .collect::<Vec<String>>()
                }
                None => ror_actions,
            };
            // resources :request_center
            let controllers =
                graph.find_funcs_by(|nd: &NodeData| is_controller(nd, handler_string));
            debug!(
                "ror endpoint controllers for {}: {:?}",
                handler_string,
                controllers.len()
            );
            for nd in controllers {
                debug!("checking controller: {}", nd.name);
                if actions.contains(&nd.name) {
                    debug!("===> found action: {}", nd.name);
                    let mut endp_ = endpoint.clone();
                    endp_.add_action(&nd.name);
                    if let Some(verb) = verb_map.get(&nd.name) {
                        endp_.add_verb(verb);
                    }
                    inter.push((endp_, nd));
                }
            }
        }

        let ret = inter
            .iter()
            .map(|(src, target)| {
                let mut src = src.clone();
                if !explicit_path {
                    if let Some(pathy) = rails_routes::generate_endpoint_path(&src, &params) {
                        src.name = pathy;
                    }
                }
                let edge = Edge::handler(&src, &target);
                (src.clone(), Some(edge))
            })
            .collect::<Vec<(NodeData, Option<Edge>)>>();

        ret
    }
    fn find_endpoint_parents(
        &self,
        node: TreeNode,
        code: &str,
        _file: &str,
        _graph: &Graph,
    ) -> Result<Vec<HandlerItem>> {
        let mut parents = Vec::new();
        let mut parent = node.parent();

        while parent.is_some() {
            let parent_node = parent.unwrap();
            if parent_node.kind().to_string() == "call" {
                // Check if this is a namespace or resources call
                if let Some(method_node) = parent_node.child_by_field_name("method") {
                    if method_node.kind().to_string() == "identifier" {
                        let method_name = method_node.utf8_text(code.as_bytes()).unwrap_or("");
                        if method_name == "namespace" || method_name == "resources" {
                            // Get the first argument which should be the route name
                            if let Some(args_node) = parent_node.child_by_field_name("arguments") {
                                if let Some(first_arg) = args_node.named_child(0) {
                                    let route_name =
                                        first_arg.utf8_text(code.as_bytes()).unwrap_or("");
                                    let item_type = if method_name == "namespace" {
                                        HandlerItemType::Namespace
                                    } else {
                                        HandlerItemType::ResourceMember
                                    };
                                    // Create HandlerItem for this parent route
                                    parents.push(HandlerItem {
                                        name: trim_quotes(route_name).to_string(),
                                        item_type,
                                    });
                                }
                            }
                        }
                    }
                }
            }
            parent = parent_node.parent();
        }

        // Reverse the order so that outermost parents come first
        parents.reverse();
        Ok(parents)
    }
    fn integration_test_query(&self) -> Option<String> {
        Some(format!(
            r#"(call
    method: (identifier) @describe (#eq? @describe "describe")
    arguments: [
        (argument_list
            [
                (constant)
                (scope_resolution)
            ] @{HANDLER}
        )
        (argument_list
            (string) @{E2E_TEST_NAME}
            (pair) @js-true (#eq? @js-true "js: true")
        )
    ]
) @{INTEGRATION_TEST}"#
        ))
    }
    fn use_integration_test_finder(&self) -> bool {
        true
    }
    fn integration_test_edge_finder(
        &self,
        nd: &NodeData,
        graph: &Graph,
        tt: NodeType,
    ) -> Option<Edge> {
        let cla = graph.find_class_by(|clnd| clnd.name == nd.name);
        if let Some(cl) = cla {
            let meta = CallsMeta {
                call_start: nd.start,
                call_end: nd.end,
                operand: None,
            };
            Some(Edge::calls(tt, nd, NodeType::Class, &cl, meta))
        } else {
            None
        }
    }
    fn use_extra_page_finder(&self) -> bool {
        true
    }
    fn is_extra_page(&self, file_name: &str) -> bool {
        let is_good_ext = file_name.ends_with(".erb") || file_name.ends_with(".haml");
        let pagename = get_page_name(file_name);
        if pagename.is_none() {
            return false;
        }
        let is_underscore = pagename.as_ref().unwrap().starts_with("_");
        let is_view = file_name.contains("/views/");
        is_view && is_good_ext && !is_underscore
    }
    fn extra_page_finder(&self, file_path: &str, graph: &Graph) -> Option<Edge> {
        let pagename = get_page_name(file_path);
        if pagename.is_none() {
            return None;
        }
        let pagename = pagename.unwrap();
        let page = NodeData::name_file(&pagename, file_path);
        // get the handler name
        let p = std::path::Path::new(file_path);
        let func_name = remove_all_extensions(p);
        let controller_name = p.parent()?.file_name()?.to_str()?;
        // println!("func_name: {}, controller_name: {}", func_name, controller_name);
        let handler =
            graph.find_func_by(|nd| nd.name == func_name && is_controller(nd, controller_name));
        if let Some(handler) = handler {
            Some(Edge::renders(&page, &handler))
        } else {
            None
        }
    }
}

fn remove_all_extensions(path: &Path) -> String {
    let mut stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string();

    while let Some(s) = Path::new(&stem).file_stem() {
        if let Some(s_str) = s.to_str() {
            if s_str == stem {
                break;
            }
            stem = s_str.to_string();
        } else {
            break;
        }
    }

    stem
}

fn trim_array_string(s: &str) -> String {
    s.trim_start_matches("%i")
        .trim_start_matches("[")
        .trim_end_matches("]")
        .to_string()
}

fn is_controller(nd: &NodeData, controller: &str) -> bool {
    nd.file.ends_with(&format!("{}_controller.rb", controller))
}
