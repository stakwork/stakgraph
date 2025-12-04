use super::consts::*;
use inflection_rs::inflection;

pub fn ruby_endpoint_finders_func() -> Vec<String> {
    // root to: 'home#index'
    let root_finder = format!(
        r#"(call
            method: (identifier) @root (#eq? @root "root")
            arguments: (argument_list
                (pair
                    key: (hash_key_symbol) @to (#eq? @to "to")
                    value: (string) @{HANDLER}
                )
            )
        ) @{ROUTE}"#
    );
    // put 'request_center/:id', to: 'request_center#update'
    let verb_finder = format!(
        r#"(call
            method: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$")
            arguments: (argument_list
                (string) @{ENDPOINT}
                (pair
                    key: (hash_key_symbol) @to (#eq? @to "to")
                    value: (string) @{HANDLER}
                )
                (pair
                    key: (hash_key_symbol) @as (#eq? @as "as")
                    value: (simple_symbol) @{ENDPOINT_ALIAS}
                )?
                (pair
                    key: (hash_key_symbol) @action (#eq? @action "action")
                    value: (simple_symbol) @{ENDPOINT_ACTION}
                )?
            )
        ) @{ROUTE}"#
    );
    // resource :dashboard, only: [:show, :update] (singular)
    let resource_finder = format!(
        r#"(call
            method: (identifier) @resource (#eq? @resource "resource")
                arguments: (argument_list
                    (simple_symbol) @{SINGULAR_RESOURCE}
                (pair
                    key: (hash_key_symbol) @only (#eq? @only "only")
                    value: (array) @{HANDLER_ACTIONS_ARRAY}
                )? 
            )
        ) @{ROUTE}"#
    );
    // resources :candidate_notes, only: %i[create update destroy]
    // !block after arguments: () will make this mutually exclusive with the collection_finder query
    let resources_finder = format!(
        r#"(call
            method: (identifier) @resources (#eq? @resources "resources")
                arguments: (argument_list
                    (simple_symbol) @{HANDLER}
                (pair
                    key: (hash_key_symbol) @only (#eq? @only "only")
                    value: (array) @{HANDLER_ACTIONS_ARRAY}
                )? 
            )
        ) @{ROUTE}"#
    );
    // resources :profiles do
    //     collection do
    //         post :enrich_profile
    let collection_finder = format!(
        r#"(call
            method: (identifier) @resources (#eq? @resources "resources")
            arguments: (argument_list
                (simple_symbol) @{HANDLER}
            )
            block: (do_block
                body: (body_statement
                    (call
                        method: (identifier) @extra-routes (#eq? @extra-routes "collection")
                        block: (do_block
                            body: (body_statement
                                (call
                                    method: (identifier) @{ENDPOINT_VERB}
                                    arguments: (argument_list) @{COLLECTION_ITEM}
                                )
                            )
                        )
                    )
                )
            )
        ) @{ROUTE}"#
    );
    // resources :profiles do
    //     member do
    //         post :enrich_profile
    let member_finder = format!(
        r#"(call
            method: (identifier) @resources (#eq? @resources "resources")
            arguments: (argument_list
                (simple_symbol) @{HANDLER}
            )
            block: (do_block
                body: (body_statement
                    (call
                        method: (identifier) @extra-routes (#eq? @extra-routes "member")
                        block: (do_block
                            body: (body_statement
                                (call
                                    method: (identifier) @{ENDPOINT_VERB}
                                    arguments: (argument_list) @{MEMBER_ITEM}
                                )
                            )
                        )
                    )
                )
            )   
        ) @{ROUTE}"#
    );
    // resources :intro_requests do
    //     post :create_from_public_page
    let do_finder = format!(
        r#"(call
            method: (identifier) @resources (#eq? @resources "resources")
            arguments: (argument_list
                (simple_symbol) @{HANDLER}
                (pair
                    key: (hash_key_symbol) @only (#eq? @only "only")
                    value: (array) @{HANDLER_ACTIONS_ARRAY}
                )?
            )
            block: (do_block
                body: (body_statement
                    (call
                        method: (identifier) @{ENDPOINT_VERB} (#match? @{ENDPOINT_VERB} "^get$|^post$|^put$|^delete$")
                        arguments: (argument_list) @{RESOURCE_ITEM}
                    )
                )
            )
        ) @{ROUTE}"#
    );
    vec![
        root_finder,
        verb_finder,
        resource_finder,
        resources_finder,
        collection_finder,
        member_finder,
        do_finder,
    ]
}

use crate::lang::{HandlerItemType, HandlerParams, NodeData};

pub fn generate_endpoint_path(endpoint: &NodeData, params: &HandlerParams) -> Option<String> {
    if endpoint.meta.get("handler").is_none() {
        return None;
    }
    let handler = endpoint.meta.get("handler").unwrap();
    
    // Check if this is a root route (handler contains #index or similar but no resource prefix)
    if handler.contains("#") {
        let parts: Vec<&str> = handler.split("#").collect();
        if parts.len() == 2 && parts[0] == "home" && params.parents.is_empty() {
            return Some("/".to_string());
        }
        
        // For explicit routes with to: (e.g., get 'status', to: 'health#status')
        // Use the endpoint.name (the route path) not the handler controller name
        if !endpoint.name.is_empty() && endpoint.name != *handler {
            // This is an explicit route - use endpoint.name as the resource
            let mut path_parts = Vec::new();
            
            // Add parent namespaces/scopes
            for parent in &params.parents {
                match parent.item_type {
                    HandlerItemType::Namespace => {
                        path_parts.push(parent.name.clone());
                    }
                    _ => (),
                }
            }
            
            path_parts.push(endpoint.name.clone());
            return Some(format!("/{}", path_parts.join("/")));
        }
    }
    
    let mut path_parts = Vec::new();

    // Get the base resource name from handler
    let resource_name = if handler.contains("#") {
        handler.split("#").next().unwrap_or("").to_string()
    } else {
        handler.to_string()
    };
    
    // Check if this is a singular resource (no :id in paths)
    let is_singular = endpoint.meta.get("is_singular").is_some();

    // For collection/member routes, exclude the last parent if it matches our resource
    let parents_to_use = if let Some(item) = &params.item {
        match item.item_type {
            HandlerItemType::Collection | HandlerItemType::Member => {
                &params.parents[..params.parents.len().saturating_sub(1)]
            }
            _ => &params.parents[..],
        }
    } else {
        // For standard RESTful actions, also exclude the last parent if it matches our resource
        if !params.parents.is_empty() && params.parents.last().unwrap().name == resource_name {
            &params.parents[..params.parents.len() - 1]
        } else {
            &params.parents[..]
        }
    };

    // Add namespaces and parent resources
    for parent in parents_to_use {
        match parent.item_type {
            HandlerItemType::Namespace => {
                path_parts.push(parent.name.clone());
            }
            HandlerItemType::ResourceMember => {
                path_parts.push(parent.name.clone());
                path_parts.push(format!(":{}_id", to_singular(&parent.name)));
            }
            _ => (),
        }
    }

    // Handle member/collection routes
    if let Some(item) = &params.item {
        match item.item_type {
            HandlerItemType::Collection => {
                path_parts.push(resource_name);
                path_parts.push(item.name.clone());
            }
            HandlerItemType::Member => {
                path_parts.push(resource_name);
                path_parts.push(":id".to_string());
                path_parts.push(item.name.clone());
            }
            HandlerItemType::ResourceMember => {
                if !path_parts.contains(&resource_name) {
                    path_parts.push(resource_name.clone());
                    path_parts.push(format!(":{}_id", to_singular(&resource_name)));
                }
                path_parts.push(item.name.clone());
            }
            HandlerItemType::Namespace => (),
        }
        return Some(format!("/{}", path_parts.join("/")));
    }

    // For standard RESTful actions
    if !path_parts.contains(&resource_name) {
        path_parts.push(resource_name.clone());
    }

    match endpoint.meta.get("action") {
        Some(action) => match action.as_str() {
            "index" => (), // just the base resource path
            "new" => path_parts.push("new".to_string()),
            "create" => (), // just the base resource path
            "show" => {
                if !is_singular {
                    path_parts.push(":id".to_string());
                }
            }
            "edit" => {
                if !is_singular {
                    path_parts.push(":id".to_string());
                }
                path_parts.push("edit".to_string());
            }
            "update" => {
                if !is_singular {
                    path_parts.push(":id".to_string());
                }
            }
            "destroy" => {
                if !is_singular {
                    path_parts.push(":id".to_string());
                }
            }
            _ => (),
        },
        None => {
            if let Some(verb) = endpoint.meta.get("verb") {
                match verb.as_str() {
                    "GET" => {
                        if endpoint.name == "show" || endpoint.name == "edit" {
                            if !is_singular {
                                path_parts.push(":id".to_string());
                            }
                            if endpoint.name == "edit" {
                                path_parts.push("edit".to_string());
                            }
                        }
                    }
                    "PUT" | "PATCH" | "DELETE" => {
                        if !is_singular {
                            path_parts.push(":id".to_string());
                        }
                    }
                    _ => (),
                }
            }
        }
    }

    Some(format!("/{}", path_parts.join("/")))
}

// Helper function to convert plural to singular
fn to_singular(plural: &str) -> String {
    inflection::singularize(plural)
}
