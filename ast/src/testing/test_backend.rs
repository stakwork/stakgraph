use anyhow::{Ok, Result};
use tracing::info;

use crate::lang::graph::{EdgeType, Graph, Node};
use crate::lang::queries::HandlerParams;
use crate::lang::{linker::normalize_backend_path, Lang};
use crate::repo::Repo;
use crate::utils::logger;

pub async fn test_backend(language: &Lang) -> Result<()> {
    logger();

    let language_kind = language.kind.clone();
    let language_in_repository = Lang::from_language(language_kind);

    let repo = Repo::new(
        &format!("src/testing/{}", language.kind.to_string()),
        language_in_repository,
        false,
        Vec::new(),
        Vec::new(),
    )?;

    let graph = repo.build_graph().await?;

    test_language_node(&graph, &language)?;
    test_package_file(&graph, &language)?;
    test_endpoints(&graph)?;
    test_person_datamodel(&graph)?;
    test_handler_functions(&graph, &language)?;

    Ok(())
}

fn test_language_node(graph: &Graph, language: &Lang) -> Result<()> {
    let language_nodes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Language(_)))
        .collect::<Vec<_>>();

    assert!(
        !language_nodes.is_empty(),
        "Language node not found for {}",
        language
    );

    let language_node = language_nodes[0].into_data();
    assert_eq!(
        language_node.name,
        language.kind.to_string(),
        "Language node name mismatch"
    );

    info!("✓ Found Language node for {}", language);

    Ok(())
}

fn test_package_file(graph: &Graph, language: &Lang) -> Result<()> {
    let package_file_name = language.kind.pkg_file();

    let file_nodes = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::File(_)))
        .collect::<Vec<_>>();

    let package_files: Vec<_> = file_nodes
        .iter()
        .filter(|n| {
            let file_data = n.into_data();
            file_data.name.contains(&package_file_name)
        })
        .collect();

    assert!(
        !package_files.is_empty(),
        "No package file found matching {}",
        package_file_name
    );

    info!("✓ Found package file {}", package_file_name);

    Ok(())
}

fn test_endpoints(graph: &Graph) -> Result<()> {
    let endpoints = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Endpoint(_)))
        .collect::<Vec<_>>();

    assert!(
        endpoints.len() >= 2,
        "Expected at least 2 endpoints, found {}",
        endpoints.len()
    );

    let get_endpoints: Vec<_> = endpoints
        .iter()
        .filter(|n| {
            let endpoint_data = n.into_data();
            let path = normalize_backend_path(&endpoint_data.name).unwrap_or_default();
            let is_get = endpoint_data.meta.get("verb").map_or(false, |v| v == "GET");
            is_get && path.contains("person")
        })
        .collect();

    assert!(!get_endpoints.is_empty(), "GET person endpoint not found");

    let post_endpoints: Vec<_> = endpoints
        .iter()
        .filter(|n| {
            let endpoint_data = n.into_data();
            let path = normalize_backend_path(&endpoint_data.name).unwrap_or_default();
            let is_post = endpoint_data
                .meta
                .get("verb")
                .map_or(false, |v| v == "POST");
            is_post && path.contains("person")
        })
        .collect();

    assert!(!post_endpoints.is_empty(), "POST person endpoint not found");

    info!("✓ Found required endpoints: GET person/:param and POST person");
    Ok(())
}

fn test_person_datamodel(graph: &Graph) -> Result<()> {
    let data_models = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::DataModel(_)))
        .collect::<Vec<_>>();

    let data_models: Vec<_> = data_models
        .iter()
        .filter(|n| {
            let data_model = n.into_data();
            data_model.name == "Person"
        })
        .collect();

    assert!(!data_models.is_empty(), "Person data model not found");

    info!("✓ Found Person data model");
    Ok(())
}

fn normalize_function_name(name: &str) -> String {
    name.replace('_', "").to_lowercase()
}

fn function_name_matches(function_name: &str, base_pattern: &str) -> bool {
    let normalized_function = normalize_function_name(function_name);
    let normalized_pattern = normalize_function_name(base_pattern);
    normalized_function.contains(&normalized_pattern)
}

fn test_handler_functions(graph: &Graph, language: &Lang) -> Result<()> {
    let endpoints = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::Endpoint(_)))
        .collect::<Vec<_>>();

    let person_model = graph
        .nodes
        .iter()
        .filter(|n| matches!(n, Node::DataModel(_)))
        .find(|n| {
            let data_model = n.into_data();
            data_model.name == "Person"
        })
        .ok_or_else(|| anyhow::anyhow!("Person data model not found"))?;

    let mut get_handler_verified = false;
    let mut post_handler_verified = false;

    for endpoint in &endpoints {
        let endpoint_data = endpoint.into_data();

        let is_get = endpoint_data.meta.get("verb").map_or(false, |v| v == "GET");

        let is_post = endpoint_data
            .meta
            .get("verb")
            .map_or(false, |v| v == "POST");

        if !is_get && !is_post {
            continue;
        }

        let handler_params = HandlerParams::default();

        let handler_results =
            language
                .lang()
                .handler_finder(endpoint_data.clone(), graph, handler_params);

        for (_, edge_opt) in handler_results {
            if let Some(edge) = edge_opt {
                let function_nodes = graph
                    .nodes
                    .iter()
                    .filter(|n| matches!(n, Node::Function(_)))
                    .filter(|n| {
                        let function_data = n.into_data();

                        function_data.name == edge.target.node_data.name
                            && function_data.file == edge.target.node_data.file
                    })
                    .collect::<Vec<_>>();

                if let Some(function_node) = function_nodes.first() {
                    let function_data = function_node.into_data();
                    let function_name = function_data.name.clone();

                    let contains_person = graph.edges.iter().any(|e| {
                        e.edge == EdgeType::Contains
                            && e.source.node_data.name == function_data.name.clone()
                            && e.source.node_data.file == function_data.file
                            && e.target.node_data.name == person_model.into_data().name
                    });

                    if contains_person {
                        if is_get && function_name_matches(&function_name, "GetPerson") {
                            info!(
                                "✓ GET handlder '{}' references Person datamodel",
                                function_name
                            );
                            get_handler_verified = true;
                        } else if is_post && function_name_matches(&function_name, "CreatePerson") {
                            info!(
                                "✓ POST handler '{}' references Person datamodel",
                                function_name
                            );
                            post_handler_verified = true;
                        }
                    }
                }
            }
        }
    }

    assert!(
        get_handler_verified,
        "GET endpoint handler for Person not found or doesn't reference Person datamodel",
    );

    assert!(
        post_handler_verified,
        "POST endpoint handler for Person not found or doesn't reference Person datamodel",
    );

    Ok(())
}
