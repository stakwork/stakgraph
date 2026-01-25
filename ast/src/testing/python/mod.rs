use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_python_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/python",
        Lang::from_str("python").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    graph.analysis();

    let mut nodes_count = 0;
    let mut edges_count = 0;

    let language_nodes = graph.find_nodes_by_type(NodeType::Language);
    nodes_count += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "python",
        "Language node name should be 'python'"
    );
    assert_eq!(
        language_nodes[0].file, "src/testing/python",
        "Language node file path is incorrect"
    );

    let repositories = graph.find_nodes_by_type(NodeType::Repository);
    nodes_count += repositories.len();
    assert_eq!(repositories.len(), 1, "Expected 1 repository node");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes_count += directories.len();
    assert_eq!(directories.len(), 4, "Expected 4 directories");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes_count += files.len();
    assert_eq!(files.len(), 20, "Expected 20 files");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes_count += imports.len();
    assert_eq!(imports.len(), 16, "Expected 16 imports");

    let calls = graph.count_edges_of_type(EdgeType::Calls);
    edges_count += calls;
    assert_eq!(calls, 17, "Expected 17 call edges");

    let implements = graph.count_edges_of_type(EdgeType::Implements);
    edges_count += implements;
    assert_eq!(implements, 1, "Expected 1 implements edges");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    assert_eq!(contains, 139, "Expected 139 contains edges");
    edges_count += contains;

    let handlers = graph.count_edges_of_type(EdgeType::Handler);
    edges_count += handlers;
    //FIXME: this ough t o be 6 hadndlers
    assert_eq!(handlers, 4, "Expected 4 handler edges");

    let uses = graph.count_edges_of_type(EdgeType::Uses);
    edges_count += uses;
    assert_eq!(uses, 0, "Expected 0 uses edges");

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges_count += of_edges;
    assert_eq!(of_edges, 1, "Expected 1 of edges");

    let parent_of = graph.count_edges_of_type(EdgeType::ParentOf);
    edges_count += parent_of;
    assert_eq!(parent_of, 1, "Expected 1 parent_of edges");

    let renders = graph.count_edges_of_type(EdgeType::Renders);
    edges_count += renders;
    assert_eq!(renders, 0, "Expected 0 renders edges");

    let argof = graph.count_edges_of_type(EdgeType::ArgOf);
    edges_count += argof;
    assert_eq!(argof, 0, "Expected 0 argof edges");

    let operand = graph.count_edges_of_type(EdgeType::Operand);
    edges_count += operand;
    assert_eq!(operand, 11, "Expected 11 operand edges");

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes_count += functions.len();
    assert_eq!(functions.len(), 27, "Expected 27 functions");

    let librabries = graph.find_nodes_by_type(NodeType::Library);
    nodes_count += librabries.len();
    assert_eq!(librabries.len(), 4, "Expected 4 libraries");

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes_count += instances.len();
    assert_eq!(instances.len(), 0, "Expected 0 instance");

    let main_import_body = format!(
        r#"import os
import signal
import subprocess
import sys
from fastapi import FastAPI
from flask import Flask
from fastapi_app.routes import router
from database import Base, engine
from flask_app.routes import flask_bp"#
    );
    let main = imports
        .iter()
        .find(|i| i.file == "src/testing/python/main.py")
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );
    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes_count += classes.len();

    assert_eq!(classes.len(), 8, "Expected 8 classes");

    let vars = graph.find_nodes_by_type(NodeType::Var);
    nodes_count += vars.len();
    assert_eq!(vars.len(), 27, "Expected 27 variables");

    let mut sorted_classes = classes.clone();
    sorted_classes.sort_by(|a, b| a.name.cmp(&b.name));

    assert!(
        classes
            .iter()
            .any(|c| c.name == "Person" && c.file == "src/testing/python/model.py"),
        "Expected Person class not found"
    );

    let class_function_edges =
        graph.find_nodes_with_edge_type(NodeType::Class, NodeType::Function, EdgeType::Operand);
    assert_eq!(class_function_edges.len(), 11, "Expected 11 methods");

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes_count += data_models.len();
    //should be 3, but some classes are picked up as datamodels
    assert_eq!(data_models.len(), 10, "Expected 10 data models");

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes_count += endpoints.len();
    assert_eq!(endpoints.len(), 6, "Expected 6 endpoints");

    let trait_nodes = graph.find_nodes_by_type(NodeType::Trait);
    nodes_count += trait_nodes.len();
    assert_eq!(trait_nodes.len(), 1, "Expected 1 traits");

    let imported_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges_count += imported_edges;
    assert_eq!(imported_edges, 10, "Expected 10 import edges");

    let person_class = graph
        .find_nodes_by_name(NodeType::Class, "Person")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("Person class not found in model.py");

    let create_or_edit_person_dm = graph
        .find_nodes_by_name(NodeType::DataModel, "CreateOrEditPerson")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::DataModel, n))
        .expect("CreateOrEditPerson DataModel not found in model.py");

    let model_py_file = graph
        .find_nodes_by_name(NodeType::File, "model.py")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::File, n))
        .expect("model.py file node not found");

    let fastapi_post_endpoint = graph
        .find_nodes_by_name(NodeType::Endpoint, "/person/")
        .into_iter()
        .find(|n| {
            n.file == "src/testing/python/fastapi_app/routes.py"
                && n.meta.get("verb") == Some(&"POST".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n))
        .expect("FastAPI /person/ POST endpoint not found");

    let create_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "create_person")
        .into_iter()
        .find(|n| n.file == "src/testing/python/fastapi_app/routes.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("create_person function not found in fastapi_app/routes.py");

    assert!(
        graph.has_edge(&model_py_file, &person_class, EdgeType::Contains),
        "Expected 'model.py' file to contain 'Person' class"
    );
    assert!(
        graph.has_edge(
            &model_py_file,
            &create_or_edit_person_dm,
            EdgeType::Contains
        ),
        "Expected 'model.py' file to contain 'CreateOrEditPerson' DataModel"
    );
    assert!(
        graph.has_edge(&fastapi_post_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected FastAPI '/person/' POST endpoint to be handled by 'create_person'"
    );

    let django_views_file = graph
        .find_nodes_by_name(NodeType::File, "views.py")
        .into_iter()
        .find(|n| n.file == "src/testing/python/django_app/views.py")
        .map(|n| Node::new(NodeType::File, n))
        .expect("Django views.py file not found");

    let django_get_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "get_person")
        .into_iter()
        .find(|n| n.file == "src/testing/python/django_app/views.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("Django get_person function not found");

    let django_create_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "create_person")
        .into_iter()
        .find(|n| n.file == "src/testing/python/django_app/views.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("Django create_person function not found");

    assert!(
        graph.has_edge(
            &django_views_file,
            &django_get_person_fn,
            EdgeType::Contains
        ),
        "Expected Django views.py to contain get_person function"
    );

    assert!(
        graph.has_edge(
            &django_views_file,
            &django_create_person_fn,
            EdgeType::Contains
        ),
        "Expected Django views.py to contain create_person function"
    );

    let flask_routes_file = graph
        .find_nodes_by_name(NodeType::File, "routes.py")
        .into_iter()
        .find(|n| n.file == "src/testing/python/flask_app/routes.py")
        .map(|n| Node::new(NodeType::File, n))
        .expect("Flask routes.py file not found");

    let flask_get_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "get_person")
        .into_iter()
        .find(|n| n.file == "src/testing/python/flask_app/routes.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("Flask get_person function not found");

    let flask_create_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "create_person")
        .into_iter()
        .find(|n| n.file == "src/testing/python/flask_app/routes.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("Flask create_person function not found");

    assert!(
        graph.has_edge(&flask_routes_file, &flask_get_person_fn, EdgeType::Contains),
        "Expected Flask routes.py to contain get_person function"
    );

    assert!(
        graph.has_edge(
            &flask_routes_file,
            &flask_create_person_fn,
            EdgeType::Contains
        ),
        "Expected Flask routes.py to contain create_person function"
    );

    let fastapi_routes_file = graph
        .find_nodes_by_name(NodeType::File, "routes.py")
        .into_iter()
        .find(|n| n.file == "src/testing/python/fastapi_app/routes.py")
        .map(|n| Node::new(NodeType::File, n))
        .expect("FastAPI routes.py file not found");

    let fastapi_get_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "get_person")
        .into_iter()
        .find(|n| n.file == "src/testing/python/fastapi_app/routes.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("FastAPI get_person function not found");

    assert!(
        graph.has_edge(
            &fastapi_routes_file,
            &fastapi_get_person_fn,
            EdgeType::Contains
        ),
        "Expected FastAPI routes.py to contain get_person function"
    );

    assert!(
        graph.has_edge(&fastapi_routes_file, &create_person_fn, EdgeType::Contains),
        "Expected FastAPI routes.py to contain create_person function"
    );

    let db_file = graph
        .find_nodes_by_name(NodeType::File, "db.py")
        .into_iter()
        .find(|n| n.file == "src/testing/python/db.py")
        .map(|n| Node::new(NodeType::File, n))
        .expect("db.py file not found");

    let get_db_fn = graph
        .find_nodes_by_name(NodeType::Function, "get_db")
        .into_iter()
        .find(|n| n.file == "src/testing/python/db.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("get_db function not found in db.py");

    let db_session_fn = graph
        .find_nodes_by_name(NodeType::Function, "db_session")
        .into_iter()
        .find(|n| n.file == "src/testing/python/db.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("db_session function not found in db.py");

    let db_session_attrs = db_session_fn.node_data.meta.get("attributes");
    assert!(
        db_session_attrs.is_some(),
        "db_session should have attributes"
    );
    let attrs = db_session_attrs.unwrap();
    assert!(
        attrs.contains("contextmanager"),
        "db_session attributes should contain 'contextmanager'"
    );

    let species_fn = graph
        .find_nodes_by_name(NodeType::Function, "species")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("species function not found in model.py");

    let species_attrs = species_fn.node_data.meta.get("attributes");
    assert!(species_attrs.is_some(), "species should have attributes");
    let species_attr_val = species_attrs.unwrap();
    assert!(
        species_attr_val.contains("property"),
        "species attributes should contain 'property'"
    );

    let is_mammal_fn = graph
        .find_nodes_by_name(NodeType::Function, "is_mammal")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("is_mammal function not found in model.py");

    let is_mammal_attrs = is_mammal_fn.node_data.meta.get("attributes");
    assert!(
        is_mammal_attrs.is_some(),
        "is_mammal should have attributes"
    );
    let is_mammal_attr_val = is_mammal_attrs.unwrap();
    assert!(
        is_mammal_attr_val.contains("staticmethod"),
        "is_mammal attributes should contain 'staticmethod'"
    );

    let create_puppy_fn = graph
        .find_nodes_by_name(NodeType::Function, "create_puppy")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("create_puppy function not found in model.py");

    let create_puppy_attrs = create_puppy_fn.node_data.meta.get("attributes");
    assert!(
        create_puppy_attrs.is_some(),
        "create_puppy should have attributes"
    );
    let create_puppy_attr_val = create_puppy_attrs.unwrap();
    assert!(
        create_puppy_attr_val.contains("classmethod"),
        "create_puppy attributes should contain 'classmethod'"
    );

    let get_animal_info_fn = graph
        .find_nodes_by_name(NodeType::Function, "get_animal_info")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("get_animal_info function not found in model.py");

    let get_animal_info_attrs = get_animal_info_fn.node_data.meta.get("attributes");
    assert!(
        get_animal_info_attrs.is_some(),
        "get_animal_info should have attributes"
    );
    let get_animal_info_attr_val = get_animal_info_attrs.unwrap();
    assert!(
        get_animal_info_attr_val.contains("lru_cache"),
        "get_animal_info attributes should contain 'lru_cache'"
    );
    assert!(
        get_animal_info_attr_val.contains("maxsize"),
        "get_animal_info attributes should contain decorator argument 'maxsize'"
    );

    let get_person_by_id_fn = graph
        .find_nodes_by_name(NodeType::Function, "get_person_by_id")
        .into_iter()
        .find(|n| n.file == "src/testing/python/db.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("get_person_by_id function not found in db.py");

    let create_new_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "create_new_person")
        .into_iter()
        .find(|n| n.file == "src/testing/python/db.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("create_new_person function not found in db.py");

    assert!(
        graph.has_edge(&db_file, &get_db_fn, EdgeType::Contains),
        "Expected db.py to contain get_db function"
    );

    assert!(
        graph.has_edge(&db_file, &db_session_fn, EdgeType::Contains),
        "Expected db.py to contain db_session function"
    );

    assert!(
        graph.has_edge(&db_file, &get_person_by_id_fn, EdgeType::Contains),
        "Expected db.py to contain get_person_by_id function"
    );

    assert!(
        graph.has_edge(&db_file, &create_new_person_fn, EdgeType::Contains),
        "Expected db.py to contain create_new_person function"
    );

    // Test function calls between database operations and route handlers
    assert!(
        graph.has_edge(&django_get_person_fn, &db_session_fn, EdgeType::Calls),
        "Expected Django get_person to call db_session"
    );

    assert!(
        graph.has_edge(&django_get_person_fn, &get_person_by_id_fn, EdgeType::Calls),
        "Expected Django get_person to call get_person_by_id"
    );

    assert!(
        graph.has_edge(&django_create_person_fn, &db_session_fn, EdgeType::Calls),
        "Expected Django create_person to call db_session"
    );

    assert!(
        graph.has_edge(
            &django_create_person_fn,
            &create_new_person_fn,
            EdgeType::Calls
        ),
        "Expected Django create_person to call create_new_person"
    );

    assert!(
        graph.has_edge(&flask_get_person_fn, &db_session_fn, EdgeType::Calls),
        "Expected Flask get_person to call db_session"
    );

    assert!(
        graph.has_edge(&flask_get_person_fn, &get_person_by_id_fn, EdgeType::Calls),
        "Expected Flask get_person to call get_person_by_id"
    );

    assert!(
        graph.has_edge(&flask_create_person_fn, &db_session_fn, EdgeType::Calls),
        "Expected Flask create_person to call db_session"
    );

    assert!(
        graph.has_edge(
            &flask_create_person_fn,
            &create_new_person_fn,
            EdgeType::Calls
        ),
        "Expected Flask create_person to call create_new_person"
    );

    assert!(
        graph.has_edge(
            &fastapi_get_person_fn,
            &get_person_by_id_fn,
            EdgeType::Calls
        ),
        "Expected FastAPI get_person to call get_person_by_id"
    );

    assert!(
        graph.has_edge(&create_person_fn, &create_new_person_fn, EdgeType::Calls),
        "Expected FastAPI create_person to call create_new_person"
    );

    let flask_get_endpoint = graph
        .find_nodes_by_name(NodeType::Endpoint, "/person/<int:id>")
        .into_iter()
        .find(|n| n.file == "src/testing/python/flask_app/routes.py")
        .map(|n| Node::new(NodeType::Endpoint, n))
        .expect("Flask GET endpoint not found");

    let flask_post_endpoint = graph
        .find_nodes_by_name(NodeType::Endpoint, "/person/")
        .into_iter()
        .find(|n| n.file == "src/testing/python/flask_app/routes.py")
        .map(|n| Node::new(NodeType::Endpoint, n))
        .expect("Flask POST endpoint not found");

    let fastapi_get_endpoint = graph
        .find_nodes_by_name(NodeType::Endpoint, "/person/{id}")
        .into_iter()
        .find(|n| n.file == "src/testing/python/fastapi_app/routes.py")
        .map(|n| Node::new(NodeType::Endpoint, n))
        .expect("FastAPI GET endpoint not found");

    assert!(
        graph.has_edge(
            &fastapi_get_endpoint,
            &fastapi_get_person_fn,
            EdgeType::Handler
        ),
        "Expected FastAPI GET endpoint to be handled by get_person"
    );

    assert!(
        graph.has_edge(&fastapi_post_endpoint, &create_person_fn, EdgeType::Handler),
        "Expected FastAPI '/person/' POST endpoint to be handled by 'create_person'"
    );

    assert!(
        graph.has_edge(&flask_get_endpoint, &flask_get_person_fn, EdgeType::Handler),
        "Expected Flask GET endpoint to be handled by get_person"
    );

    assert!(
        graph.has_edge(
            &flask_post_endpoint,
            &flask_create_person_fn,
            EdgeType::Handler
        ),
        "Expected Flask POST endpoint to be handled by create_person"
    );

    let repr_fn = graph
        .find_nodes_by_name(NodeType::Function, "__repr__")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("__repr__ method not found in model.py");

    let str_fn = graph
        .find_nodes_by_name(NodeType::Function, "__str__")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("__str__ method not found in model.py");

    assert!(
        graph.has_edge(&model_py_file, &repr_fn, EdgeType::Contains),
        "Expected model.py to contain __repr__ method"
    );

    assert!(
        graph.has_edge(&model_py_file, &str_fn, EdgeType::Contains),
        "Expected model.py to contain __str__ method"
    );

    assert!(
        graph.has_edge(&person_class, &repr_fn, EdgeType::Operand),
        "Expected Person class to have __repr__ method"
    );

    assert!(
        graph.has_edge(&person_class, &str_fn, EdgeType::Operand),
        "Expected Person class to have __str__ method"
    );

    // Test data model relationships and usage
    let person_response_dm = graph
        .find_nodes_by_name(NodeType::DataModel, "PersonResponse")
        .into_iter()
        .find(|n| n.file == "src/testing/python/model.py")
        .map(|n| Node::new(NodeType::DataModel, n))
        .expect("PersonResponse DataModel not found in model.py");

    assert!(
        graph.has_edge(&model_py_file, &person_response_dm, EdgeType::Contains),
        "Expected model.py to contain PersonResponse DataModel"
    );

    assert!(
        graph.has_edge(
            &fastapi_get_person_fn,
            &person_response_dm,
            EdgeType::Contains
        ),
        "Expected FastAPI get_person to contain PersonResponse DataModel"
    );

    assert!(
        graph.has_edge(&create_person_fn, &person_response_dm, EdgeType::Contains),
        "Expected FastAPI create_person to contain PersonResponse DataModel"
    );

    let _person_class_import_edge = graph
        .find_nodes_with_edge_type(NodeType::File, NodeType::Class, EdgeType::Imports)
        .into_iter()
        .find(|(source, target)| {
            source.file == "src/testing/python/db.py" && target.name == "Person"
        })
        .expect("Expected db.py to import Person class");

    let session_local_var = graph
        .find_nodes_by_name(NodeType::Var, "SessionLocal")
        .into_iter()
        .find(|n| n.file == "src/testing/python/database.py")
        .map(|n| Node::new(NodeType::Var, n))
        .expect("SessionLocal variable not found in database.py");

    assert!(
        graph.has_edge(&get_db_fn, &session_local_var, EdgeType::Contains),
        "Expected get_db function to contain SessionLocal variable"
    );

    assert!(
        graph.has_edge(&db_session_fn, &session_local_var, EdgeType::Contains),
        "Expected db_session function to contain SessionLocal variable"
    );

    let main_file = graph
        .find_nodes_by_name(NodeType::File, "main.py")
        .into_iter()
        .find(|n| n.file == "src/testing/python/main.py")
        .map(|n| Node::new(NodeType::File, n))
        .expect("main.py file not found");
    let cleanup_fn = graph
        .find_nodes_by_name(NodeType::Function, "cleanup")
        .into_iter()
        .find(|n| n.file == "src/testing/python/main.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("cleanup function not found in main.py");

    let signal_handler_fn = graph
        .find_nodes_by_name(NodeType::Function, "signal_handler")
        .into_iter()
        .find(|n| n.file == "src/testing/python/main.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("signal_handler function not found in main.py");

    let run_servers_fn = graph
        .find_nodes_by_name(NodeType::Function, "run_servers")
        .into_iter()
        .find(|n| n.file == "src/testing/python/main.py")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("run_servers function not found in main.py");

    assert!(
        graph.has_edge(&main_file, &cleanup_fn, EdgeType::Contains),
        "Expected main.py to contain cleanup function"
    );

    assert!(
        graph.has_edge(&main_file, &signal_handler_fn, EdgeType::Contains),
        "Expected main.py to contain signal_handler function"
    );

    assert!(
        graph.has_edge(&main_file, &run_servers_fn, EdgeType::Contains),
        "Expected main.py to contain run_servers function"
    );

    assert!(
        graph.has_edge(&signal_handler_fn, &cleanup_fn, EdgeType::Calls),
        "Expected signal_handler to call cleanup"
    );

    assert!(
        graph.has_edge(&run_servers_fn, &cleanup_fn, EdgeType::Calls),
        "Expected run_servers to call cleanup"
    );

    let django_settings_file = graph
        .find_nodes_by_name(NodeType::File, "settings.py")
        .into_iter()
        .find(|n| n.file == "src/testing/python/django_app/settings.py")
        .map(|n| Node::new(NodeType::File, n))
        .expect("Django settings.py file not found");

    let secret_key_var = graph
        .find_nodes_by_name(NodeType::Var, "SECRET_KEY")
        .into_iter()
        .find(|n| n.file == "src/testing/python/django_app/settings.py")
        .map(|n| Node::new(NodeType::Var, n))
        .expect("SECRET_KEY variable not found in Django settings.py");

    let debug_var = graph
        .find_nodes_by_name(NodeType::Var, "DEBUG")
        .into_iter()
        .find(|n| n.file == "src/testing/python/django_app/settings.py")
        .map(|n| Node::new(NodeType::Var, n))
        .expect("DEBUG variable not found in Django settings.py");

    assert!(
        graph.has_edge(&django_settings_file, &secret_key_var, EdgeType::Contains),
        "Expected Django settings.py to contain SECRET_KEY variable"
    );

    assert!(
        graph.has_edge(&django_settings_file, &debug_var, EdgeType::Contains),
        "Expected Django settings.py to contain DEBUG variable"
    );

    let unit_tests = graph.find_nodes_by_type(NodeType::UnitTest);
    nodes_count += unit_tests.len();
    assert_eq!(unit_tests.len(), 5, "Expected 5 Unit Tests");

    let integration_tests = graph.find_nodes_by_type(NodeType::IntegrationTest);
    nodes_count += integration_tests.len();
    assert_eq!(integration_tests.len(), 2, "Expected 2 Integration Tests");

    let e2e_tests = graph.find_nodes_by_type(NodeType::E2eTest);
    nodes_count += e2e_tests.len();
    assert_eq!(e2e_tests.len(), 3, "Expected 3 E2E Tests");

    let puppy_test = unit_tests
        .iter()
        .find(|t| t.name == "test_puppy_creation")
        .map(|n| Node::new(NodeType::UnitTest, n.clone()))
        .expect("test_puppy_creation not found");

    let api_test = integration_tests
        .iter()
        .find(|t| t.name == "test_create_person_api")
        .map(|n| Node::new(NodeType::IntegrationTest, n.clone()))
        .expect("test_create_person_api not found");

    let flow_test = e2e_tests
        .iter()
        .find(|t| t.name == "test_create_user_flow")
        .map(|n| Node::new(NodeType::E2eTest, n.clone()))
        .expect("test_create_user_flow not found");

    assert!(
        graph.has_edge(&puppy_test, &create_puppy_fn, EdgeType::Calls),
        "Expected test_puppy_creation to call create_puppy"
    );

    let unit_test_file = files
        .iter()
        .find(|f| f.file.starts_with("src/testing/python/tests") && f.name == "unit_test.py")
        .expect("unit_test.py not found");
    assert!(graph.has_edge(
        &Node::new(NodeType::File, unit_test_file.clone()),
        &puppy_test,
        EdgeType::Contains
    ));

    let e2e_test_file = files
        .iter()
        .find(|f| f.file.starts_with("src/testing/python/tests") && f.name == "e2e_test.py")
        .expect("e2e_test.py not found");
    assert!(graph.has_edge(
        &Node::new(NodeType::File, e2e_test_file.clone()),
        &flow_test,
        EdgeType::Contains
    ));

    assert!(
        graph.has_edge(&api_test, &fastapi_post_endpoint, EdgeType::Calls),
        "Expected test_create_person_api to call FastAPI POST endpoint"
    );

    let _modern_file = files
        .iter()
        .find(|f| f.file.ends_with("modern.py"))
        .expect("modern.py not found");

    let typed_global = vars
        .iter()
        .find(|v| v.name == "typed_global")
        .expect("typed_global not found");
    assert_eq!(
        typed_global.data_type,
        Some("int".to_string()),
        "Expected typed_global to have type int"
    );

    let _async_func = functions
        .iter()
        .find(|f| f.name == "fetch_data")
        .expect("async fetch_data function not found");

    let _async_method = functions
        .iter()
        .find(|f| f.name == "process")
        .expect("async process method not found");

    let user_class = classes
        .iter()
        .find(|c| c.name == "User")
        .expect("User class not found");

    assert!(
        user_class
            .meta
            .get("attributes")
            .map_or(false, |a: &String| a.contains("dataclass")),
        "User class should have dataclass attribute"
    );

    let (nodes, edges) = graph.get_graph_size();

    assert_eq!(
        nodes as usize, nodes_count,
        "Expected {} nodes, found {}",
        nodes_count, nodes
    );
    assert_eq!(
        edges as usize, edges_count,
        "Expected {} edges, found {}",
        edges_count, edges
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_python() {
    use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
    test_python_generic::<ArrayGraph>().await.unwrap();
    test_python_generic::<BTreeMapGraph>().await.unwrap();

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();
        test_python_generic::<Neo4jGraph>().await.unwrap();
    }
}
