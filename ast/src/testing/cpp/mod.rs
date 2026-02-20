use crate::lang::graphs::{EdgeType, NodeType};
use crate::lang::{Graph, Node};
use crate::{lang::Lang, repo::Repo};
use shared::error::Result;
use std::str::FromStr;

pub async fn test_cpp_web_api_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/cpp/web_api",
        Lang::from_str("cpp").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let functions = graph.find_nodes_by_type(NodeType::Function);
    let create_person_fn = functions
        .iter()
        .find(|f| f.name == "createPerson" && f.file.ends_with("web_api/model.cpp"))
        .expect("createPerson function not found");

    assert_eq!(
        create_person_fn.docs,
        Some("Creates a new person in the database".to_string()),
        "createPerson should have documentation"
    );

    // graph.analysis();

    let mut nodes = 0;
    let mut edges = 0;

    let language_nodes = graph.find_nodes_by_name(NodeType::Language, "cpp");
    nodes += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node");
    assert_eq!(
        language_nodes[0].name, "cpp",
        "Language node name should be 'cpp'"
    );
    assert!(
        "src/testing/cpp/web_api".contains(language_nodes[0].file.as_str()),
        "Language node file path is incorrect"
    );
    let repositories = graph.find_nodes_by_type(NodeType::Repository);
    nodes += repositories.len();
    assert_eq!(repositories.len(), 1, "Expected 1 repository node");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes += files.len();
    assert_eq!(files.len(), 7, "Expected 7 files");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes += directories.len();
    assert_eq!(directories.len(), 0, "Expected 0 directory");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes += imports.len();
    assert_eq!(imports.len(), 6, "Expected 6 imports");

    let main_import_body = format!(
        r#"#include "crow.h"

#include "routes.h"

#include "model.h"
"#
    );
    let main = imports
        .iter()
        .find(|i| i.file == "src/testing/cpp/web_api/main.cpp")
        .unwrap();

    assert_eq!(
        main.body, main_import_body,
        "Model import body is incorrect"
    );

    let classes = graph.find_nodes_by_type(NodeType::Class);
    nodes += classes.len();
    assert_eq!(classes.len(), 1, "Expected 1 class");
    assert_eq!(
        classes[0].name, "Database",
        "Class name should be 'Database'"
    );
    assert_eq!(
        classes[0].file, "src/testing/cpp/web_api/model.h",
        "Class file path is incorrect"
    );

    let data_models = graph.find_nodes_by_type(NodeType::DataModel);
    nodes += data_models.len();
    assert_eq!(data_models.len(), 1, "Expected 1 data models");
    assert!(
        data_models
            .iter()
            .any(|dm| dm.name == "Person" && dm.file == "src/testing/cpp/web_api/model.h"),
        "Expected Person data model not found"
    );

    let endpoints = graph.find_nodes_by_type(NodeType::Endpoint);
    nodes += endpoints.len();
    assert_eq!(endpoints.len(), 3, "Expected 3 endpoints");

    let get_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person/<int>" && e.meta.get("verb") == Some(&"ANY".to_string()))
        .expect("ANY endpoint not found");
    assert_eq!(get_endpoint.file, "src/testing/cpp/web_api/routes.cpp");

    let post_endpoint = endpoints
        .iter()
        .find(|e| e.name == "/person" && e.meta.get("verb") == Some(&"POST".to_string()))
        .expect("POST endpoint not found");
    assert_eq!(post_endpoint.file, "src/testing/cpp/web_api/routes.cpp");

    let anon_post = endpoints
        .iter()
        .find(|e| e.name == "/anon-post")
        .expect("POST /anon-post endpoint not found");
    assert!(
        anon_post
            .meta
            .get("handler")
            .unwrap()
            .contains("_METHOD_anon-post_lambda_L13"),
        "Incorrect anon post handler: {:?}",
        anon_post.meta.get("handler")
    );

    let handler_edges_count = graph.count_edges_of_type(EdgeType::Handler);
    edges += handler_edges_count;
    assert_eq!(handler_edges_count, 3, "Expected 3 handler edges");

    let function_calls = graph.count_edges_of_type(EdgeType::Calls);
    edges += function_calls;
    assert_eq!(function_calls, 3, "Expected 3 function calls");

    let contains = graph.count_edges_of_type(EdgeType::Contains);
    edges += contains;
    assert_eq!(contains, 30, "Expected 30 contains edges");

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges += of_edges;
    assert_eq!(of_edges, 2, "Expected 2 of edge");

    let nested_in = graph.count_edges_of_type(EdgeType::NestedIn);
    edges += nested_in;
    assert_eq!(
        nested_in, 4,
        "Expected 4 NestedIn edges for lambda functions"
    );

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes += variables.len();
    assert_eq!(variables.len(), 1, "Expected 1 variables");

    let instances = graph.find_nodes_by_type(NodeType::Instance);
    nodes += instances.len();
    assert_eq!(instances.len(), 1, "Expected 1 instances");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes += libraries.len();
    assert_eq!(libraries.len(), 2, "Expected 2 libraries");

    let pkg_file = files
        .iter()
        .find(|f| f.name == "CMakeLists.txt" && f.file == "src/testing/cpp/web_api/CMakeLists.txt")
        .map(|f| Node::new(NodeType::File, f.clone()))
        .expect("CMakeLists.txt not found");

    let sqlite_lib = libraries
        .iter()
        .find(|l| l.name == "SQLite3" && l.file == pkg_file.node_data.file)
        .map(|l| Node::new(NodeType::Library, l.clone()))
        .expect("sqlite3 library not found");

    let json_lib = libraries
        .iter()
        .find(|l| l.name == "nlohmann_json" && l.file == pkg_file.node_data.file)
        .map(|l| Node::new(NodeType::Library, l.clone()))
        .expect("json library not found");

    assert!(
        graph.has_edge(&pkg_file, &sqlite_lib, EdgeType::Contains),
        "Expected 'SQLite3' library to be contained in 'CMakeLists.txt'"
    );
    assert!(
        graph.has_edge(&pkg_file, &json_lib, EdgeType::Contains),
        "Expected 'nlohmann_json' library to be contained in 'CMakeLists.txt'"
    );

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes += functions.len();
    assert_eq!(functions.len(), 11, "Expected 11 functions");

    let database = classes
        .into_iter()
        .find(|c| c.name == "Database" && c.file == "src/testing/cpp/web_api/model.h")
        .map(|c| Node::new(NodeType::Class, c))
        .expect("Database class not found");

    let db = instances
        .into_iter()
        .find(|i| i.name == "db" && i.file == "src/testing/cpp/web_api/main.cpp")
        .map(|i| Node::new(NodeType::Instance, i))
        .expect("db instance not found");

    assert!(
        graph.has_edge(&db, &database, EdgeType::Of),
        "Expected 'Database' class to be of 'db' instance"
    );

    let person_data_model = graph
        .find_nodes_by_name(NodeType::DataModel, "Person")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/web_api/model.h")
        .map(|n| Node::new(NodeType::DataModel, n))
        .expect("Person DataModel not found in model.h");

    let database_class = graph
        .find_nodes_by_name(NodeType::Class, "Database")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/web_api/model.h")
        .map(|n| Node::new(NodeType::Class, n))
        .expect("Database class not found in model.h");

    let main_fn = graph
        .find_nodes_by_name(NodeType::Function, "main")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/web_api/main.cpp")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("main function not found in main.cpp");

    let setup_routes_fn = graph
        .find_nodes_by_name(NodeType::Function, "setup_routes")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/web_api/routes.cpp")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("setup_routes function not found in routes.cpp");

    let _post_endpoint = graph
        .find_nodes_by_name(NodeType::Endpoint, "/person")
        .into_iter()
        .find(|n| {
            n.file == "src/testing/cpp/web_api/routes.cpp"
                && n.meta.get("verb") == Some(&"POST".to_string())
        })
        .map(|n| Node::new(NodeType::Endpoint, n))
        .expect("POST /person endpoint not found in routes.cpp");
    let _new_person_fn = graph
        .find_nodes_by_name(NodeType::Function, "new_person")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/web_api/routes.cpp")
        .map(|n| Node::new(NodeType::Function, n))
        .expect("new_person function not found in routes.cpp");

    let model_h_file = graph
        .find_nodes_by_name(NodeType::File, "model.h")
        .into_iter()
        .find(|n| n.file == "src/testing/cpp/web_api/model.h")
        .map(|n| Node::new(NodeType::File, n))
        .expect("model.h file node not found");

    assert!(
        graph.has_edge(&model_h_file, &person_data_model, EdgeType::Contains),
        "Expected 'Database' class to contain 'Person' DataModel"
    );
    assert!(
        graph.has_edge(&model_h_file, &database_class, EdgeType::Contains),
        "Expected 'Database' class to contain 'Person' DataModel"
    );
    assert!(
        graph.has_edge(&main_fn, &setup_routes_fn, EdgeType::Calls),
        "Expected 'main' function to call 'setup_routes' function"
    );

    let (num_nodes, num_edges) = graph.get_graph_size();

    assert_eq!(
        num_nodes, nodes as u32,
        "Expected {} nodes found {}",
        nodes, num_nodes
    );
    assert_eq!(
        num_edges, edges as u32,
        "Expected {} edges found {}",
        edges, num_edges
    );

    Ok(())
}

pub async fn test_cpp_cuda_generic<G: Graph>() -> Result<()> {
    let repo = Repo::new(
        "src/testing/cpp/cuda",
        Lang::from_str("cpp").unwrap(),
        false,
        Vec::new(),
        Vec::new(),
    )
    .unwrap();

    let graph = repo.build_graph_inner::<G>().await?;

    let mut nodes = 0;
    let mut edges = 0;

    let functions = graph.find_nodes_by_type(NodeType::Function);
    nodes += functions.len();
    assert_eq!(
        functions.len(),
        101,
        "Expected 101 functions from CUDA suite"
    );

    let variables = graph.find_nodes_by_type(NodeType::Var);
    nodes += variables.len();
    assert_eq!(variables.len(), 3, "Expected 3 variables in CUDA suite");

    let files = graph.find_nodes_by_type(NodeType::File);
    nodes += files.len();
    assert_eq!(files.len(), 28, "Expected 28 files");

    let cu_files = files.iter().filter(|f| f.file.ends_with(".cu")).count();
    assert_eq!(cu_files, 23, "Expected 23 .cu files");

    let h_files = files.iter().filter(|f| f.file.ends_with(".h")).count();
    assert_eq!(h_files, 2, "Expected 2 .h files");

    let cuh_files = files.iter().filter(|f| f.file.ends_with(".cuh")).count();
    assert_eq!(cuh_files, 1, "Expected 1 .cuh file");

    let cpp_files = files.iter().filter(|f| f.file.ends_with(".cpp")).count();
    assert_eq!(cpp_files, 1, "Expected 1 .cpp file");

    let directories = graph.find_nodes_by_type(NodeType::Directory);
    nodes += directories.len();
    assert_eq!(
        directories.len(),
        7,
        "Expected 7 directories for CUDA structure"
    );

    let repositories = graph.find_nodes_by_type(NodeType::Repository);
    nodes += repositories.len();
    assert_eq!(repositories.len(), 1, "Expected 1 repository node");

    let language_nodes = graph.find_nodes_by_name(NodeType::Language, "cpp");
    nodes += language_nodes.len();
    assert_eq!(language_nodes.len(), 1, "Expected 1 language node for cpp");

    let imports = graph.find_nodes_by_type(NodeType::Import);
    nodes += imports.len();
    assert_eq!(imports.len(), 26, "Expected 26 imports");

    let libraries = graph.find_nodes_by_type(NodeType::Library);
    nodes += libraries.len();
    assert_eq!(libraries.len(), 1, "Expected 1 library node");

    let _vector_add = functions
        .iter()
        .find(|f| f.name == "vectorAdd" && f.file.ends_with("kernels/vector_add.cu"))
        .expect("vectorAdd kernel not found");

    let _matrix_mul = functions
        .iter()
        .find(|f| f.name == "matrixMulTiled" && f.file.ends_with("kernels/matrix_multiply.cu"))
        .expect("matrixMulTiled kernel not found");

    let _reduce_kernel = functions
        .iter()
        .find(|f| f.name == "reduce_sum" && f.file.ends_with("kernels/reduction.cu"))
        .expect("reduce_sum kernel not found");

    let _convolve = functions
        .iter()
        .find(|f| f.name == "convolve2D" && f.file.ends_with("kernels/convolution.cu"))
        .expect("convolve2D kernel not found");

    let _tensor_ops = functions
        .iter()
        .find(|f| f.name == "tensorTranspose" && f.file.ends_with("kernels/tensor_ops.cu"))
        .expect("tensorTranspose kernel not found");

    let _unified_mem = functions
        .iter()
        .find(|f| f.name == "unifiedMemoryKernel" && f.file.ends_with("memory/unified_memory.cu"))
        .expect("unifiedMemoryKernel not found");

    let _pinned_mem = functions
        .iter()
        .find(|f| f.name == "pinnedMemoryKernel" && f.file.ends_with("memory/pinned_memory.cu"))
        .expect("pinnedMemoryKernel not found");

    let _texture_mem = functions
        .iter()
        .find(|f| f.name == "textureMemoryKernel" && f.file.ends_with("memory/texture_memory.cu"))
        .expect("textureMemoryKernel not found");

    let _shared_mem = functions
        .iter()
        .find(|f| f.name == "sharedMemoryReduction" && f.file.ends_with("memory/shared_memory.cu"))
        .expect("sharedMemoryReduction not found");

    let _p2p = functions
        .iter()
        .find(|f| f.name == "p2pTransferKernel" && f.file.ends_with("multi_gpu/p2p_access.cu"))
        .expect("p2pTransferKernel not found");

    let _multi_device = functions
        .iter()
        .find(|f| f.name == "multiDeviceKernel" && f.file.ends_with("multi_gpu/multi_device.cu"))
        .expect("multiDeviceKernel not found");

    let _ipc_mem = functions
        .iter()
        .find(|f| f.name == "ipcMemoryKernel" && f.file.ends_with("multi_gpu/ipc_memory.cu"))
        .expect("ipcMemoryKernel not found");

    let _async_kernel = functions
        .iter()
        .find(|f| f.name == "asyncKernel" && f.file.ends_with("streams/async_ops.cu"))
        .expect("asyncKernel not found");

    let _priority_kernel = functions
        .iter()
        .find(|f| f.name == "priorityKernel" && f.file.ends_with("streams/stream_priorities.cu"))
        .expect("priorityKernel not found");

    let _concurrent_kernel = functions
        .iter()
        .find(|f| f.name == "concurrentKernel" && f.file.ends_with("streams/concurrent_kernels.cu"))
        .expect("concurrentKernel not found");

    let _coop_groups = functions
        .iter()
        .find(|f| {
            f.name == "cooperativeGroupsReduction"
                && f.file.ends_with("advanced/cooperative_groups.cu")
        })
        .expect("cooperativeGroupsReduction not found");

    let _dynamic_parallel = functions
        .iter()
        .find(|f| {
            f.name == "dynamicParallelismKernel"
                && f.file.ends_with("advanced/dynamic_parallelism.cu")
        })
        .expect("dynamicParallelismKernel not found");

    let _tensor_cores = functions
        .iter()
        .find(|f| f.name == "tensorCoreMatmul" && f.file.ends_with("advanced/tensor_cores.cu"))
        .expect("tensorCoreMatmul not found");

    let _cuda_graphs = functions
        .iter()
        .find(|f| f.name == "graphKernel" && f.file.ends_with("advanced/cuda_graphs.cu"))
        .expect("graphKernel not found");

    let _main_fn = functions
        .iter()
        .find(|f| f.name == "main" && f.file.ends_with("host/main.cu"))
        .expect("main function not found");

    let _device_query = functions
        .iter()
        .find(|f| f.name == "queryAllDevices" && f.file.ends_with("host/device_query.cpp"))
        .expect("queryAllDevices not found");

    let calls_edges = graph.count_edges_of_type(EdgeType::Calls);
    edges += calls_edges;
    assert_eq!(calls_edges, 4, "Check Calls edge count");

    let contains_edges = graph.count_edges_of_type(EdgeType::Contains);
    edges += contains_edges;
    assert_eq!(contains_edges, 168, "Expected 168 Contains edges");

    let nested_in_edges = graph.count_edges_of_type(EdgeType::NestedIn);
    edges += nested_in_edges;
    assert_eq!(nested_in_edges, 0, "Check NestedIn edge count");

    let uses_edges = graph.count_edges_of_type(EdgeType::Uses);
    edges += uses_edges;
    assert_eq!(uses_edges, 0, "Check Uses edge count");

    let of_edges = graph.count_edges_of_type(EdgeType::Of);
    edges += of_edges;
    assert_eq!(of_edges, 1, "Check Of edge count");

    let imports_edges = graph.count_edges_of_type(EdgeType::Imports);
    edges += imports_edges;

    let (num_nodes, num_edges) = graph.get_graph_size();
    assert_eq!(
        num_nodes, nodes as u32,
        "Expected {} nodes found {}",
        nodes, num_nodes
    );
    assert_eq!(
        num_edges, edges as u32,
        "Expected {} edges found {}",
        edges, num_edges
    );
    assert_eq!(num_nodes as u32, 168, "Expected 168 nodes");
    assert_eq!(num_edges as u32, 173, "Expected 173 edges");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cpp() {
    #[cfg(not(feature = "neo4j"))]
    {
        use crate::lang::graphs::{ArrayGraph, BTreeMapGraph};
        test_cpp_web_api_generic::<ArrayGraph>().await.unwrap();
        test_cpp_web_api_generic::<BTreeMapGraph>().await.unwrap();

        test_cpp_cuda_generic::<ArrayGraph>().await.unwrap();
        test_cpp_cuda_generic::<BTreeMapGraph>().await.unwrap();
    }

    #[cfg(feature = "neo4j")]
    {
        use crate::lang::graphs::Neo4jGraph;
        let graph = Neo4jGraph::default();
        graph.clear().await.unwrap();

        test_cpp_web_api_generic::<Neo4jGraph>().await.unwrap();

        graph.clear().await.unwrap();

        test_cpp_cuda_generic::<Neo4jGraph>().await.unwrap();
    }
}
