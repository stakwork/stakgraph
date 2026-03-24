#[cfg(feature = "neo4j")]
mod inner {
    use std::path::Path;

    use ast::lang::graphs::graph_ops::GraphOps;
    use ast::lang::graphs::neo4j::{execute_node_query, find_nodes_by_name_contains_query};
    use ast::lang::graphs::{EdgeType, NodeType};
    use ast::repo::{Repo, Repos};
    use ast::Lang;
    use console::style;
    use lsp::Language;
    use shared::Result;

    use crate::args::{GraphArgs, GraphCommand};
    use crate::output::Output;
    use crate::progress::CliSpinner;
    use crate::utils::common_ancestor;

    const ALL_NODE_TYPES: &[NodeType] = &[
        NodeType::Repository,
        NodeType::Package,
        NodeType::Language,
        NodeType::Directory,
        NodeType::File,
        NodeType::Import,
        NodeType::Library,
        NodeType::Class,
        NodeType::Trait,
        NodeType::Instance,
        NodeType::Function,
        NodeType::Endpoint,
        NodeType::Request,
        NodeType::DataModel,
        NodeType::Feature,
        NodeType::Page,
        NodeType::Var,
        NodeType::UnitTest,
        NodeType::IntegrationTest,
        NodeType::E2eTest
    ];

    const ALL_EDGE_TYPES: &[EdgeType] = &[
        EdgeType::Calls,
        EdgeType::Uses,
        EdgeType::Operand,
        EdgeType::ArgOf,
        EdgeType::Contains,
        EdgeType::Imports,
        EdgeType::Of,
        EdgeType::Handler,
        EdgeType::Includes,
        EdgeType::Renders,
        EdgeType::ParentOf,
        EdgeType::Implements,
        EdgeType::NestedIn,
    ];

    async fn connect_graph_ops() -> Result<GraphOps> {
        let mut ops = GraphOps::new();
        ops.connect().await?;
        Ok(ops)
    }

    pub async fn run_graph(args: &GraphArgs, out: &mut Output) -> Result<()> {
        match &args.command {
            GraphCommand::Ingest(a) => run_ingest(&a.path, out).await,
            GraphCommand::Search(a) => run_search(&a.query, &a.node_type, a.limit, out).await,
            GraphCommand::Node(a) => run_node(&a.name, out).await,
            GraphCommand::Schema => run_schema(out),
            GraphCommand::Clear => run_clear(out).await,
            GraphCommand::Stats => run_stats(out).await,
        }
    }

    async fn run_ingest(path: &str, out: &mut Output) -> Result<()> {
        let canonical = std::fs::canonicalize(path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string());

        let mut file_list: Vec<String> = Vec::new();
        let root_path = Path::new(&canonical);
        if root_path.is_dir() {
            for entry in walkdir::WalkDir::new(root_path) {
                let entry = entry.map_err(|e| shared::Error::Io(e.into()))?;
                if entry.file_type().is_file() {
                    let p = entry.path().to_string_lossy().to_string();
                    if Language::from_path(&p).is_some() {
                        file_list.push(p);
                    }
                }
            }
        } else if root_path.is_file() {
            file_list.push(canonical.clone());
        }

        if file_list.is_empty() {
            out.writeln("No supported source files found.".to_string())?;
            return Ok(());
        }

        let spinner = CliSpinner::new("Building graph from source files...");

        let mut repos_vec: Vec<Repo> = Vec::new();
        let mut by_lang: std::collections::HashMap<Language, Vec<String>> =
            std::collections::HashMap::new();
        for f in &file_list {
            if let Some(lang) = Language::from_path(f) {
                by_lang.entry(lang).or_default().push(f.clone());
            }
        }
        for (language, files) in by_lang {
            let lang = Lang::from_language(language);
            if let Some(root) = common_ancestor(&files) {
                let file_refs: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
                let repo = Repo::from_files(&file_refs, root, lang, false, false, false)?;
                repos_vec.push(repo);
            }
        }

        let repos = Repos(repos_vec);
        spinner.set_message("Parsing source files...");
        let btree_graph = repos.build_graphs_local().await?;

        let (node_count, edge_count) = {
            use ast::lang::graphs::graph::Graph;
            let (n, e) = btree_graph.get_graph_size();
            (n, e)
        };

        spinner.set_message("Uploading to Neo4j...");
        let mut ops = connect_graph_ops().await.map_err(|e| {
            spinner.finish_with_message("Failed to connect to Neo4j");
            e
        })?;

        let (final_nodes, final_edges) = ops
            .upload_btreemap_to_neo4j(&btree_graph, None)
            .await
            .map_err(|e| {
                spinner.finish_with_message("Upload failed");
                e
            })?;

        spinner.finish_with_message("Ingest complete");
        out.writeln(format!(
            "Parsed:    {} nodes, {} edges",
            node_count, edge_count
        ))?;
        out.writeln(format!(
            "Neo4j now: {} nodes, {} edges",
            final_nodes, final_edges
        ))?;
        Ok(())
    }

    async fn run_search(
        query: &str,
        node_type_strs: &[String],
        limit: usize,
        out: &mut Output,
    ) -> Result<()> {
        let mut ops = connect_graph_ops().await?;
        let connection = ops.graph.ensure_connected().await?;

        let search_types: Vec<NodeType> = if node_type_strs.is_empty() {
            ALL_NODE_TYPES.to_vec()
        } else {
            node_type_strs
                .iter()
                .filter_map(|s| s.parse::<NodeType>().ok())
                .collect()
        };

        let mut results: Vec<(NodeType, ast::lang::NodeData)> = Vec::new();
        for nt in &search_types {
            let (q, params) = find_nodes_by_name_contains_query(nt, query);
            let nodes = execute_node_query(&connection, q, params).await;
            for node in nodes {
                results.push((nt.clone(), node));
                if results.len() >= limit {
                    break;
                }
            }
            if results.len() >= limit {
                break;
            }
        }

        if results.is_empty() {
            out.writeln(format!("No results for {:?}", query))?;
            return Ok(());
        }

        for (node_type, node_data) in &results {
            let type_label = style(node_type.to_string()).bold().cyan();
            let name_label = style(&node_data.name).bold();
            let file_label = style(&node_data.file).dim();
            out.writeln(format!(
                "  {}  {}  [{}:{}]",
                type_label, name_label, file_label, node_data.start
            ))?;
        }
        out.writeln(format!("\n{} result(s)", results.len()))?;
        Ok(())
    }

    async fn run_node(name: &str, out: &mut Output) -> Result<()> {
        let mut ops = connect_graph_ops().await?;
        let connection = ops.graph.ensure_connected().await?;

        let mut found: Vec<(NodeType, ast::lang::NodeData)> = Vec::new();
        for nt in ALL_NODE_TYPES {
            use ast::lang::graphs::neo4j::find_nodes_by_name_query;
            let (q, params) = find_nodes_by_name_query(nt, name, "");
            let nodes = execute_node_query(&connection, q, params).await;
            for node in nodes {
                found.push((nt.clone(), node));
            }
        }

        if found.is_empty() {
            out.writeln(format!("No node found with name {:?}", name))?;
            return Ok(());
        }

        for (node_type, node_data) in &found {
            let header = style(format!("[{}] {}", node_type, node_data.name))
                .bold()
                .cyan();
            out.writeln(header.to_string())?;
            out.writeln(format!("  file:  {}", node_data.file))?;
            out.writeln(format!(
                "  lines: {}–{}",
                node_data.start, node_data.end
            ))?;
            if !node_data.body.is_empty() {
                let preview: String = node_data.body.lines().take(5).collect::<Vec<_>>().join("\n");
                out.writeln(format!("  body:\n{}", preview))?;
            }
            out.newline()?;
        }
        Ok(())
    }

    fn run_schema(out: &mut Output) -> Result<()> {
        out.writeln(style("Node types:").bold().to_string())?;
        for nt in ALL_NODE_TYPES {
            out.writeln(format!("  {}", nt))?;
        }
        out.newline()?;
        out.writeln(style("Edge types:").bold().to_string())?;
        for et in ALL_EDGE_TYPES {
            out.writeln(format!("  {:?}", et))?;
        }
        Ok(())
    }

    async fn run_clear(out: &mut Output) -> Result<()> {
        use std::io::{self, BufRead, Write};

        print!("This will delete all nodes and edges from the graph. Continue? [y/N] ");
        io::stdout().flush().ok();

        let stdin = io::stdin();
        let line = stdin.lock().lines().next().unwrap_or(Ok(String::new()))?;
        if line.trim().to_lowercase() != "y" {
            out.writeln("Aborted.".to_string())?;
            return Ok(());
        }

        let mut ops = connect_graph_ops().await?;
        let spinner = CliSpinner::new("Clearing graph...");
        ops.clear().await?;
        spinner.finish_with_message("Graph cleared");
        let (nodes, edges) = ops.get_graph_size().await?;
        out.writeln(format!("Graph now: {} nodes, {} edges", nodes, edges))?;
        Ok(())
    }

    async fn run_stats(out: &mut Output) -> Result<()> {
        let ops = connect_graph_ops().await?;
        let (nodes, edges) = ops.get_graph_size().await?;
        out.writeln(format!("Nodes: {}", nodes))?;
        out.writeln(format!("Edges: {}", edges))?;
        Ok(())
    }

    pub async fn print_caller_counts(
        out: &mut Output,
        graph: &ast::lang::graphs::ArrayGraph,
        files_to_print: &[String],
    ) {
        use ast::lang::graphs::graph_ops::GraphOps;
        use ast::lang::graphs::neo4j::{boltmap_insert_str, execute_count_query};
        use ast::lang::graphs::NodeType;
        use neo4rs::BoltMap;

        let mut ops = GraphOps::new();
        if ops.connect().await.is_err() {
            eprintln!("warning: neo4j unavailable, skipping caller count annotations");
            return;
        }
        let connection = match ops.graph.ensure_connected().await {
            Ok(c) => c,
            Err(_) => {
                eprintln!("warning: neo4j unavailable, skipping caller count annotations");
                return;
            }
        };

        let functions: Vec<&ast::lang::NodeData> = graph
            .nodes
            .iter()
            .filter(|n| {
                n.node_type == NodeType::Function
                    && files_to_print.iter().any(|f| *f == n.node_data.file)
            })
            .map(|n| &n.node_data)
            .collect();

        if functions.is_empty() {
            return;
        }

        out.writeln(
            console::style("\n--- Caller counts (from graph) ---")
                .bold()
                .to_string(),
        )
        .ok();

        for func in functions {
            let query =
                "MATCH (caller:Function)-[:CALLS]->(target:Function {name: $name, file: $file}) RETURN count(caller) AS cnt"
                    .to_string();
            let mut params = BoltMap::new();
            boltmap_insert_str(&mut params, "name", &func.name);
            boltmap_insert_str(&mut params, "file", &func.file);

            let count = execute_count_query(&connection, query, params).await;
            if count > 0 {
                out.writeln(format!(
                    "  {:>4} caller(s)  {}  [{}]",
                    count,
                    console::style(&func.name).bold(),
                    func.file
                ))
                .ok();
            }
        }
    }
}

#[cfg(feature = "neo4j")]
pub use inner::{print_caller_counts, run_graph};
