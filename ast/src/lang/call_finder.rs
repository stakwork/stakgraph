use super::parse::utils::trim_quotes;
use super::queries::consts::{IMPORTS_ALIAS, IMPORTS_FROM, IMPORTS_NAME};
use super::{graphs::Graph, *};
use tree_sitter::QueryCursor;

pub fn node_data_finder<G: Graph>(
    func_name: &str,
    operand: &Option<String>,
    graph: &G,
    current_file: &str,
    source_start: usize,
    source_node_type: NodeType,
    import_names: Option<Vec<(String, Vec<String>)>>,
    lang: &Lang,
) -> Option<NodeData> {
    func_target_file_finder(
        func_name,
        operand,
        graph,
        current_file,
        source_start,
        source_node_type,
        import_names,
        lang,
    )
}

pub fn func_target_file_finder<G: Graph>(
    func_name: &str,
    operand: &Option<String>,
    graph: &G,
    current_file: &str,
    source_start: usize,
    source_node_type: NodeType,
    import_names: Option<Vec<(String, Vec<String>)>>,
    lang: &Lang,
) -> Option<NodeData> {
    log_cmd(format!(
        "func_target_file_finder {:?} from file {:?}",
        func_name, current_file
    ));

    // First try: find only one function file
    if let Some(tf) = find_only_one_function_file(
        func_name,
        graph,
        source_start,
        current_file,
        source_node_type,
        operand,
        lang,
    ) {
        return Some(tf);
    }

    // Second try: find in the same file
    if let Some(tf) = find_function_in_same_file(func_name, current_file, graph, source_start) {
        return Some(tf);
    }

    if let Some(import_names) = import_names {
        if let Some(tf) = find_function_by_import(func_name, import_names, graph) {
            return Some(tf);
        }
    }

    // Fourth try: find in the same directory
    if let Some(tf) = find_function_in_same_directory(func_name, current_file, graph, source_start)
    {
        return Some(tf);
    }

    // Fifth try: If operand exists, try operand-based resolution
    if let Some(ref operand) = operand {
        if let Some(target_file) = find_function_with_operand(operand, func_name, graph) {
            if let Some(tf) = graph.find_node_by_name_and_file_contains(
                NodeType::Function,
                func_name,
                &target_file,
            ) {
                println!(
                    "[OPERAND] resolved {}.{} in {}",
                    operand, func_name, target_file
                );
                return Some(tf);
            }
        }
    }

    // Sixth try: Check if function is nested in a variable (e.g., authOptions.callbacks.signIn)
    if let Some(ref operand) = operand {
        if let Some(tf) = find_nested_function_in_variable(operand, func_name, graph, current_file)
        {
            return Some(tf);
        }
    }

    None
}

pub fn get_imports_for_file<G: Graph>(
    current_file: &str,
    lang: &Lang,
    graph: &G,
) -> Option<Vec<(String, Vec<String>)>> {
    let import_nodes = graph.find_nodes_by_file_ends_with(NodeType::Import, current_file);
    let import_node = import_nodes.first()?;
    let code = import_node.body.as_str();

    let imports_query = lang.lang().imports_query()?;
    let q = lang.q(&imports_query, &NodeType::Import);

    let tree = match lang.lang().parse(code, &NodeType::Import) {
        Ok(t) => t,
        Err(_) => return None,
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(&q, tree.root_node(), code.as_bytes());
    let mut results = Vec::new();

    while let Some(m) = matches.next() {
        let mut import_names = Vec::new();
        let mut import_source = None;
        let mut import_aliases = Vec::new();

        if Lang::loop_captures_multi(&q, m, code, |body, _node, o| {
            if o == IMPORTS_NAME {
                import_names.push(body);
            } else if o == IMPORTS_ALIAS {
                import_aliases.push(body);
            } else if o == IMPORTS_FROM {
                import_source = Some(trim_quotes(&body).to_string());
            }
            Ok(())
        })
        .is_err()
        {
            continue;
        }

        if !import_aliases.is_empty() {
            import_names = import_aliases;
        }

        if let Some(source_path) = import_source {
            let mut resolved_path = lang.lang().resolve_import_path(&source_path, current_file);

            if resolved_path.starts_with("@/") {
                resolved_path = resolved_path[2..].to_string();
            }

            let exts = lang.kind.exts();
            if let Some(ext) = exts.iter().find(|&&e| resolved_path.ends_with(e)) {
                resolved_path = resolved_path.trim_end_matches(ext).to_string();
            }

            results.push((resolved_path, import_names));
        }
    }

    if results.is_empty() {
        None
    } else {
        Some(results)
    }
}

fn find_function_by_import<G: Graph>(
    func_name: &str,
    import_names: Vec<(String, Vec<String>)>,
    graph: &G,
) -> Option<NodeData> {
    for (resolved_path, names) in import_names {
        if !names.iter().any(|n| n.as_str() == func_name) {
            continue;
        }

        if let Some(target) =
            graph.find_node_by_name_and_file_contains(NodeType::Function, func_name, &resolved_path)
        {
            if !target.body.is_empty() {
                log_cmd(format!(
                    "::: found function by import: {:?} (resolved: {:?})",
                    func_name, resolved_path
                ));
                return Some(target);
            }
        }
    }

    None
}

fn find_only_one_function_file<G: Graph>(
    func_name: &str,
    graph: &G,
    source_start: usize,
    current_file: &str,
    source_node_type: NodeType,
    operand: &Option<String>,
    lang: &Lang,
) -> Option<NodeData> {
    use lsp::Language;

    // For strict languages (TypeScript), skip the "find only one" heuristic
    // when an operand exists to avoid false positives with method calls
    if operand.is_some() && matches!(lang.kind, Language::Typescript) {
        return None;
    }
    let mut target_files_starts = Vec::new();
    let nodes = graph.find_nodes_by_name(NodeType::Function, func_name);
    if nodes.is_empty() {
        log_cmd(format!("::: found zero {:?}", func_name));
        return None;
    }
    for node in nodes {
        let is_same = node.start == source_start && node.file == current_file;
        // NOT empty functions (interfaces)
        if !node.body.is_empty() && (!is_same || source_node_type != NodeType::Function) {
            target_files_starts.push(node);
        }
    }

    if target_files_starts.len() == 1 {
        return Some(target_files_starts[0].clone());
    }
    // TODO: disclude "mock"
    log_cmd(format!("::: found more than one {:?}", func_name));
    target_files_starts.retain(|x| !x.file.contains("mock"));
    if target_files_starts.len() == 1 {
        log_cmd(format!("::: discluded mocks for!!! {:?}", func_name));
        return Some(target_files_starts[0].clone());
    }
    None
}

fn find_function_with_operand<G: Graph>(
    operand: &str,
    func_name: &str,
    graph: &G,
) -> Option<String> {
    let mut target_file = None;
    let mut instance = None;

    let operand_nodes = graph.find_nodes_by_name(NodeType::Instance, operand);
    if let Some(node) = operand_nodes.into_iter().next() {
        instance = Some(node.clone());
    }
    if let Some(i) = instance {
        if let Some(dt) = &i.data_type {
            let function_nodes = graph.find_nodes_by_name(NodeType::Function, func_name);
            for node in function_nodes {
                if node.meta.get("operand") == Some(dt) {
                    target_file = Some(node.file.clone());
                    break;
                }
            }
        }
    }
    target_file
}

fn find_nested_function_in_variable<G: Graph>(
    var_name: &str,
    func_name: &str,
    graph: &G,
    _current_file: &str,
) -> Option<NodeData> {
    let var_nodes = graph.find_nodes_by_name(NodeType::Var, var_name);
    if var_nodes.is_empty() {
        return None;
    }

    let func_nodes = graph.find_nodes_by_name(NodeType::Function, func_name);
    if func_nodes.is_empty() {
        return None;
    }

    for func in func_nodes.clone() {
        if let Some(nested_in) = func.meta.get("nested_in") {
            let n = crate::lang::parse::utils::trim_quotes(nested_in);
            let v = crate::lang::parse::utils::trim_quotes(var_name);
            if n == v {
                return Some(func);
            }
        }
    }

    None
}

fn find_function_in_same_file<G: Graph>(
    func_name: &str,
    current_file: &str,
    graph: &G,
    source_start: usize,
) -> Option<NodeData> {
    let node =
        graph.find_node_by_name_and_file_end_with(NodeType::Function, func_name, current_file);
    if let Some(node) = node {
        // dont return like Label->label
        if node.name != func_name && node.name.to_lowercase() == func_name.to_lowercase() {
            return None;
        }
        if !node.body.is_empty() && node.file == current_file && node.start != source_start {
            log_cmd(format!(
                "::: found function in same file: {:?}",
                current_file
            ));
            return Some(node);
        }
    }

    None
}

fn find_function_in_same_directory<G: Graph>(
    func_name: &str,
    current_file: &str,
    graph: &G,
    source_start: usize,
) -> Option<NodeData> {
    let current_dir = std::path::Path::new(current_file)
        .parent()
        .and_then(|p| p.to_str())?;

    let nodes = graph.find_nodes_by_name(NodeType::Function, func_name);
    let mut same_dir_files = Vec::new();

    log_cmd(format!(
        "::: dir found {:?} nodes name: {:?} file: {:?} in dir: {:?}",
        nodes.len(),
        func_name,
        current_file,
        current_dir
    ));
    for node in &nodes {
        // dont return like Label->label
        if node.name != func_name && node.name.to_lowercase() == func_name.to_lowercase() {
            return None;
        }
        if !node.body.is_empty() {
            if let Some(node_dir) = std::path::Path::new(&node.file)
                .parent()
                .and_then(|p| p.to_str())
            {
                if node_dir == current_dir && !node.file.contains("mock") {
                    let is_same = node.start == source_start && node.file == current_file;
                    if !is_same {
                        log_cmd(format!(
                            "::: found function in same directory! file: {:?}",
                            current_file
                        ));
                        same_dir_files.push(node);
                    }
                }
            }
        }
    }

    if same_dir_files.len() == 1 {
        log_cmd(format!(
            "::: found function in same directory: {:?}",
            current_dir
        ));
        return Some(same_dir_files[0].clone());
    }

    None
}

fn log_cmd(cmd: String) {
    // if cmd.contains("src/components/designer/bitcoin/BitcoinDetails.tsx") {
    //     tracing::info!("{}", cmd);
    // }
    tracing::debug!("{}", cmd);
}

fn _find_function_files<G: Graph>(func_name: &str, graph: &G) -> Vec<String> {
    let mut target_files = Vec::new();
    let function_nodes = graph.find_nodes_by_name(NodeType::Function, func_name);
    for node in function_nodes {
        if !node.body.is_empty() {
            target_files.push(node.file.clone());
        }
    }
    target_files
}

fn _pick_target_file_from_graph<G: Graph>(target_name: &str, graph: &G) -> Option<String> {
    let function_nodes = graph.find_nodes_by_name(NodeType::Function, target_name);
    function_nodes.first().map(|node| node.file.clone())
}
