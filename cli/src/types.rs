use serde::Serialize;
use shared::Result;

use super::output::{write_json_success, Output, OutputMode};

const NODE_TYPES: &[&str] = &[
    "Repository",
    "Package",
    "Language",
    "Directory",
    "File",
    "Import",
    "Library",
    "Class",
    "Trait",
    "Instance",
    "Function",
    "Endpoint",
    "Request",
    "DataModel",
    "Concept",
    "Page",
    "Var",
    "UnitTest",
    "IntegrationTest",
    "E2eTest",
    "Mock",
];

const EDGE_TYPES: &[&str] = &[
    "Calls",
    "Uses",
    "Operand",
    "ArgOf",
    "Contains",
    "Imports",
    "Of",
    "Handler",
    "Includes",
    "Renders",
    "ParentOf",
    "Implements",
    "NestedIn",
];

#[derive(Serialize)]
struct TypesData {
    node_types: Vec<&'static str>,
    edge_types: Vec<&'static str>,
}

pub fn run(out: &mut Output, output_mode: OutputMode) -> Result<()> {
    if output_mode.is_json() {
        write_json_success(
            out,
            "types",
            TypesData {
                node_types: NODE_TYPES.to_vec(),
                edge_types: EDGE_TYPES.to_vec(),
            },
            Vec::new(),
        )?;
        return Ok(());
    }

    out.writeln("Node types:")?;
    for t in NODE_TYPES {
        out.writeln(format!("  {}", t))?;
    }
    out.newline()?;
    out.writeln("Edge types:")?;
    for t in EDGE_TYPES {
        out.writeln(format!("  {}", t))?;
    }
    Ok(())
}
