use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageConfig {
    pub name: String,
    pub file_extensions: Vec<String>,
    pub lsp_executable: String,
    pub lsp_args: Vec<String>,
    pub package_files: Vec<String>,
    pub root_markers: Vec<String>,
}
