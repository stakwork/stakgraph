use shared::error::Result;
use crate::language::{Language, PROGRAMMING_LANGUAGES};
use std::path::Path;
use std::fs;

pub fn detect_workspace_type(root: &str) -> Option<Language> {
    let root_path = Path::new(root);
    
    for lang in PROGRAMMING_LANGUAGES {
        let workspace_files = lang.workspace_files();
        if workspace_files.is_empty() {
            continue;
        }
        
        for workspace_file in workspace_files {
            let file_path = root_path.join(workspace_file);
            if file_path.exists() {
                return Some(lang);
            }
        }
    }
    
    None
}

pub fn parse_workspace_config(root: &str, lang: Language) -> Result<Vec<String>> {
    let root_path = Path::new(root);
    let config_key = lang.workspace_config_key();
    
    if config_key.is_none() {
        return Ok(Vec::new());
    }
    
    let key = config_key.unwrap();
    
    match lang {
        Language::Typescript | Language::React | Language::Svelte | Language::Angular => {
            let pnpm_workspace = root_path.join("pnpm-workspace.yaml");
            if pnpm_workspace.exists() {
                return parse_pnpm_workspace(pnpm_workspace.to_str().unwrap(), key);
            }
            
            let package_json = root_path.join("package.json");
            if package_json.exists() {
                return parse_npm_workspace(package_json.to_str().unwrap(), key);
            }
            
            Ok(Vec::new())
        }
        Language::Rust => {
            let cargo_toml = root_path.join("Cargo.toml");
            if cargo_toml.exists() {
                return parse_cargo_workspace(cargo_toml.to_str().unwrap(), key);
            }
            Ok(Vec::new())
        }
        _ => Ok(Vec::new()),
    }
}

fn parse_pnpm_workspace(path: &str, key: &str) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)?;
    let yaml: serde_yaml::Value = serde_yaml::from_str(&contents)
        .map_err(|e| shared::Error::Custom(format!("Failed to parse YAML: {}", e)))?;
    
    if let Some(packages) = yaml.get(key) {
        if let Some(array) = packages.as_sequence() {
            let patterns: Vec<String> = array
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            return Ok(patterns);
        }
    }
    
    Ok(Vec::new())
}

fn parse_npm_workspace(path: &str, key: &str) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)?;
    let json: serde_json::Value = serde_json::from_str(&contents)
        .map_err(|e| shared::Error::Custom(format!("Failed to parse JSON: {}", e)))?;
    
    if let Some(workspaces) = json.get("workspaces") {
        if let Some(array) = workspaces.as_array() {
            let patterns: Vec<String> = array
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            return Ok(patterns);
        }
        
        if let Some(packages) = workspaces.get(key) {
            if let Some(array) = packages.as_array() {
                let patterns: Vec<String> = array
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                return Ok(patterns);
            }
        }
    }
    
    Ok(Vec::new())
}

fn parse_cargo_workspace(path: &str, key: &str) -> Result<Vec<String>> {
    let contents = fs::read_to_string(path)?;
    let toml: toml::Value = toml::from_str(&contents)
        .map_err(|e| shared::Error::Custom(format!("Failed to parse TOML: {}", e)))?;
    
    if let Some(workspace) = toml.get("workspace") {
        if let Some(members) = workspace.get(key) {
            if let Some(array) = members.as_array() {
                let patterns: Vec<String> = array
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                return Ok(patterns);
            }
        }
    }
    
    Ok(Vec::new())
}

pub fn expand_patterns(root: &str, patterns: Vec<String>) -> Vec<String> {
    let mut result = Vec::new();
    let root_path = Path::new(root);
    
    for pattern in patterns {
        if pattern.contains('*') {
            let full_pattern = root_path.join(&pattern);
            if let Some(pattern_str) = full_pattern.to_str() {
                if let Ok(paths) = glob::glob(pattern_str) {
                    for entry in paths.flatten() {
                        if entry.is_dir() {
                            if let Some(path_str) = entry.to_str() {
                                result.push(path_str.to_string());
                            }
                        }
                    }
                }
            }
        } else {
            let full_path = root_path.join(&pattern);
            if full_path.is_dir() {
                if let Some(path_str) = full_path.to_str() {
                    result.push(path_str.to_string());
                }
            }
        }
    }
    
    result
}

pub fn find_workspace_packages(root: &str) -> Vec<String> {
    let workspace_lang = match detect_workspace_type(root) {
        Some(lang) => lang,
        None => return Vec::new(),
    };
    
    let patterns = match parse_workspace_config(root, workspace_lang) {
        Ok(patterns) => patterns,
        Err(_) => return Vec::new(),
    };
    
    if patterns.is_empty() {
        return Vec::new();
    }
    
    expand_patterns(root, patterns)
}

pub fn get_relative_path(root: &str, full_path: &str) -> String {
    let root_path = Path::new(root);
    let full = Path::new(full_path);
    
    if let Ok(rel) = full.strip_prefix(root_path) {
        rel.to_str().unwrap_or("").to_string()
    } else {
        full_path.to_string()
    }
}

pub fn is_workspace(root: &str) -> bool {
    detect_workspace_type(root).is_some()
}

pub fn has_framework_dependency(root: &str, lang: &Language) -> bool {
    let package_json = Path::new(root).join("package.json");
    if !package_json.exists() {
        return false;
    }
    
    let Ok(contents) = fs::read_to_string(&package_json) else {
        return false;
    };
    
    let Ok(json): std::result::Result<serde_json::Value, _> = serde_json::from_str(&contents) else {
        return false;
    };
    
    let framework_deps = lang.framework_dependencies();
    if framework_deps.is_empty() {
        return true;
    }
    
    for dep_key in ["dependencies", "devDependencies"] {
        if let Some(deps) = json.get(dep_key).and_then(|v| v.as_object()) {
            for framework_dep in &framework_deps {
                if deps.contains_key(*framework_dep) {
                    return true;
                }
            }
        }
    }
    
    false
}
