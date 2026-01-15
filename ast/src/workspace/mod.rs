use lsp::language::Language;
use shared::error::Result;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct WorkspacePackage {
    pub path: PathBuf,
    pub name: String,
    pub language: Language,
}

pub fn detect_workspaces(root: &Path) -> Result<Option<Vec<WorkspacePackage>>> {
    let mut packages = scan_children(root)?;

    if let Some(root_pkg) = try_as_package(root) {
        let subpackage_langs: Vec<_> = packages.iter().map(|p| p.language.clone()).collect();
        if !subpackage_langs.contains(&root_pkg.language) {
            packages.push(root_pkg);
        }
    }

    packages.sort_by(|a, b| a.path.cmp(&b.path));
    packages.dedup_by(|a, b| a.path == b.path);

    if packages.len() >= 2 {
        Ok(Some(packages))
    } else {
        Ok(None)
    }
}

fn scan_children(dir: &Path) -> Result<Vec<WorkspacePackage>> {
    let mut packages = Vec::new();
    for entry in std::fs::read_dir(dir)?.flatten() {
        let path = entry.path();
        if path.is_dir() && !is_skip_dir(&path) {
            packages.extend(scan_packages(&path, 1)?);
        }
    }
    Ok(packages)
}

fn scan_packages(dir: &Path, depth: usize) -> Result<Vec<WorkspacePackage>> {
    let mut packages = Vec::new();
    if let Some(pkg) = try_as_package(dir) {
        packages.push(pkg);
    }
    if depth < 3 {
        for entry in std::fs::read_dir(dir)?.flatten() {
            let path = entry.path();
            if path.is_dir() && !is_skip_dir(&path) {
                packages.extend(scan_packages(&path, depth + 1)?);
            }
        }
    }
    Ok(packages)
}

fn try_as_package(dir: &Path) -> Option<WorkspacePackage> {
    let lang = detect_language(dir)?;
    if !is_actual_package(dir, &lang) {
        return None;
    }
    let name = dir.file_name()?.to_str()?.to_string();
    let path = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    Some(WorkspacePackage {
        path,
        name,
        language: lang,
    })
}

fn detect_language(dir: &Path) -> Option<Language> {
    if dir.join("Cargo.toml").exists() {
        return Some(Language::Rust);
    }
    if dir.join("go.mod").exists() {
        return Some(Language::Go);
    }
    if dir.join("package.json").exists() {
        return Some(classify_js_package(dir));
    }
    if dir.join("requirements.txt").exists()
        || dir.join("setup.py").exists()
        || dir.join("pyproject.toml").exists()
    {
        return Some(Language::Python);
    }
    if dir.join("Gemfile").exists() {
        return Some(Language::Ruby);
    }
    if dir.join("composer.json").exists() {
        return Some(Language::Php);
    }
    if dir.join("pom.xml").exists() {
        return Some(Language::Java);
    }
    None
}

fn is_actual_package(dir: &Path, lang: &Language) -> bool {
    match lang {
        Language::Rust => std::fs::read_to_string(dir.join("Cargo.toml"))
            .map(|c| c.contains("[package]"))
            .unwrap_or(false),
        Language::Typescript | Language::React | Language::Angular | Language::Svelte => {
            let has_workspaces = std::fs::read_to_string(dir.join("package.json"))
                .ok()
                .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
                .and_then(|j| j.get("workspaces").cloned())
                .is_some();
            !has_workspaces
        }
        Language::Python => {
            if dir.join("pyproject.toml").exists() {
                return std::fs::read_to_string(dir.join("pyproject.toml"))
                    .map(|c| c.contains("[project]") || c.contains("[tool.poetry]"))
                    .unwrap_or(false);
            }
            true
        }
        _ => true,
    }
}

fn is_skip_dir(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    matches!(
        name,
        ".git" | "node_modules" | "target" | "vendor" | "build" | "dist" | ".venv" | "venv"
    )
}

fn classify_js_package(dir: &Path) -> Language {
    if let Ok(content) = std::fs::read_to_string(dir.join("package.json")) {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(deps) = json.get("dependencies") {
                if deps.get("react").is_some()
                    || deps.get("next").is_some()
                    || deps.get("vue").is_some()
                {
                    return Language::React;
                }
                if deps.get("@angular/core").is_some() {
                    return Language::Angular;
                }
                if deps.get("svelte").is_some() {
                    return Language::Svelte;
                }
            }
        }
    }
    Language::Typescript
}
