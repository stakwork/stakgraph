use lsp::language::{Language, PROGRAMMING_LANGUAGES};
use shared::error::Result;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct WorkspacePackage {
    pub path: PathBuf,
    pub name: String,
    pub language: Language,
}

pub fn detect_workspaces(root: &Path) -> Result<Option<Vec<WorkspacePackage>>> {
    if let Some(pkgs) = try_cargo_workspace(root)? {
        return Ok(Some(pkgs));
    }

    if let Some(pkgs) = try_go_workspace(root)? {
        return Ok(Some(pkgs));
    }

    if let Some(pkgs) = try_npm_workspace(root)? {
        return Ok(Some(pkgs));
    }

    if let Some(pkgs) = try_multi_package_scan(root)? {
        return Ok(Some(pkgs));
    }

    Ok(None)
}

fn try_cargo_workspace(root: &Path) -> Result<Option<Vec<WorkspacePackage>>> {
    let cargo_toml = root.join("Cargo.toml");
    if !cargo_toml.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&cargo_toml)?;
    if !content.contains("[workspace]") {
        return Ok(None);
    }

    let mut packages = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("members") {
            if let Some(start) = line.find('[') {
                if let Some(end) = line.find(']') {
                    let members_str = &line[start + 1..end];
                    for member in members_str.split(',') {
                        let member = member.trim().trim_matches('"').trim_matches('\'');
                        if !member.is_empty() {
                            let pkg_path = root.join(member);
                            if pkg_path.exists() {
                                packages.push(WorkspacePackage {
                                    path: pkg_path.canonicalize().unwrap_or(pkg_path),
                                    name: member.to_string(),
                                    language: Language::Rust,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    if packages.is_empty() {
        Ok(None)
    } else {
        Ok(Some(packages))
    }
}

fn try_go_workspace(root: &Path) -> Result<Option<Vec<WorkspacePackage>>> {
    let go_work = root.join("go.work");
    if !go_work.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&go_work)?;
    let mut packages = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.starts_with("./") || line.starts_with("use") {
            let path_str = line
                .trim_start_matches("use")
                .trim()
                .trim_start_matches("./")
                .trim_end_matches(')')
                .trim();

            if !path_str.is_empty() && !path_str.starts_with('(') {
                let pkg_path = root.join(path_str);
                if pkg_path.exists() {
                    packages.push(WorkspacePackage {
                        path: pkg_path.canonicalize().unwrap_or(pkg_path),
                        name: path_str.to_string(),
                        language: Language::Go,
                    });
                }
            }
        }
    }

    if packages.is_empty() {
        Ok(None)
    } else {
        Ok(Some(packages))
    }
}

fn try_npm_workspace(root: &Path) -> Result<Option<Vec<WorkspacePackage>>> {
    let package_json = root.join("package.json");
    if !package_json.exists() {
        return Ok(None);
    }

    let content = std::fs::read_to_string(&package_json)?;
    let json: serde_json::Value = serde_json::from_str(&content)?;

    let workspaces = match json.get("workspaces") {
        Some(serde_json::Value::Array(arr)) => arr,
        _ => return Ok(None),
    };

    let mut packages = Vec::new();

    for ws in workspaces {
        if let Some(pattern) = ws.as_str() {
            let expanded = expand_glob(root, pattern);
            for pkg_path in expanded {
                if pkg_path.exists() {
                    let lang = classify_package(&pkg_path);
                    let name = pkg_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    packages.push(WorkspacePackage {
                        path: pkg_path.canonicalize().unwrap_or(pkg_path),
                        name,
                        language: lang,
                    });
                }
            }
        }
    }

    if root.join("go.mod").exists() {
        packages.push(WorkspacePackage {
            path: root.canonicalize().unwrap_or_else(|_| root.to_path_buf()),
            name: "root".to_string(),
            language: Language::Go,
        });
    }

    scan_non_js_packages(root, &mut packages)?;

    if packages.is_empty() {
        Ok(None)
    } else {
        Ok(Some(packages))
    }
}

fn scan_non_js_packages(root: &Path, packages: &mut Vec<WorkspacePackage>) -> Result<()> {
    let existing_paths: Vec<_> = packages.iter().map(|p| p.path.clone()).collect();

    for entry in std::fs::read_dir(root)?.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        for subentry in std::fs::read_dir(&path).into_iter().flatten().flatten() {
            let subpath = subentry.path();
            if !subpath.is_dir() {
                continue;
            }

            let is_non_js = subpath.join("requirements.txt").exists()
                || subpath.join("setup.py").exists()
                || (subpath.join("Cargo.toml").exists() && !subpath.join("package.json").exists())
                || (subpath.join("go.mod").exists() && !subpath.join("package.json").exists());

            if is_non_js {
                let canonical = subpath.canonicalize().unwrap_or_else(|_| subpath.clone());
                if !existing_paths.contains(&canonical) {
                    let lang = classify_package(&subpath);
                    let name = subpath
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    packages.push(WorkspacePackage {
                        path: canonical,
                        name,
                        language: lang,
                    });
                }
            }
        }
    }

    Ok(())
}

fn try_multi_package_scan(root: &Path) -> Result<Option<Vec<WorkspacePackage>>> {
    let mut packages = Vec::new();

    let entries = std::fs::read_dir(root)?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if has_package_file(&path) {
                let lang = classify_package(&path);
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                packages.push(WorkspacePackage {
                    path: path.canonicalize().unwrap_or_else(|_| path.clone()),
                    name,
                    language: lang,
                });
            }

            for subentry in std::fs::read_dir(&path).into_iter().flatten().flatten() {
                let subpath = subentry.path();
                if subpath.is_dir() && has_package_file(&subpath) {
                    let lang = classify_package(&subpath);
                    let name = subpath
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    packages.push(WorkspacePackage {
                        path: subpath.canonicalize().unwrap_or(subpath),
                        name,
                        language: lang,
                    });
                }
            }
        }
    }

    if packages.len() >= 2 {
        Ok(Some(packages))
    } else {
        Ok(None)
    }
}

fn has_package_file(path: &Path) -> bool {
    for lang in &PROGRAMMING_LANGUAGES {
        for pkg_file in lang.pkg_files() {
            if path.join(pkg_file).exists() {
                return true;
            }
        }
    }
    path.join("setup.py").exists() || path.join("pyproject.toml").exists()
}

fn classify_package(path: &Path) -> Language {
    if path.join("Cargo.toml").exists() {
        return Language::Rust;
    }
    if path.join("go.mod").exists() {
        return Language::Go;
    }
    if path.join("requirements.txt").exists()
        || path.join("setup.py").exists()
        || path.join("pyproject.toml").exists()
    {
        return Language::Python;
    }
    if path.join("Gemfile").exists() {
        return Language::Ruby;
    }
    if path.join("composer.json").exists() {
        return Language::Php;
    }
    if path.join("pom.xml").exists() {
        return Language::Java;
    }

    if let Ok(content) = std::fs::read_to_string(path.join("package.json")) {
        return classify_from_package_json(&content);
    }

    Language::Typescript
}

fn classify_from_package_json(content: &str) -> Language {
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(content) {
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
    Language::Typescript
}

fn expand_glob(root: &Path, pattern: &str) -> Vec<PathBuf> {
    let mut results = Vec::new();

    if pattern.ends_with("/*") {
        let dir = pattern.trim_end_matches("/*");
        let dir_path = root.join(dir);
        if dir_path.exists() {
            if let Ok(entries) = std::fs::read_dir(&dir_path) {
                for entry in entries.flatten() {
                    if entry.path().is_dir() {
                        results.push(entry.path());
                    }
                }
            }
        }
    } else {
        let path = root.join(pattern);
        if path.exists() {
            results.push(path);
        }
    }

    results
}
