use crate::coverage::types::{Language, LanguageDetector};
use std::path::Path;
use walkdir::WalkDir;


pub struct DefaultLanguageDetector;

impl LanguageDetector for DefaultLanguageDetector{
    fn detect(&self, repo_path: &Path) -> Vec<Language> {
        let mut languages: Vec<Language> = Vec::new();

        for entry in WalkDir::new(repo_path)
        .min_depth(1)
        .max_depth(4)
        .into_iter()
        .filter_entry(|e| !self.should_skip_entry(e))
        .filter_map(|e| e.ok()) {

            if !entry.file_type().is_file() {
                continue;
            }

            let file_name = entry.file_name().to_string_lossy().to_lowercase();

            if self.is_typescript_indicator(&file_name) && !languages.contains(&Language::TypeScript) {
                languages.push(Language::TypeScript);
            } 
             if self.is_rust_indicator(&file_name) && !languages.contains(&Language::Rust) {
                languages.push(Language::Rust);
            }
            if self.is_python_indicator(&file_name) && !languages.contains(&Language::Python) {
                languages.push(Language::Python);
            }

            if self.is_java_indicator(&file_name) && !languages.contains(&Language::Java) {
                languages.push(Language::Java);
            }
            if self.is_go_indicator(&file_name) && !languages.contains(&Language::Go) {
                languages.push(Language::Go);
            }
        }
        languages
    }
}


impl DefaultLanguageDetector{
    pub fn new() -> Self {
        Self
    }


    fn is_typescript_indicator(&self, file_name: &str) -> bool {
        matches!(file_name, "package.json" | "tsconfig.json" | "yarn.lock" | "pnpm-lock.yaml")
    }

    fn is_rust_indicator(&self, file_name: &str) -> bool {
        matches!(file_name, "Cargo.toml" | "Cargo.lock")
    }
    
    fn is_python_indicator(&self, file_name: &str) -> bool {
        matches!(file_name, "requirements.txt" | "pyproject.toml" | "setup.py" | "poetry.lock")
    }
    
    fn is_java_indicator(&self, file_name: &str) -> bool {
        matches!(file_name, "pom.xml" | "build.gradle" | "build.gradle.kts")
    }

    fn is_go_indicator(&self, file_name: &str) -> bool {
        matches!(file_name, "go.mod" | "go.sum")
    }
     fn should_skip_file(&self, file_name: &str) -> bool {
        if let Some(ext) = file_name.split('.').last() {
            return matches!(ext,
                "exe" | "dll" | "so" | "dylib" | "a" | "lib" |
                "png" | "jpg" | "jpeg" | "gif" | "ico" | "svg" |
                "zip" | "tar" | "gz" | "rar" | "7z" |
                "pdf" | "doc" | "docx" | "xls" | "xlsx"
            );
        }
        false
    }
     fn should_skip_directory(&self, dir_name: &str) -> bool {
        matches!(dir_name, 
            "node_modules" | "target" | "dist" | "build" | "out" | 
            ".git" | ".svn" | ".hg" | 
            "__pycache__" | ".pytest_cache" | "venv" | ".venv" |
            "coverage" | ".coverage" | "htmlcov" |
            ".idea" | ".vscode" | ".vs" |
            "bin" | "obj" | ".gradle"
        )
    }
     fn should_skip_entry(&self, entry: &walkdir::DirEntry) -> bool {
        let name = entry.file_name().to_string_lossy().to_lowercase();
        
        if name.starts_with('.') && name != ".github" {
            return true;
        }
        
        if entry.file_type().is_dir() && self.should_skip_directory(&name) {
            return true;
        }
        
        if entry.file_type().is_file() && self.should_skip_file(&name) {
            return true;
        }
        
        false
    }
}
