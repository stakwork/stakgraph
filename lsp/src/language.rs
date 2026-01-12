use crate::config::LanguageConfig;
use serde::{Deserialize, Serialize};
use shared::error::{Error, Result};
use std::{fmt::Display, str::FromStr};

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub enum Language {
    Bash,
    Toml,
    Rust,
    React,
    Go,
    Typescript,
    Python,
    Ruby,
    Kotlin,
    Swift,
    Java,
    Svelte,
    Angular,
    Cpp,
    Php,
}

pub const PROGRAMMING_LANGUAGES: [Language; 13] = [
    Language::Rust,
    Language::Go,
    Language::Typescript,
    Language::React,
    Language::Python,
    Language::Ruby,
    Language::Kotlin,
    Language::Swift,
    Language::Java,
    Language::Svelte,
    Language::Angular,
    Language::Cpp,
    Language::Php,
];

struct LanguageDefinition {
    language: Language,
    name: &'static str,
    exts: &'static [&'static str],
    pkg_files: &'static [&'static str],
    skip_dirs: &'static [&'static str],
    skip_file_ends: &'static [&'static str],
    lsp_exec: &'static str,
    lsp_args: &'static [&'static str],
    post_clone: &'static [&'static str],
    test_id_regex: Option<&'static str>,
}

const DEFINITIONS: &[LanguageDefinition] = &[
    LanguageDefinition {
        language: Language::Rust,
        name: "rust",
        exts: &["rs"],
        pkg_files: &["Cargo.toml"],
        skip_dirs: &["target", ".git"],
        skip_file_ends: &[],
        lsp_exec: "rust-analyzer",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Go,
        name: "go",
        exts: &["go"],
        pkg_files: &["go.mod"],
        skip_dirs: &["vendor", ".git"],
        skip_file_ends: &[],
        lsp_exec: "gopls",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Typescript,
        name: "typescript",
        exts: &["ts", "js"],
        pkg_files: &["package.json"],
        skip_dirs: &["node_modules", ".git"],
        skip_file_ends: &[".min.js"],
        lsp_exec: "typescript-language-server",
        lsp_args: &["--stdio"],
        post_clone: &["npm install --force"],
        test_id_regex: Some(r#"data-testid=(?:["']([^"']+)["']|\{['"`]([^'"`]+)['"`]\})"#),
    },
    LanguageDefinition {
        language: Language::React,
        name: "react",
        exts: &["jsx", "tsx", "mdx", "ts", "js", "html", "css"],
        pkg_files: &["package.json"],
        skip_dirs: &["node_modules", ".git"],
        skip_file_ends: &[".min.js"],
        lsp_exec: "typescript-language-server",
        lsp_args: &["--stdio"],
        post_clone: &["npm install --force"],
        test_id_regex: Some(r#"data-testid=(?:["']([^"']+)["']|\{['"`]([^'"`]+)['"`]\})"#),
    },
    LanguageDefinition {
        language: Language::Python,
        name: "python",
        exts: &["py", "ipynb"],
        pkg_files: &["requirements.txt"],
        skip_dirs: &["__pycache__", ".git", ".venv", "venv"],
        skip_file_ends: &[],
        lsp_exec: "pylsp",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: Some("get_by_test_id"),
    },
    LanguageDefinition {
        language: Language::Ruby,
        name: "ruby",
        exts: &["rb"],
        pkg_files: &["Gemfile"],
        skip_dirs: &["migrate", "tmp", ".git"],
        skip_file_ends: &[],
        lsp_exec: "ruby-lsp",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: Some(r#"get_by_test_id\(['"]([^'"]+)['"]\)"#),
    },
    LanguageDefinition {
        language: Language::Kotlin,
        name: "kotlin",
        exts: &["kt", "kts", "java"],
        pkg_files: &[".gradle.kts", ".gradle", ".properties"],
        skip_dirs: &["build", ".git"],
        skip_file_ends: &["gradlew"],
        lsp_exec: "kotlin-language-server",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Swift,
        name: "swift",
        exts: &["swift", "plist"],
        pkg_files: &["Podfile", "Cartfile"],
        skip_dirs: &[".git", "Pods"],
        skip_file_ends: &[],
        lsp_exec: "sourcekit-lsp",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Java,
        name: "java",
        exts: &["java", "gradle", "gradlew"],
        pkg_files: &["pom.xml"],
        skip_dirs: &[".idea", "build", ".git"],
        skip_file_ends: &[],
        lsp_exec: "jdtls",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Bash,
        name: "bash",
        exts: &["sh"],
        pkg_files: &[],
        skip_dirs: &[".git"],
        skip_file_ends: &[],
        lsp_exec: "",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Toml,
        name: "toml",
        exts: &["toml"],
        pkg_files: &[],
        skip_dirs: &[".git"],
        skip_file_ends: &[],
        lsp_exec: "",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Svelte,
        name: "svelte",
        exts: &["svelte", "ts", "js", "html", "css"],
        pkg_files: &["package.json"],
        skip_dirs: &[".git", " node_modules"],
        skip_file_ends: &[".config.ts", ".config.ts"],
        lsp_exec: "svelte-language-server",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Angular,
        name: "angular",
        exts: &["ts", "js", "html", "css"],
        pkg_files: &["package.json"],
        skip_dirs: &[".git", " node_modules"],
        skip_file_ends: &["spec.ts"],
        lsp_exec: "angular-language-server",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Cpp,
        name: "cpp",
        exts: &["cpp", "h"],
        pkg_files: &["CMakeLists.txt"],
        skip_dirs: &[".git", "build", "out", "CMakeFiles"],
        skip_file_ends: &[],
        lsp_exec: "",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
    LanguageDefinition {
        language: Language::Php,
        name: "php",
        exts: &["php"],
        pkg_files: &["composer.json"],
        skip_dirs: &[".git", "vendor"],
        skip_file_ends: &[],
        lsp_exec: "",
        lsp_args: &[],
        post_clone: &[],
        test_id_regex: None,
    },
];

impl Language {
    fn def(&self) -> &'static LanguageDefinition {
        DEFINITIONS
            .iter()
            .find(|d| d.language == *self)
            .expect("Language definition not found")
    }

    pub fn is_frontend(&self) -> bool {
        matches!(
            self,
            Self::Typescript | Self::React | Self::Kotlin | Self::Swift
        )
    }
    pub fn pkg_files(&self) -> Vec<&'static str> {
        self.def().pkg_files.to_vec()
    }

    pub fn exts(&self) -> Vec<&'static str> {
        self.def().exts.to_vec()
    }

    // React overrides Typescript if detected
    pub fn overrides(&self) -> Vec<Language> {
        match self {
            Self::React => vec![Self::Typescript, Self::Svelte, Self::Angular],
            Self::Svelte => vec![Self::Typescript, Self::Angular],
            Self::Angular => vec![Self::Typescript],
            _ => Vec::new(),
        }
    }

    pub fn skip_dirs(&self) -> Vec<&'static str> {
        self.def().skip_dirs.to_vec()
    }

    pub fn skip_file_ends(&self) -> Vec<&'static str> {
        self.def().skip_file_ends.to_vec()
    }

    pub fn only_include_files(&self) -> Vec<&'static str> {
        Vec::new()
    }

    pub fn default_do_lsp(&self) -> bool {
        if let Ok(use_lsp) = std::env::var("USE_LSP") {
            if use_lsp == "true" || use_lsp == "1" {
                return !self.def().lsp_exec.is_empty();
            }
        }
        false
    }

    pub fn lsp_exec(&self) -> String {
        self.def().lsp_exec.to_string()
    }

    pub fn version_arg(&self) -> String {
        if self.lsp_exec().is_empty() {
            "".to_string()
        } else {
            "--version".to_string()
        }
    }

    pub fn lsp_args(&self) -> Vec<String> {
        self.def().lsp_args.iter().map(|s| s.to_string()).collect()
    }

    pub fn to_config(&self) -> LanguageConfig {
        let def = self.def();
        LanguageConfig {
            name: def.name.to_string(),
            file_extensions: def.exts.iter().map(|s| s.to_string()).collect(),
            lsp_executable: def.lsp_exec.to_string(),
            lsp_args: def.lsp_args.iter().map(|s| s.to_string()).collect(),
            package_files: def.pkg_files.iter().map(|s| s.to_string()).collect(),
            root_markers: def.skip_dirs.iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn post_clone_cmd(&self, use_lsp: bool) -> Vec<&'static str> {
        if let Ok(lsp_skip) = std::env::var("LSP_SKIP_POST_CLONE") {
            if lsp_skip == "true" || lsp_skip == "1" {
                return Vec::new();
            }
        }
        if let Ok(repo_path) = std::env::var("REPO_PATH") {
            if !repo_path.is_empty() {
                tracing::info!("skipping post clone cmd for local repo. If its a js/ts repo, run npm install first!");
                return Vec::new();
            }
        }
        if !use_lsp {
            return Vec::new();
        }
        if let Ok(use_lsp_env) = std::env::var("USE_LSP") {
            if use_lsp_env == "false" || use_lsp_env == "0" {
                return Vec::new();
            }
        }
        self.def().post_clone.to_vec()
    }

    pub fn test_id_regex(&self) -> Option<&'static str> {
        self.def().test_id_regex
    }
    pub fn is_package_file(&self, file_name: &str) -> bool {
        self.pkg_files()
            .iter()
            .any(|pkg_file| file_name.ends_with(pkg_file))
    }
    pub fn is_from_language(&self, file: &str) -> bool {
        if !file.contains('.') {
            return true;
        }
        if let Some(ext) = file.split('.').next_back() {
            self.exts().contains(&ext) || self.is_package_file(file)
        } else {
            true //dirs, lang, repo
        }
    }
    pub fn is_source_file(&self, file_name: &str) -> bool {
        if self.is_package_file(file_name) {
            return true;
        }

        if let Some(ext) = file_name.split('.').next_back() {
            self.exts().contains(&ext)
        } else {
            false
        }
    }
    pub fn from_path(path: &str) -> Option<Self> {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        for lang in PROGRAMMING_LANGUAGES.iter() {
            if lang.exts().iter().any(|e| e.eq_ignore_ascii_case(&ext)) {
                return Some(lang.clone());
            }
        }
        None
    }
}

impl Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Rust => "rust",
            Self::Go => "go",
            Self::Typescript => "typescript",
            Self::React => "react",
            Self::Python => "python",
            Self::Ruby => "ruby",
            Self::Kotlin => "kotlin",
            Self::Swift => "swift",
            Self::Java => "java",
            Self::Bash => "bash",
            Self::Toml => "toml",
            Self::Svelte => "svelte",
            Self::Angular => "angular",
            Self::Cpp => "cpp",
            Self::Php => "php",
        };
        write!(f, "{}", s)
    }
}

impl FromStr for Language {
    type Err = shared::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "python" => Ok(Language::Python),
            "Python" => Ok(Language::Python),
            "go" => Ok(Language::Go),
            "Go" => Ok(Language::Go),
            "golang" => Ok(Language::Go),
            "Golang" => Ok(Language::Go),
            "react" => Ok(Language::React),
            "React" => Ok(Language::React),
            "tsx" => Ok(Language::React),
            "jsx" => Ok(Language::React),
            "ts" => Ok(Language::Typescript),
            "js" => Ok(Language::Typescript),
            "typescript" => Ok(Language::Typescript),
            "TypeScript" => Ok(Language::Typescript),
            "javascript" => Ok(Language::Typescript),
            "JavaScript" => Ok(Language::Typescript),
            "ruby" => Ok(Language::Ruby),
            "Ruby" => Ok(Language::Ruby),
            "RubyOnRails" => Ok(Language::Ruby),
            "rust" => Ok(Language::Rust),
            "Rust" => Ok(Language::Rust),
            "bash" => Ok(Language::Bash),
            "Bash" => Ok(Language::Bash),
            "toml" => Ok(Language::Toml),
            "Toml" => Ok(Language::Toml),
            "kotlin" => Ok(Language::Kotlin),
            "Kotlin" => Ok(Language::Kotlin),
            "swift" => Ok(Language::Swift),
            "Swift" => Ok(Language::Swift),
            "java" => Ok(Language::Java),
            "Java" => Ok(Language::Java),
            "svelte" => Ok(Language::Svelte),
            "Svelte" => Ok(Language::Svelte),
            "angular" => Ok(Language::Angular),
            "Angular" => Ok(Language::Angular),
            "cpp" => Ok(Language::Cpp),
            "Cpp" => Ok(Language::Cpp),
            "c++" => Ok(Language::Cpp),
            "C++" => Ok(Language::Cpp),
            "php" => Ok(Language::Php),
            "Php" => Ok(Language::Php),
            "PHP" => Ok(Language::Php),

            _ => Err(Error::Custom("unsupported language".to_string())),
        }
    }
}

pub fn common_binary_exts() -> Vec<&'static str> {
    vec![
        "png", "jpg", "jpeg", "gif", "bmp", "svg", "ico", "tif", "tiff", "webp", "mp4", "mov",
        "avi", "mkv", "webm", "mp3", "wav", "ogg", "flac", "ttf", "otf", "woff", "woff2", "zip",
        "rar", "7z", "tar", "gz", "deb", "pkg", "dmg", "pdf", "doc", "docx", "xls", "xlsx", "ppt",
        "pptx", "exe", "dll", "so", "a", "o", "jar", "class", "pyc",
    ]
}
pub fn junk_directories() -> Vec<&'static str> {
    vec![
        ".git",
        ".vscode",
        "target",
        "build",
        "dist",
        "venv",
        "node_modules",
    ]
}
