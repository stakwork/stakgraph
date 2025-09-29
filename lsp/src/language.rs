use serde::{Deserialize, Serialize};
use shared::error::{Error, Result};
use std::str::FromStr;

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
}

pub const PROGRAMMING_LANGUAGES: [Language; 12] = [
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
];

impl Language {
    pub fn is_frontend(&self) -> bool {
        matches!(
            self,
            Self::Typescript | Self::React | Self::Kotlin | Self::Swift
        )
    }
    pub fn pkg_files(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["Cargo.toml"],
            Self::Go => vec!["go.mod"],
            Self::Typescript | Self::React => vec!["package.json"],
            Self::Python => vec!["requirements.txt"],
            Self::Ruby => vec!["Gemfile"],
            Self::Kotlin => vec![".gradle.kts", ".gradle", ".properties"],
            Self::Swift => vec!["Podfile", "Cartfile"],
            Self::Java => vec!["pom.xml"],
            Self::Bash => vec![],
            Self::Toml => vec![],
            Self::Svelte => vec!["package.json"],
            Self::Angular => vec!["package.json"],
            Self::Cpp => vec!["CMakeLists.txt"],
        }
    }

    pub fn exts(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["rs"],
            Self::Go => vec!["go"],
            Self::Python => vec!["py", "ipynb"],
            Self::Ruby => vec!["rb"],
            Self::Kotlin => vec!["kt", "kts", "java"],
            Self::Swift => vec!["swift", "plist"],
            Self::Java => vec!["java", "gradle", "gradlew"],
            Self::Bash => vec!["sh"],
            Self::Toml => vec!["toml"],
            // how to separate ts and js?
            Self::Typescript => vec!["ts", "js"],
            Self::React => vec!["jsx", "tsx", "mdx","ts", "js", "html", "css"],
            Self::Svelte => vec!["svelte", "ts", "js", "html", "css"],
            Self::Angular => vec!["ts", "js", "html", "css"],
            Self::Cpp => vec!["cpp", "h"],
        }
    }

    // React overrides Typescript if detected
    pub fn overrides(&self) -> Vec<Language> {
        match self {
            Self::React => vec![Self::Typescript, Self::Svelte, Self::Angular],
            Self::Svelte => vec![Self::Typescript],
            Self::Angular => vec![Self::Typescript],
            _ => Vec::new(),
        }
    }

    pub fn skip_dirs(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["target", ".git"],
            Self::Go => vec!["vendor", ".git"],
            Self::Typescript | Self::React => vec!["node_modules", ".git"],
            Self::Python => vec!["__pycache__", ".git", ".venv", "venv"],
            Self::Ruby => vec!["migrate", "tmp", ".git"],
            Self::Kotlin => vec!["build", ".git"],
            Self::Swift => vec![".git", "Pods"],
            Self::Java => vec![".idea", "build", ".git"],
            Self::Bash => vec![".git"],
            Self::Toml => vec![".git"],
            Self::Svelte => vec![".git", " node_modules"],
            Self::Angular => vec![".git", " node_modules"],
            Self::Cpp => vec![".git", "build", "out", "CMakeFiles"],
        }
    }

    pub fn skip_file_ends(&self) -> Vec<&'static str> {
        match self {
            Self::Typescript | Self::React => vec![".min.js"],
            Self::Svelte => vec![".config.ts", ".config.ts"],
            Self::Angular => vec!["spec.ts"],
            Self::Kotlin => vec!["gradlew"],
            _ => Vec::new(),
        }
    }

    pub fn only_include_files(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript | Self::React => Vec::new(),
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Kotlin => Vec::new(),
            Self::Swift => Vec::new(),
            Self::Java => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
            Self::Svelte => Vec::new(),
            Self::Angular => Vec::new(),
            Self::Cpp => Vec::new(),
        }
    }

    pub fn default_do_lsp(&self) -> bool {
        if let Ok(use_lsp) = std::env::var("USE_LSP") {
            if use_lsp == "false" || use_lsp == "0" {
                return false;
            }
        }
        match self {
            Self::Rust => true,
            Self::Go => true,
            Self::Typescript => true,
            Self::React => true,
            Self::Python => false,
            Self::Ruby => false,
            Self::Kotlin => false,
            Self::Swift => false,
            Self::Java => true,
            Self::Bash => false,
            Self::Toml => false,
            Self::Svelte => false,
            Self::Angular => false,
            Self::Cpp => false,
        }
    }

    pub fn lsp_exec(&self) -> String {
        match self {
            Self::Rust => "rust-analyzer",
            Self::Go => "gopls",
            Self::Typescript | Self::React => "typescript-language-server",
            Self::Python => "pylsp",
            Self::Ruby => "ruby-lsp",
            Self::Kotlin => "kotlin-language-server",
            Self::Swift => "sourcekit-lsp",
            Self::Java => "jdtls",
            Self::Bash => "",
            Self::Toml => "",
            Self::Svelte => "svelte-language-server",
            Self::Angular => "angular-language-server",
            Self::Cpp => "",
        }
        .to_string()
    }

    pub fn version_arg(&self) -> String {
        match self {
            Self::Rust => "--version",
            Self::Go => "version",
            Self::Typescript | Self::React => "--version",
            Self::Python => "--version",
            Self::Ruby => "--version",
            Self::Kotlin => "--version",
            Self::Swift => "--version",
            Self::Java => "--version",
            Self::Bash => "",
            Self::Toml => "",
            Self::Svelte => "--version",
            Self::Angular => "--version",
            Self::Cpp => "--version",
        }
        .to_string()
    }

    pub fn lsp_args(&self) -> Vec<String> {
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript | Self::React => vec!["--stdio".to_string()],
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Kotlin => Vec::new(),
            Self::Swift => Vec::new(),
            Self::Java => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
            Self::Svelte => Vec::new(),
            Self::Angular => Vec::new(),
            Self::Cpp => Vec::new(),
        }
    }

    pub fn to_string(&self) -> String {
        match self {
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
        }
        .to_string()
    }

    pub fn post_clone_cmd(&self) -> Vec<&'static str> {
        if let Ok(use_lsp) = std::env::var("USE_LSP") {
            if use_lsp == "false" || use_lsp == "0" {
                return Vec::new();
            }
        }
        if let Ok(lsp_skip) = std::env::var("LSP_SKIP_POST_CLONE") {
            if lsp_skip == "true" || lsp_skip == "1" {
                return Vec::new();
            }
        }
        // for local repo, assume its already cloned
        if let Ok(repo_path) = std::env::var("REPO_PATH") {
            if !repo_path.is_empty() {
                tracing::info!("skipping post clone cmd for local repo. If its a js/ts repo, run npm install first!");
                return Vec::new();
            }
        }
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript | Self::React => vec!["npm install --force"],
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Kotlin => Vec::new(),
            Self::Swift => Vec::new(),
            Self::Java => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
            Self::Svelte => Vec::new(),
            Self::Angular => Vec::new(),
            Self::Cpp => Vec::new(),
        }
    }

    pub fn test_id_regex(&self) -> Option<&'static str> {
        match self {
            Self::Typescript | Self::React => {
                Some(r#"data-testid=(?:["']([^"']+)["']|\{['"`]([^'"`]+)['"`]\})"#)
            }
            Self::Python => Some("get_by_test_id"),
            Self::Ruby => Some(r#"get_by_test_id\(['"]([^'"]+)['"]\)"#),
            _ => None,
        }
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
        if let Some(ext) = file.split('.').last() {
            if self.exts().contains(&ext) || self.is_package_file(file) {
                return true;
            } else {
                return false;
            }
        } else {
            return true; //dirs, lang, repo
        }
    }
    pub fn is_source_file(&self, file_name: &str) -> bool {
        if self.is_package_file(file_name) {
            return true;
        }

        if let Some(ext) = file_name.split('.').last() {
            if self.exts().contains(&ext) {
                return true;
            } else {
                return false;
            }
        }
        false
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

            _ => Err(Error::Custom(format!("unsupported language"))),
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
