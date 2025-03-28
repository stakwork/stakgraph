use serde::{Deserialize, Serialize};
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
}

pub const PROGRAMMING_LANGUAGES: [Language; 9] = [
    Language::Rust,
    Language::Go,
    Language::Typescript,
    Language::React,
    Language::Python,
    Language::Ruby,
    Language::Kotlin,
    Language::Swift,
    Language::Java,
];

impl Language {
    pub fn is_frontend(&self) -> bool {
        matches!(
            self,
            Self::Typescript | Self::React | Self::Kotlin | Self::Swift
        )
    }
    pub fn pkg_file(&self) -> &'static str {
        match self {
            Self::Rust => "Cargo.toml",
            Self::Go => "go.mod",
            Self::Typescript | Self::React => "package.json",
            Self::Python => "requirements.txt",
            Self::Ruby => "Gemfile",
            Self::Kotlin => "build.gradle.kts",
            Self::Swift => "Podfile",
            Self::Java => "pom.xml",
            Self::Bash => "",
            Self::Toml => "",
        }
    }

    pub fn exts(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["rs"],
            Self::Go => vec!["go"],
            Self::Typescript | Self::React => vec!["jsx", "tsx", "ts", "js"],
            Self::Python => vec!["py", "ipynb"],
            Self::Ruby => vec!["rb"],
            Self::Kotlin => vec!["kt", "kts"],
            Self::Swift => vec!["swift", "xcodeproj", "xcworkspace"],
            Self::Java => vec!["java", "gradle", "gradlew"],
            Self::Bash => vec!["sh"],
            Self::Toml => vec!["toml"],
        }
    }

    pub fn skip_dirs(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["target", ".git"],
            Self::Go => vec!["vendor", ".git"],
            Self::Typescript | Self::React => vec!["node_modules", ".git"],
            Self::Python => vec!["__pycache__", ".git", ".venv", "venv"],
            Self::Ruby => vec!["migrate", "tmp", ".git"],
            Self::Kotlin => vec![".gradle", ".idea", "build", ".git"],
            Self::Swift => vec![".git", "Pods"],
            Self::Java => vec![".gradle", ".idea", "build", ".git"],
            Self::Bash => vec![".git"],
            Self::Toml => vec![".git"],
        }
    }

    pub fn skip_file_ends(&self) -> Vec<&'static str> {
        match self {
            Self::Typescript | Self::React => vec![".min.js"],
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
        }
    }

    pub fn default_do_lsp(&self) -> bool {
        match self {
            Self::Rust => true,
            Self::Go => true,
            Self::Typescript | Self::React => true,
            Self::Python => false,
            Self::Ruby => false,
            Self::Kotlin => true,
            Self::Swift => true,
            Self::Java => true,
            Self::Bash => false,
            Self::Toml => false,
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
        }
        .to_string()
    }

    pub fn post_clone_cmd(&self) -> Vec<&'static str> {
        if std::env::var("LSP_SKIP_POST_CLONE").is_ok() {
            return Vec::new();
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
}

impl FromStr for Language {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
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
            _ => Err(anyhow::anyhow!("unsupported language")),
        }
    }
}
