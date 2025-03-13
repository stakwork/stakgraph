use serde::{Deserialize, Serialize};
use std::str::FromStr;

#[derive(Serialize, Deserialize, Clone, Debug, Eq, PartialEq)]
pub enum Language {
    Bash,
    Toml,
    Rust,
    Go,
    Typescript,
    Python,
    Ruby,
}

pub const PROGRAMMING_LANGUAGES: [Language; 5] = [
    Language::Rust,
    Language::Go,
    Language::Typescript,
    Language::Python,
    Language::Ruby,
];

impl Language {
    pub fn pkg_file(&self) -> &'static str {
        match self {
            Self::Rust => "Cargo.toml",
            Self::Go => "go.mod",
            Self::Typescript => "package.json",
            Self::Python => "requirements.txt",
            Self::Ruby => "Gemfile",
            Self::Bash => "",
            Self::Toml => "",
        }
    }
    pub fn exts(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["rs"],
            Self::Go => vec!["go"],
            Self::Typescript => vec!["jsx", "tsx", "ts", "js"],
            Self::Python => vec!["py", "ipynb"],
            Self::Ruby => vec!["rb"],
            Self::Bash => vec!["sh"],
            Self::Toml => vec!["toml"],
        }
    }
    pub fn skip_dirs(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => vec!["target", ".git"],
            Self::Go => vec!["vendor", ".git"],
            Self::Typescript => vec!["node_modules", ".git"],
            Self::Python => vec!["__pycache__", ".git", ".venv", "venv"],
            Self::Ruby => vec!["migrate", "tmp", ".git"],
            Self::Bash => vec![".git"],
            Self::Toml => vec![".git"],
        }
    }
    pub fn skip_file_ends(&self) -> Vec<&'static str> {
        match self {
            Self::Typescript => vec![".min.js"],
            _ => Vec::new(),
        }
    }
    pub fn only_include_files(&self) -> Vec<&'static str> {
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript => Vec::new(),
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
        }
    }
    pub fn default_do_lsp(&self) -> bool {
        match self {
            Self::Rust => true,
            Self::Go => true,
            Self::Typescript => true,
            Self::Python => false,
            Self::Ruby => false,
            Self::Bash => false,
            Self::Toml => false,
        }
    }
    pub fn lsp_exec(&self) -> String {
        match self {
            Self::Rust => "rust-analyzer",
            Self::Go => "gopls",
            Self::Typescript => "typescript-language-server",
            Self::Python => "pylsp",
            Self::Ruby => "ruby-lsp",
            Self::Bash => "",
            Self::Toml => "",
        }
        .to_string()
    }
    pub fn version_arg(&self) -> String {
        match self {
            Self::Rust => "--version",
            Self::Go => "version",
            Self::Typescript => "--version",
            Self::Python => "--version",
            Self::Ruby => "--version",
            Self::Bash => "",
            Self::Toml => "",
        }
        .to_string()
    }
    pub fn lsp_args(&self) -> Vec<String> {
        match self {
            Self::Rust => Vec::new(),
            Self::Go => Vec::new(),
            Self::Typescript => {
                vec!["--stdio".to_string()]
            }
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            // Self::Ruby => vec!["stdio".to_string()],
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
        }
    }
    pub fn to_string(&self) -> String {
        match self {
            Self::Rust => "rust",
            Self::Go => "go",
            Self::Typescript => "typescript",
            Self::Python => "python",
            Self::Ruby => "ruby",
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
            Self::Typescript => vec!["npm install --force"],
            Self::Python => Vec::new(),
            Self::Ruby => Vec::new(),
            // Self::Ruby => {
            //     vec![
            //         r#"rm .ruby-version"#,
            //         r#"echo "3.2.2" > .ruby-version"#,
            //         r#"sed -i -e -r "s/ruby '[0-9]+\.[0-9]+\.[0-9]+'/ruby '3.2.2'/g" Gemfile"#,
            //     ]
            // }
            Self::Bash => Vec::new(),
            Self::Toml => Vec::new(),
        }
    }
    pub fn test_id_regex(&self) -> Option<&'static str> {
        match self {
            Self::Typescript => Some(r#"data-testid=(?:["']([^"']+)["']|\{['"`]([^'"`]+)['"`]\})"#),
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
            "react" => Ok(Language::Typescript),
            "React" => Ok(Language::Typescript),
            "tsx" => Ok(Language::Typescript),
            "ts" => Ok(Language::Typescript),
            "ruby" => Ok(Language::Ruby),
            "Ruby" => Ok(Language::Ruby),
            "RubyOnRails" => Ok(Language::Ruby),
            "rust" => Ok(Language::Rust),
            "Rust" => Ok(Language::Rust),
            "bash" => Ok(Language::Bash),
            "Bash" => Ok(Language::Bash),
            "toml" => Ok(Language::Toml),
            "Toml" => Ok(Language::Toml),
            _ => Err(anyhow::anyhow!("unsupported language")),
        }
    }
}
