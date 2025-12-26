use crate::lang::NodeType;
use crate::repo::Repo;
use shared::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tree_sitter::Tree;

impl std::fmt::Debug for ParsedFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParsedFile")
            .field("path", &self.path)
            .field("code_len", &self.code.len())
            .finish()
    }
}

pub struct ParsedFile {
    pub path: String,
    code: Arc<String>,
    tree: Tree,
}

impl ParsedFile {
    pub fn new(path: String, code: String, tree: Tree) -> Self {
        Self {
            path,
            code: Arc::new(code),
            tree,
        }
    }

    pub fn get_tree(&self) -> &Tree {
        &self.tree
    }

    pub fn get_code(&self) -> &str {
        &self.code
    }
}

impl std::fmt::Debug for ParsedFileCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ParsedFileCache {{ files: {} }}", self.files.len())
    }
}

pub struct ParsedFileCache {
    files: HashMap<String, ParsedFile>,
}

impl ParsedFileCache {
    pub fn new() -> Self {
        Self {
            files: HashMap::new(),
        }
    }

    pub fn get(&self, file: &str) -> Option<&ParsedFile> {
        self.files.get(file)
    }

    pub fn insert(&mut self, file: String, parsed: ParsedFile) {
        self.files.insert(file, parsed);
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }
}

pub trait PreParse {
    fn pre_parse_all_files(&self, filez: &[(String, String)]) -> Result<ParsedFileCache>;
}

impl PreParse for Repo {
    fn pre_parse_all_files(&self, filez: &[(String, String)]) -> Result<ParsedFileCache> {
        use std::sync::Arc;
        use tracing::info;

        let mut cache = ParsedFileCache::new();
        let mut i = 0;
        let total = filez.len();

        info!("Pre-parsing {} files...", total);

        for (filepath, code) in filez {
            i += 1;
            if i % 20 == 0 || i == total {
                self.send_status_progress(i, total, 2);
            }

            let code_owned = code.clone();
            let code_arc = Arc::new(code_owned.clone());

            let tree = self.lang.lang().parse(&code_arc, &NodeType::Function)?;

            let parsed = ParsedFile::new(filepath.clone(), code_owned, tree);
            cache.insert(filepath.clone(), parsed);

            if self.lang.kind.is_package_file(filepath) {
                let lib_code = code.clone();
                let lib_code_arc = Arc::new(lib_code.clone());

                let lib_tree = self.lang.lang().parse(&lib_code_arc, &NodeType::Library)?;

                let lib_key = format!("{}.lib", filepath);
                let lib_parsed = ParsedFile::new(lib_key.clone(), lib_code, lib_tree);
                cache.insert(lib_key, lib_parsed);
            }
        }

        info!("Pre-parsed {} files", cache.len());
        Ok(cache)
    }
}
