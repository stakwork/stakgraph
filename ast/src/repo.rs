pub use crate::builder::progress::StatusUpdate;
#[cfg(feature = "neo4j")]
use crate::builder::streaming::{nodes_to_bolt_format, GraphStreamingUploader};
use crate::lang::graphs::Graph;
#[cfg(feature = "neo4j")]
use crate::lang::graphs::Neo4jGraph;
use crate::lang::{linker, ArrayGraph, BTreeMapGraph, Lang};
use git_url_parse::GitUrl;
use ignore::WalkBuilder;
use lsp::language::{Language, PROGRAMMING_LANGUAGES};
use lsp::{git::git_clone, spawn_analyzer, strip_tmp, CmdSender};
use shared::{Context, Error, Result};
use std::str::FromStr;
use std::{fs, path::PathBuf};
use tokio::sync::broadcast::Sender;
use tracing::{info, warn};
use walkdir::{DirEntry, WalkDir};

const CONF_FILE_PATH: &str = ".ast.json";

pub async fn clone_repo(
    url: &str,
    path: &str,
    username: Option<String>,
    pat: Option<String>,
    commit: Option<&str>,
    branch: Option<&str>,
) -> Result<()> {
    // check if the path exists
    if fs::metadata(path).is_ok() {
        // delete unless SKIP_RECLONE is set
        let skip_reclone_env = std::env::var("SKIP_RECLONE").unwrap_or_default();
        let skip_reclone = skip_reclone_env == "true" || skip_reclone_env == "1";
        if !skip_reclone {
            fs::remove_dir_all(path).ok();
            git_clone(url, path, username, pat, commit, branch).await?;
        } else {
            info!("=> Skipping reclone for {:?}", path);
        }
    } else {
        git_clone(url, path, username, pat, commit, branch).await?;
    }
    Ok(())
}

pub struct Repo {
    pub url: String,
    pub root: PathBuf, // the absolute path to the repo (/tmp/stakwork/hive)
    pub lang: Lang,
    pub lsp_tx: Option<CmdSender>,
    pub files_filter: Vec<String>,
    pub revs: Vec<String>,
    pub status_tx: Option<Sender<StatusUpdate>>,
    pub allow_unverified_calls: bool,
    pub skip_calls: bool,
}

pub struct Repos(pub Vec<Repo>);

impl Repos {
    pub async fn set_status_tx(&mut self, status_tx: Sender<StatusUpdate>) {
        for repo in &mut self.0 {
            repo.status_tx = Some(status_tx.clone());
        }
    }
    pub async fn build_graphs(&self) -> Result<BTreeMapGraph> {
        self.build_graphs_inner().await
    }
    pub async fn build_graphs_array(&self) -> Result<ArrayGraph> {
        self.build_graphs_inner().await
    }
    pub async fn build_graphs_btree(&self) -> Result<BTreeMapGraph> {
        self.build_graphs_inner().await
    }
    pub async fn build_graphs_btree_with_streaming(
        &self,
        streaming: bool,
    ) -> Result<BTreeMapGraph> {
        self.build_graphs_inner_impl::<BTreeMapGraph>(streaming)
            .await
    }
    pub async fn build_graphs_inner_with_streaming<G: Graph>(&self, streaming: bool) -> Result<G> {
        self.build_graphs_inner_impl(streaming).await
    }
    pub async fn build_graphs_inner<G: Graph>(&self) -> Result<G> {
        let streaming = std::env::var("STREAM_UPLOAD").is_ok();
        self.build_graphs_inner_impl(streaming).await
    }
    async fn build_graphs_inner_impl<G: Graph>(&self, streaming: bool) -> Result<G> {
        if self.0.is_empty() {
            return Err(Error::Custom("Language is not supported".into()));
        }
        let mut graph = G::new(String::new(), Language::Typescript);
        if let Some(first_repo) = self.0.first() {
            graph.set_allow_unverified_calls(first_repo.allow_unverified_calls);
        }
        #[cfg(feature = "neo4j")]
        let mut streaming_ctx: Option<(Neo4jGraph, GraphStreamingUploader)> = if streaming {
            let neo = Neo4jGraph::default();
            let _ = neo.connect().await;
            Some((neo, GraphStreamingUploader::new()))
        } else {
            None
        };
        for repo in &self.0 {
            info!("building graph for {:?}", repo);
            let subgraph = repo.build_graph_inner_with_streaming(streaming).await?;
            graph.extend_graph(subgraph);
            #[cfg(feature = "neo4j")]
            if let Some((neo, uploader)) = &mut streaming_ctx {
                let all_nodes = graph.get_all_nodes();
                if !all_nodes.is_empty() {
                    let bolt_nodes = nodes_to_bolt_format(all_nodes);
                    let _ = uploader
                        .flush_stage(neo, "repo_complete", &bolt_nodes)
                        .await;
                }
            }
        }

        if let Some(first_repo) = &self.0.get(0) {
            first_repo.send_status_update("linking_graphs", 14);
        }
        info!("linking e2e tests");
        linker::link_e2e_tests(&mut graph)?;
        info!("linking api nodes");
        linker::link_api_nodes(&mut graph)?;
        #[cfg(feature = "neo4j")]
        if let Some((neo, uploader)) = &mut streaming_ctx {
            let all_nodes = graph.get_all_nodes();
            if !all_nodes.is_empty() {
                let bolt_nodes = nodes_to_bolt_format(all_nodes);
                let _ = uploader
                    .flush_stage(neo, "cross_repo_linking", &bolt_nodes)
                    .await;
            }
            let edges = graph.get_edges_vec();
            let _ = uploader
                .flush_edges_stage(neo, "cross_repo_linking", &edges)
                .await;
        }

        let (nodes_size, edges_size) = graph.get_graph_size();
        println!("Final Graph: {} nodes and {} edges", nodes_size, edges_size);

        Ok(graph)
    }
}

// from the .ast.json file
#[derive(Debug, serde::Deserialize)]
pub struct AstConfig {
    #[serde(skip_serializing_if = "Option::is_empty")]
    pub skip_dirs: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_empty")]
    pub only_include_files: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_empty")]
    pub skip_file_ends: Option<Vec<String>>,
}

// actual config (merged with lang-specific configs)
#[derive(Debug, serde::Deserialize, Default)]
pub struct Config {
    pub skip_dirs: Vec<String>,
    pub skip_file_ends: Vec<String>,
    pub only_include_files: Vec<String>,
    pub exts: Vec<String>,
}

impl Repo {
    pub fn new(
        root: &str,
        lang: Lang,
        lsp: bool,
        files_filter: Vec<String>,
        revs: Vec<String>,
    ) -> Result<Self> {
        // if let Some(new_files) = check_revs(&root, revs) {
        //     files_filter = new_files;
        // }
        for cmd in lang.kind.post_clone_cmd() {
            Self::run_cmd(&cmd, &root)?;
        }
        let lsp_tx = Self::start_lsp(&root, &lang, lsp)?;
        Ok(Self {
            url: "".into(),
            root: root.into(),
            lang,
            lsp_tx,
            files_filter,
            revs,
            status_tx: None,
            allow_unverified_calls: false,
            skip_calls: false,
        })
    }
    pub async fn new_clone_multi_detect(
        urls: &str,
        username: Option<String>,
        pat: Option<String>,
        files_filter: Vec<String>,
        revs: Vec<String>,
        commit: Option<&str>,
        branch: Option<&str>,
        use_lsp: Option<bool>,
    ) -> Result<Repos> {
        let urls = urls
            .split(',')
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        // Validate revs count - it should be empty or a multiple of urls count
        if !revs.is_empty() && revs.len() % urls.len() != 0 {
            return Err(Error::Custom(format!(
                "Number of revisions ({}) must be a multiple of the number of repositories ({})",
                revs.len(),
                urls.len()
            )));
        }
        // Calculate how many revs per repo
        let revs_per_repo = if revs.is_empty() {
            0
        } else {
            revs.len() / urls.len()
        };
        let mut repos: Vec<Repo> = Vec::new();
        for (i, url) in urls.iter().enumerate() {
            let gurl = GitUrl::parse(url).map_err(|e| {
                Error::Custom(format!("Failed to parse Git URL for {}: {}", url, e))
            })?;
            let root = format!("/tmp/{}", gurl.fullname);
            println!(
                "Cloning repo to {:?} with branch {}...",
                &root,
                branch.unwrap_or("default")
            );
            clone_repo(url, &root, username.clone(), pat.clone(), commit, branch).await?;
            // Extract the revs for this specific repository
            let repo_revs = if revs_per_repo > 0 {
                revs[i * revs_per_repo..(i + 1) * revs_per_repo].to_vec()
            } else {
                Vec::new()
            };
            let detected = Self::new_multi_detect(
                &root,
                Some(url.clone()),
                files_filter.clone(),
                repo_revs,
                use_lsp,
            )
            .await?;
            repos.extend(detected.0);
        }
        Ok(Repos(repos))
    }
    pub async fn new_multi_detect(
        root: &str,
        url: Option<String>,
        files_filter: Vec<String>,
        revs: Vec<String>,
        use_lsp: Option<bool>,
    ) -> Result<Repos> {
        // First, collect all detected languages
        let mut detected_langs: Vec<Language> = Vec::new();
        for l in PROGRAMMING_LANGUAGES {
            if let Ok(only_lang) = std::env::var("ONLY_LANG") {
                if only_lang != l.to_string() {
                    continue;
                }
            }
            let conf = Config {
                exts: stringy(l.exts()),
                skip_dirs: stringy(l.skip_dirs()),
                ..Default::default()
            };
            let source_files = walk_files(&root.into(), &conf)
                .map_err(|e| Error::Custom(format!("Failed to walk files at {}: {}", root, e)))?;
            let has_pkg_file = source_files.iter().any(|f| {
                let fname = f.display().to_string();
                if l.pkg_files().is_empty() {
                    return true;
                }
                let found_pkg_file = l
                    .pkg_files()
                    .iter()
                    .any(|pkg_file| fname.ends_with(pkg_file));
                found_pkg_file
            });
            if has_pkg_file {
                // Don't add duplicate languages
                if !detected_langs.iter().any(|lang| lang == &l) {
                    detected_langs.push(l);
                }
            }
        }
        // Filter out overridden languages
        let mut overridden_langs: Vec<Language> = Vec::new();
        for lang in &detected_langs {
            for overridden in lang.overrides() {
                overridden_langs.push(overridden);
            }
        }
        let filtered_langs: Vec<Language> = detected_langs
            .into_iter()
            .filter(|lang| !overridden_langs.contains(lang))
            .collect();

        if filtered_langs.is_empty() {
            return Err(Error::Custom(format!(
                "Language is not supported yet: {}",
                root
            )));
        }
        // Then, set up each repository with LSP
        let mut repos: Vec<Repo> = Vec::new();
        for l in filtered_langs {
            let thelang = Lang::from_language(l);
            // Run post-clone commands
            for cmd in thelang.kind.post_clone_cmd() {
                Self::run_cmd(&cmd, &root).map_err(|e| {
                    Error::Custom(format!("Failed to cmd {} in {}: {}", cmd, root, e))
                })?;
            }
            // Start LSP server
            let lsp_enabled = use_lsp.unwrap_or_else(|| thelang.kind.default_do_lsp());
            let lsp_tx = Self::start_lsp(&root, &thelang, lsp_enabled)
                .map_err(|e| Error::Custom(format!("Failed to start LSP: {}", e)))?;
            // Add to repositories
            repos.push(Repo {
                url: url.clone().map(|u| u.into()).unwrap_or_default(),
                root: root.into(),
                lang: thelang,
                lsp_tx,
                files_filter: files_filter.clone(),
                revs: revs.clone(),
                status_tx: None,
                allow_unverified_calls: false,
                skip_calls: false,
            });
        }
        println!("REPOS!!! {:?}", repos);
        Ok(Repos(repos))
    }
    pub async fn new_clone_to_tmp(
        url: &str,
        language_indicator: Option<&str>,
        lsp: bool,
        username: Option<String>,
        pat: Option<String>,
        files_filter: Vec<String>,
        revs: Vec<String>,
        branch: Option<&str>,
    ) -> Result<Self> {
        let lang = Lang::from_str(language_indicator.context("no lang indicated")?)?;

        let gurl = GitUrl::parse(url)?;
        let root = format!("/tmp/{}", gurl.fullname);
        println!("Cloning to {:?}... lsp: {}", &root, lsp);
        clone_repo(url, &root, username, pat, None, branch).await?;
        // if let Some(new_files) = check_revs(&root, revs) {
        //     files_filter = new_files;
        // }
        for cmd in lang.kind.post_clone_cmd() {
            Self::run_cmd(&cmd, &root)?;
        }
        let lsp_tx = Self::start_lsp(&root, &lang, lsp)?;
        Ok(Self {
            url: url.to_string(),
            root: root.into(),
            lang,
            lsp_tx,
            files_filter,
            revs,
            status_tx: None,
            allow_unverified_calls: false,
            skip_calls: false,
        })
    }
    fn run_cmd(cmd: &str, root: &str) -> Result<()> {
        info!("Running cmd: {:?}", cmd);
        let mut arr = cmd.split(" ").collect::<Vec<&str>>();
        if arr.len() == 0 {
            return Err(Error::Custom("empty cmd".into()));
        }
        let first = arr.remove(0);
        let mut proc = std::process::Command::new(first);
        for a in arr {
            proc.arg(a);
        }
        let _ = proc.current_dir(&root).status().ok();
        info!("Finished running: {:?}!", cmd);
        Ok(())
    }
    fn start_lsp(root: &str, lang: &Lang, lsp: bool) -> Result<Option<CmdSender>> {
        Ok(if lsp {
            let (tx, rx) = tokio::sync::mpsc::channel(10000);
            spawn_analyzer(&root.into(), &lang.kind, rx)?;
            Some(tx)
        } else {
            None
        })
    }
    pub fn delete_from_tmp(&self) -> Result<()> {
        fs::remove_dir_all(&self.root)?;
        Ok(())
    }
    fn merge_config_with_lang(&self) -> Config {
        let mut skip_dirs = stringy(self.lang.kind.skip_dirs());
        let mut only_include_files = stringy(self.lang.kind.only_include_files());
        let mut skip_file_ends = stringy(self.lang.kind.skip_file_ends());
        if let Some(fconfig) = self.read_config_file() {
            if let Some(sd) = fconfig.skip_dirs {
                skip_dirs.extend(sd);
            }
            if let Some(oif) = fconfig.only_include_files {
                only_include_files.extend(oif);
            }
            if let Some(sfe) = fconfig.skip_file_ends {
                skip_file_ends.extend(sfe);
            }
        }
        if self.files_filter.len() > 0 {
            only_include_files.extend(self.files_filter.clone());
        }
        let mut exts = self.lang.kind.exts();
        exts.push("md");
        Config {
            skip_dirs,
            skip_file_ends,
            only_include_files,
            exts: stringy(exts),
        }
    }
    pub fn collect(&self) -> Result<Vec<PathBuf>> {
        let conf = self.merge_config_with_lang();
        info!("CONFIG: {:?}", conf);
        let source_files = walk_files(&self.root, &conf)?;
        Ok(source_files)
    }
    pub fn collect_dirs_with_tmp(&self) -> Result<Vec<PathBuf>> {
        let conf = self.merge_config_with_lang();
        info!("==>>ROOT: {:?}", self.root);
        let dirs = walk_dirs(&self.root, &conf)?;
        Ok(dirs)
    }
    fn read_config_file(&self) -> Option<AstConfig> {
        let config_path = self.root.join(CONF_FILE_PATH);
        match std::fs::read_to_string(&config_path) {
            Ok(s) => match serde_json::from_str::<AstConfig>(&s) {
                Ok(c) => Some(c),
                Err(_e) => {
                    warn!("Failed to parse config file {:?}", _e);
                    return None;
                }
            },
            Err(_) => None,
        }
    }
    pub fn collect_extra_pages(
        &self,
        yes_extra_page: impl Fn(&str) -> bool,
    ) -> Result<Vec<String>> {
        let source_files = walk_files_arbitrary(&self.root, yes_extra_page)?;
        Ok(source_files)
    }
    #[cfg(feature = "openssl")]
    pub fn get_last_revisions(path: &str, count: usize) -> Result<Vec<String>> {
        let repo = git2::Repository::open(path).context("Failed to open git repository")?;
        let mut revwalk = repo.revwalk().context("Failed to create revwalk:")?;
        revwalk.push_head().context("Failed to push head:")?;
        revwalk
            .set_sorting(git2::Sort::TIME)
            .context("Failed to set sorting:")?;

        let mut commits = Vec::new();
        for oid_result in revwalk.take(count) {
            if let Ok(oid) = oid_result {
                if let Ok(commit) = repo.find_commit(oid) {
                    commits.push(commit.id().to_string());
                }
            }
        }

        commits.reverse();

        if commits.is_empty() {
            return Err(Error::Custom("No commits found in repository".into()));
        }

        Ok(commits)
    }
    pub fn get_path_from_url(url: &str) -> Result<String> {
        let gurl = GitUrl::parse(url)?;
        Ok(format!("/tmp/{}", gurl.fullname))
    }
    pub fn collect_all_files(&self) -> Result<Vec<PathBuf>> {
        let mut all_files = Vec::new();

        let walker = WalkBuilder::new(&self.root)
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .build();

        for result in walker {
            match result {
                Ok(entry) => {
                    let path = entry.path();
                    if path.is_file() {
                        let relative_path = strip_tmp(path).display().to_string();

                        if self.should_not_include(path, &relative_path) {
                            continue;
                        }
                        all_files.push(path.to_path_buf());
                    }
                }
                Err(err) => {
                    warn!("Error walking directory: {}", err);
                }
            }
        }
        Ok(all_files)
    }
    fn should_not_include(&self, path: &std::path::Path, relative_path: &str) -> bool {
        let conf = self.merge_config_with_lang();
        let fname = path.display().to_string();

        if !conf.only_include_files.is_empty() {
            return !only_files(path, &conf.only_include_files);
        }

        if path.components().any(|c| {
            lsp::language::junk_directories().contains(&c.as_os_str().to_str().unwrap_or(""))
        }) {
            return true;
        }
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if lsp::language::common_binary_exts().contains(&ext) {
                return true;
            }
        }

        if self.lang.kind.is_package_file(relative_path) {
            return false;
        }

        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if self.lang.kind.exts().contains(&ext) {
                return false;
            }
        }

        for other_lang in PROGRAMMING_LANGUAGES {
            if other_lang == self.lang.kind {
                continue;
            }

            if other_lang.is_package_file(relative_path) {
                return true;
            }

            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if other_lang.exts().contains(&ext) && !self.lang.kind.exts().contains(&ext) {
                    return true;
                }
            }
        }

        if skip_end(&fname, &conf.skip_file_ends) {
            return true;
        }
        false
    }
    pub fn from_files(
        file_paths: &[&str],
        root: std::path::PathBuf,
        lang: Lang,
        allow_unverified_calls: bool,
        skip_calls: bool,
    ) -> Result<Self> {
        let files_filter = file_paths.iter().map(|p| p.to_string()).collect();
        Ok(Self {
            url: String::new(),
            root,
            lang,
            lsp_tx: None,
            files_filter,
            revs: Vec::new(),
            status_tx: None,
            allow_unverified_calls,
            skip_calls,
        })
    }
    pub fn from_single_file(
        file_path: &str,
        lang: Lang,
        allow_unverified_calls: bool,
        skip_calls: bool,
    ) -> Result<Self> {
        let root = std::path::Path::new(file_path)
            .parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| Error::Custom("Invalid file path".into()))?;
        let files_filter = vec![file_path.to_string()];
        Ok(Self {
            url: String::new(),
            root,
            lang,
            lsp_tx: None,
            files_filter,
            revs: Vec::new(),
            status_tx: None,
            allow_unverified_calls,
            skip_calls,
        })
    }
}

fn walk_dirs(dir: &PathBuf, conf: &Config) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for entry in WalkDir::new(dir)
        .min_depth(1)
        .into_iter()
        .filter_entry(|e| !skip_dir(e, &conf.skip_dirs))
    {
        let entry = entry?;
        if entry.metadata()?.is_dir() {
            dirs.push(entry.path().to_path_buf());
        }
    }
    Ok(dirs)
}

fn is_hidden(entry: &DirEntry) -> bool {
    entry
        .file_name()
        .to_str()
        .map(|s| s.starts_with("."))
        .unwrap_or(false)
}

fn walk_files(dir: &PathBuf, conf: &Config) -> Result<Vec<PathBuf>> {
    let mut source_files: Vec<PathBuf> = Vec::new();
    for entry in WalkDir::new(dir)
        .min_depth(1)
        .into_iter()
        .filter_entry(|e| !skip_dir(e, &conf.skip_dirs) && !is_hidden(e))
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let fname = path.display().to_string();
            for l in PROGRAMMING_LANGUAGES {
                let found_pkg_file = l
                    .pkg_files()
                    .iter()
                    .any(|pkg_file| fname.ends_with(pkg_file));
                if found_pkg_file {
                    source_files.push(path.to_path_buf());
                }
            }
            if let Some(ext) = path.extension() {
                if let Some(ext) = ext.to_str() {
                    if conf.exts.contains(&ext.to_string()) || conf.exts.contains(&"*".to_string())
                    {
                        if !skip_end(&fname, &conf.skip_file_ends) {
                            if only_files(path, &conf.only_include_files) {
                                source_files.push(path.to_path_buf());
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(source_files)
}
fn skip_dir(entry: &DirEntry, skip_dirs: &Vec<String>) -> bool {
    if is_hidden(entry) {
        return true;
    }
    // FIXME skip all for all...?
    for l in PROGRAMMING_LANGUAGES {
        if entry
            .file_name()
            .to_str()
            .map(|s| l.skip_dirs().contains(&s))
            .unwrap_or(false)
        {
            return true;
        }
    }
    entry
        .file_name()
        .to_str()
        .map(|s| skip_dirs.contains(&s.to_string()))
        .unwrap_or(false)
}
fn only_files(path: &std::path::Path, only_include_files: &Vec<String>) -> bool {
    if only_include_files.is_empty() {
        return true;
    }
    let fname = path.display().to_string();
    for oif in only_include_files.iter() {
        if fname.ends_with(oif) {
            return true;
        }
    }
    false
}

fn skip_end(fname: &str, ends: &Vec<String>) -> bool {
    for e in ends.iter() {
        if fname.ends_with(e) {
            return true;
        }
    }
    false
}

fn _filenamey(f: &PathBuf) -> String {
    let full = f.display().to_string();
    if !f.starts_with("/tmp/") {
        return full;
    }
    let mut parts = full.split("/").collect::<Vec<&str>>();
    parts.drain(0..4);
    parts.join("/")
}

fn stringy(inp: Vec<&'static str>) -> Vec<String> {
    inp.iter().map(|s| s.to_string()).collect()
}

impl std::fmt::Display for Repo {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Repo Kind: {:?}", self.lang.kind)
    }
}
impl std::fmt::Debug for Repo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Repo Kind: {:?}", self.lang.kind)
    }
}

#[cfg(feature = "openssl")]
pub fn check_revs_files(repo_path: &str, mut revs: Vec<String>) -> Option<Vec<String>> {
    if revs.len() == 0 {
        return None;
    }
    if revs.len() == 1 {
        revs.push("HEAD".into());
    }
    let old_rev = revs.get(0)?;
    let new_rev = revs.get(1)?;
    crate::gat::get_changed_files(repo_path, old_rev, new_rev).ok()
}

fn walk_files_arbitrary(dir: &PathBuf, directive: impl Fn(&str) -> bool) -> Result<Vec<String>> {
    let mut source_files: Vec<String> = Vec::new();
    for entry in WalkDir::new(dir).min_depth(1).into_iter() {
        let entry = entry?;
        if entry.metadata()?.is_file() {
            let fname = entry.path().display().to_string();
            if directive(&fname) {
                source_files.push(fname);
            }
        }
    }
    Ok(source_files)
}
