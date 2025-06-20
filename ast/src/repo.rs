use crate::lang::graphs::Graph;
use crate::lang::{linker, ArrayGraph, BTreeMapGraph, Lang};
use anyhow::{anyhow, Context, Result};
use git_url_parse::GitUrl;
use lsp::language::{Language, PROGRAMMING_LANGUAGES};
use lsp::{git::git_clone, spawn_analyzer, strip_root, CmdSender};
use std::str::FromStr;
use std::{fs, path::PathBuf};
use tracing::{info, warn};
use walkdir::{DirEntry, WalkDir};

const CONF_FILE_PATH: &str = ".ast.json";

pub async fn clone_repo(
    url: &str,
    path: &str,
    username: Option<String>,
    pat: Option<String>,
) -> Result<()> {
    Ok(git_clone(url, path, username, pat).await?)
}

pub struct Repo {
    pub url: String,
    pub root: PathBuf,
    pub lang: Lang,
    pub lsp_tx: Option<CmdSender>,
    pub files_filter: Vec<String>,
    pub revs: Vec<String>,
}

pub struct Repos(pub Vec<Repo>);

impl Repos {
    pub async fn build_graphs(&self) -> Result<ArrayGraph> {
        self.build_graphs_inner().await
    }

    pub async fn build_graphs_btree(&self) -> Result<BTreeMapGraph> {
        self.build_graphs_inner::<BTreeMapGraph>().await
    }
    pub async fn build_graphs_inner<G: Graph>(&self) -> Result<G> {
        let mut graph = G::new();
        for repo in &self.0 {
            info!("building graph for {:?}", repo);
            let subgraph = repo.build_graph_inner().await?;
            graph.extend_graph(subgraph);
        }

        info!("linking e2e tests");
        linker::link_e2e_tests(&mut graph)?;
        info!("linking api nodes");
        linker::link_api_nodes(&mut graph)?;

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
        })
    }
    pub async fn new_clone_multi_detect(
        urls: &str,
        username: Option<String>,
        pat: Option<String>,
        files_filter: Vec<String>,
        revs: Vec<String>,
    ) -> Result<Repos> {
        let urls = urls
            .split(',')
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        // Validate revs count - it should be empty or a multiple of urls count
        if !revs.is_empty() && revs.len() % urls.len() != 0 {
            return Err(anyhow::anyhow!(
                "Number of revisions ({}) must be a multiple of the number of repositories ({})",
                revs.len(),
                urls.len()
            ));
        }
        // Calculate how many revs per repo
        let revs_per_repo = if revs.is_empty() {
            0
        } else {
            revs.len() / urls.len()
        };
        let mut repos: Vec<Repo> = Vec::new();
        for (i, url) in urls.iter().enumerate() {
            let gurl = GitUrl::parse(url)?;
            let root = format!("/tmp/{}", gurl.fullname);
            println!("Cloning to {:?}...", &root);
            fs::remove_dir_all(&root).ok();
            clone_repo(url, &root, username.clone(), pat.clone()).await?;
            // Extract the revs for this specific repository
            let repo_revs = if revs_per_repo > 0 {
                revs[i * revs_per_repo..(i + 1) * revs_per_repo].to_vec()
            } else {
                Vec::new()
            };
            let detected =
                Self::new_multi_detect(&root, Some(url.clone()), files_filter.clone(), repo_revs)
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
            let source_files = walk_files(&root.into(), &conf)?;
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
        // Then, set up each repository with LSP
        let mut repos: Vec<Repo> = Vec::new();
        for l in filtered_langs {
            let thelang = Lang::from_language(l);
            // Run post-clone commands
            for cmd in thelang.kind.post_clone_cmd() {
                Self::run_cmd(&cmd, &root)?;
            }
            // Start LSP server
            let lsp_tx = Self::start_lsp(&root, &thelang, thelang.kind.default_do_lsp())?;
            // Add to repositories
            repos.push(Repo {
                url: url.clone().map(|u| u.into()).unwrap_or_default(),
                root: root.into(),
                lang: thelang,
                lsp_tx,
                files_filter: files_filter.clone(),
                revs: revs.clone(),
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
    ) -> Result<Self> {
        let lang = Lang::from_str(language_indicator.context("no lang indicated")?)?;

        let gurl = GitUrl::parse(url)?;
        let root = format!("/tmp/{}", gurl.fullname);
        println!("Cloning to {:?}... lsp: {}", &root, lsp);
        fs::remove_dir_all(&root).ok();
        clone_repo(url, &root, username, pat).await?;
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
        })
    }
    fn run_cmd(cmd: &str, root: &str) -> Result<()> {
        info!("Running cmd: {:?}", cmd);
        let mut arr = cmd.split(" ").collect::<Vec<&str>>();
        if arr.len() == 0 {
            return Err(anyhow!("empty cmd"));
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
            let (tx, rx) = std::sync::mpsc::channel();
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
        exts.extend(self.lang.kind.config_exts());
        exts.extend(self.lang.kind.presentation_exts());
        exts.extend(self.lang.kind.third_party_exts());
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
    pub fn collect_dirs(&self) -> Result<Vec<PathBuf>> {
        let conf = self.merge_config_with_lang();
        let dirs = walk_dirs(&self.root, &conf)?
            .iter()
            .map(|d| strip_root(d, &self.root))
            .collect();
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
    pub fn get_last_revisions(path: &str, count: usize) -> Result<Vec<String>> {
        let repo = git2::Repository::open(path)?;
        let mut revwalk = repo.revwalk()?;
        revwalk.push_head()?;
        revwalk.set_sorting(git2::Sort::TIME)?;

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
            return Err(anyhow::anyhow!("No commits found in repository"));
        }

        Ok(commits)
    }
    pub fn get_path_from_url(url: &str) -> Result<String> {
        let gurl = GitUrl::parse(url)?;
        Ok(format!("/tmp/{}", gurl.fullname))
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
