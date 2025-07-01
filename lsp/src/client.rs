use crate::{Cmd, Language, Position, Res};

use anyhow::{anyhow, Result};
use async_lsp::concurrency::{Concurrency, ConcurrencyLayer};
use async_lsp::panic::{CatchUnwind, CatchUnwindLayer};
use async_lsp::router::Router;
use async_lsp::tracing::{Tracing, TracingLayer};
use async_lsp::MainLoop;
use async_lsp::{LanguageServer, ServerSocket};
use lsp_types::notification::{
    DidChangeWatchedFiles, LogMessage, Progress, PublishDiagnostics, ShowMessage,
};
use lsp_types::request::{
    GotoImplementationParams, GotoImplementationResponse, WorkDoneProgressCreate,
};
use lsp_types::Position as LspPosition;
use lsp_types::*;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use tokio::sync::oneshot;
use tower::ServiceBuilder;
use tracing::{debug, info, trace};

pub struct LspClient {
    root: PathBuf,
    server: ServerSocket,
}

#[derive(Debug)]
struct Stop;

pub struct ClientState {
    indexed_tx: Option<oneshot::Sender<()>>,
}

pub type ClientLoop = MainLoop<Tracing<CatchUnwind<Concurrency<Router<ClientState>>>>>;

impl LspClient {
    pub fn new(root_dir: &PathBuf, lang: &Language) -> (Self, ClientLoop, oneshot::Receiver<()>) {
        debug!("new: {:?}", lang);
        let (tx, rx) = oneshot::channel();
        let (client, mainloop) = start(tx, root_dir, lang);
        (client, mainloop, rx)
    }
    pub fn new_from(root: PathBuf, server: ServerSocket) -> Self {
        Self { root, server }
    }
    fn file_path(&self, f: &PathBuf) -> Result<Url> {
        let root_dir = Path::new(&self.root).canonicalize()?;
        let file = root_dir.join(&f);
        let file = Url::from_file_path(file).map_err(|_| anyhow!("bad file"))?;
        Ok(file)
    }
    pub async fn handle(&mut self, cmd: Cmd) -> Result<Res> {
        trace!("handle: {:?}", cmd);
        Ok(match cmd {
            Cmd::DidOpen(di) => {
                let fp = self.file_path(&di.file)?;
                self.did_open(&fp, &di.text, &di.lang.to_string()).await?;
                Res::Opened(fp.to_string())
            }
            Cmd::GotoDefinition(pos) => {
                let fp = self.file_path(&pos.file)?;
                Res::GotoDefinition(match self.definition(&fp, pos.line, pos.col).await? {
                    Some(def) => Position::from_def(def, &self.root),
                    None => None,
                })
            }
            Cmd::GotoImplementations(pos) => {
                let fp = self.file_path(&pos.file)?;
                Res::GotoImplementations(
                    match self.implementation(&fp, pos.line, pos.col).await? {
                        Some(def) => Position::from_def(def, &self.root),
                        None => None,
                    },
                )
            }
            Cmd::Hover(pos) => {
                let fp = self.file_path(&pos.file)?;
                match self.hover(&fp, pos.line, pos.col).await? {
                    Some(hov) => hov.try_into()?,
                    None => Res::Hover(None),
                }
            }
            Cmd::Stop => Res::Stopping,
        })
    }
    pub async fn stop(&mut self) -> Result<()> {
        self.server.shutdown(()).await?;
        self.server.exit(())?;
        self.server.emit(Stop)?;
        Ok(())
    }
    pub async fn init(&mut self) -> Result<InitializeResult> {
        debug!("LSP init... {:?}", self.root);
        let ret = self
            .server
            .initialize(InitializeParams {
                workspace_folders: Some(vec![WorkspaceFolder {
                    uri: Url::from_file_path(&self.root).unwrap(),
                    name: "root".into(),
                }]),
                capabilities: ClientCapabilities {
                    window: Some(WindowClientCapabilities {
                        work_done_progress: Some(true),
                        ..WindowClientCapabilities::default()
                    }),
                    ..ClientCapabilities::default()
                },
                ..InitializeParams::default()
            })
            .await?;
        self.server.initialized(InitializedParams {})?;
        Ok(ret)
    }
    pub async fn did_open(&mut self, uri: &Url, text: &str, language: &str) -> Result<()> {
        Ok(self.server.did_open(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: uri.clone(),
                language_id: language.into(),
                version: 0,
                text: text.into(),
            },
        })?)
    }
    pub async fn definition(
        &mut self,
        uri: &Url,
        line: u32,
        col: u32,
    ) -> Result<Option<GotoDefinitionResponse>> {
        Ok(self
            .server
            .definition(GotoDefinitionParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: LspPosition::new(line, col),
                },
                partial_result_params: Default::default(),
                work_done_progress_params: Default::default(),
            })
            .await?)
    }
    pub async fn implementation(
        &mut self,
        uri: &Url,
        line: u32,
        col: u32,
    ) -> Result<Option<GotoImplementationResponse>> {
        Ok(self
            .server
            .implementation(GotoImplementationParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: LspPosition::new(line, col),
                },
                partial_result_params: Default::default(),
                work_done_progress_params: Default::default(),
            })
            .await?)
    }
    pub async fn hover(&mut self, uri: &Url, line: u32, col: u32) -> Result<Option<Hover>> {
        Ok(self
            .server
            .hover(HoverParams {
                text_document_position_params: TextDocumentPositionParams {
                    text_document: TextDocumentIdentifier { uri: uri.clone() },
                    position: LspPosition::new(line, col),
                },
                work_done_progress_params: Default::default(),
            })
            .await?)
    }
}

pub fn strip_root(f: &Path, root: &PathBuf) -> PathBuf {
    if f.starts_with(root) {
        let endpart = f.strip_prefix(root).unwrap();
        endpart.into()
    } else {
        f.into()
    }
}

fn start(
    indexed_tx: oneshot::Sender<()>,
    root_dir: &PathBuf,
    lang: &Language,
) -> (LspClient, ClientLoop) {
    info!("starting LSP client for {:?}", lang);
    let (mainloop, server) = async_lsp::MainLoop::new_client(|_server| {
        let mut router = Router::new(ClientState {
            indexed_tx: Some(indexed_tx),
        });
        // https://github.com/golang/vscode-go/issues/1153

        router
            .notification::<PublishDiagnostics>(|_, _| ControlFlow::Continue(()))
            .notification::<DidChangeWatchedFiles>(|_this, c| {
                info!("===> DidChangeWatchedFiles: {:?}", c);
                ControlFlow::Continue(())
            })
            .notification::<ShowMessage>(|_, params| {
                debug!("ShowMessage::: {:?}: {}", params.typ, params.message);
                ControlFlow::Continue(())
            })
            .notification::<LogMessage>(|this, params| {
                debug!("LogMessage::: {:?}: {}", params.typ, params.message);
                if let Some(tx) = this.indexed_tx.take() {
                    let _: Result<_, _> = tx.send(());
                }
                ControlFlow::Continue(())
            })
            .unhandled_notification(|_, _| ControlFlow::Continue(()))
            .unhandled_event(|_, _| ControlFlow::Continue(()))
            .event(|_, ev| {
                if matches!(ev, Stop) {
                    ControlFlow::Break(Ok(()))
                } else {
                    debug!("event: {:?}", ev);
                    ControlFlow::Continue(())
                }
            });
        router.request::<WorkDoneProgressCreate, _>(|_, _| async { Ok(()) });
        match lang {
            Language::Rust => {
                router.notification::<Progress>(|this, prog| {
                    println!("Progress: {:?}", prog);
                    info!("{:?} {:?}", prog.token, prog.value);

                    if matches!(prog.token, NumberOrString::String(s) if s == "rustAnalyzer/Indexing") {
                        let ProgressParamsValue::WorkDone(wd) = prog.value;
                        if let WorkDoneProgress::Report(report) = wd {
                            let per = report.percentage.unwrap_or(0);
                            if let Some(msg) = report.message {
                                println!("=> {msg} ({per}%)");
                            }
                            if per == 100 {
                                if let Some(tx) = this.indexed_tx.take() {
                                    let _: Result<_, _> = tx.send(());
                                }
                            }
                        }
                    }
                    ControlFlow::Continue(())
                });
            }
            _ => {
                //
            }
        }

        ServiceBuilder::new()
            .layer(TracingLayer::default())
            .layer(CatchUnwindLayer::default())
            .layer(ConcurrencyLayer::default())
            .service(router)
    });

    let lsp_client = LspClient::new_from(root_dir.into(), server);
    (lsp_client, mainloop)
}
