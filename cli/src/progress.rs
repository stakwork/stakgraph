use ast::repo::StatusUpdate;
use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use std::io::IsTerminal;
use std::time::Duration;
use tokio::sync::broadcast;

pub struct CliSpinner {
    spinner: Option<ProgressBar>,
}

impl CliSpinner {
    pub fn new(message: &str) -> Self {
        let spinner = if std::io::stderr().is_terminal() {
            let pb = ProgressBar::new_spinner();
            let style = ProgressStyle::default_spinner()
                .template("{spinner:.cyan} {msg}")
                .unwrap_or_else(|_| ProgressStyle::default_spinner());
            pb.set_style(style);
            pb.set_message(message.to_string());
            pb.enable_steady_tick(Duration::from_millis(80));
            Some(pb)
        } else {
            None
        };

        Self { spinner }
    }

    pub fn set_message(&self, message: impl Into<String>) {
        if let Some(pb) = &self.spinner {
            pb.set_message(message.into());
        }
    }

    pub fn finish_and_clear(&self) {
        if let Some(pb) = &self.spinner {
            pb.finish_and_clear();
        }
    }

    pub fn finish_with_message(&self, message: &str) {
        if let Some(pb) = &self.spinner {
            pb.finish_with_message(style(message).green().to_string());
        }
    }
}

pub struct ProgressTracker {
    bar: Option<ProgressBar>,
    rx: broadcast::Receiver<StatusUpdate>,
}

impl ProgressTracker {
    pub fn new(verbose: bool) -> (Self, broadcast::Sender<StatusUpdate>) {
        let (tx, rx) = broadcast::channel(100);

        let bar = if verbose && std::io::stdout().is_terminal() {
            let pb = ProgressBar::new(16);
            let style = ProgressStyle::default_bar()
                .template("{spinner:.cyan} [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .map(|s| s.progress_chars("█▓▒░ "))
                .unwrap_or_else(|_| ProgressStyle::default_bar().progress_chars("█▓▒░ "));
            pb.set_style(style);
            pb.set_message("Initializing...");
            Some(pb)
        } else {
            None
        };

        (Self { bar, rx }, tx)
    }

    pub async fn run(mut self) {
        loop {
            match self.rx.recv().await {
                Ok(update) => {
                    if let Some(ref bar) = self.bar {
                        if update.step > 0 {
                            bar.set_position(update.step as u64);
                        }

                        if let Some(desc) = update.step_description {
                            bar.set_message(desc);
                        } else if !update.message.is_empty() {
                            bar.set_message(update.message);
                        }
                    }
                }
                Err(_) => break,
            }
        }

        if let Some(bar) = self.bar {
            bar.finish_with_message(style("Complete").green().to_string());
        }
    }
}
