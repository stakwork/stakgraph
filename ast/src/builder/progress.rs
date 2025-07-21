use crate::repo::Repo;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct StatusUpdate {
    pub status: String,
    pub message: String,
    pub step: u32,
    pub total_steps: u32,
    pub progress: u32,
    pub stats: Option<HashMap<String, usize>>,
    pub step_description: Option<String>,
}

impl StatusUpdate {
    pub fn as_json_str(&self) -> String {
        match serde_json::to_string(self) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Error serializing StatusUpdate: {}", e);
                "{}".to_string()
            }
        }
    }
}

impl Repo {
    pub fn send_status_update(&self, msg: &str, step: u32) {
        let step_description = match msg {
            "add_repository_and_language_nodes" => "Setting up repository",
            "collect_and_add_directories" => "Scanning directories",
            "process_and_add_files" => "Processing files",
            "setup_lsp" => "Initializing language server",
            "process_libraries" => "Analyzing libraries",
            "process_imports" => "Processing imports",
            "process_variables" => "Analyzing variables",
            "process_classes" => "Detecting classes",
            "process_instances_and_traits" => "Processing traits",
            "process_data_models" => "Analyzing data models",
            "process_functions_and_tests" => "Processing functions and tests",
            "process_pages_and_templates" => "Analyzing pages and templates",
            "process_endpoints" => "Detecting endpoints",
            "process_integration_tests" => "Processing tests",
            "process_function_calls" => "Analyzing function calls",
            "linking_graphs" => "Linking graphs",
            _ => msg,
        };

        let formatted_msg = format!("Step {}: {}", step, step_description);

        let su = StatusUpdate {
            status: "".to_string(),
            message: formatted_msg,
            step,
            total_steps: 16,
            progress: 0,
            stats: None,
            step_description: Some(step_description.to_string()),
        };

        info!("status_update: {:?}", su);
        if let Some(status_tx) = &self.status_tx {
            if let Err(e) = status_tx.send(su) {
                tracing::error!("Error sending status update: {}", e);
            }
        }
    }

    pub fn send_status_progress(&self, progress: usize, total_files: usize) {
        if total_files == 0 {
            return;
        }

        let current_progress = ((progress as f64 / total_files as f64) * 100.0).min(100.0) as u32;
        if let (Some(ref request_id), Some(ref status_map)) =
            (&self.request_id, &self.async_status_map)
        {
            let mut map = crate::utils::sync_fn(|| status_map.lock());
            if let Some(status) = map.get_mut(request_id) {
                status.progress = current_progress;
            }
        }
        let now = std::time::Instant::now();

        static LAST_PROGRESS: std::sync::atomic::AtomicU32 =
            std::sync::atomic::AtomicU32::new(u32::MAX);
        static LAST_TIME: std::sync::OnceLock<std::sync::Mutex<std::time::Instant>> =
            std::sync::OnceLock::new();

        let last_progress = LAST_PROGRESS.load(std::sync::atomic::Ordering::Relaxed);
        let last_time_mutex = LAST_TIME.get_or_init(|| std::sync::Mutex::new(now));

        let should_send = if (current_progress as i32 - last_progress as i32).abs() >= 2 {
            let last_time = *last_time_mutex.lock().unwrap();
            let time_elapsed = now.duration_since(last_time).as_millis() >= 50;
            let is_complete = current_progress >= 100;
            time_elapsed || is_complete
        } else {
            false
        };

        if should_send {
            LAST_PROGRESS.store(current_progress, std::sync::atomic::Ordering::Relaxed);
            *last_time_mutex.lock().unwrap() = now;

            let su = StatusUpdate {
                total_steps: 16,
                progress: current_progress,
                ..Default::default()
            };
            debug!("progress: {:?}", su);
            if let Some(status_tx) = &self.status_tx {
                if let Err(e) = status_tx.send(su) {
                    tracing::error!("Error sending progress update: {}", e);
                }
            }
        }
    }

    pub fn send_status_with_stats(&self, stats: HashMap<String, usize>) {
        let su = StatusUpdate {
            total_steps: 16,
            stats: Some(stats),
            ..Default::default()
        };

        debug!("stats update: {:?}", su);
        if let Some(status_tx) = &self.status_tx {
            if let Err(e) = status_tx.send(su) {
                tracing::error!("Error sending stats update: {}", e);
            }
        }
    }
}
