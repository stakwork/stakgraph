use tracing::info;

pub fn get_rss_mb() -> f64 {
    let pid = std::process::id();

    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        // `ps -o rss=` outputs RSS in kilobytes
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
        {
            if let Ok(s) = std::str::from_utf8(&output.stdout) {
                if let Ok(kb) = s.trim().parse::<u64>() {
                    return kb as f64 / 1024.0;
                }
            }
        }
    }

    0.0
}

pub fn log_memory(stage: &str) {
    let rss = get_rss_mb();
    info!("[perf][memory] {} rss={:.2}MB", stage, rss);
}
