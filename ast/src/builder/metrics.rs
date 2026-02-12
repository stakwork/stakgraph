use crate::lang::graphs::Graph;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct MemorySnapshot {
    #[allow(dead_code)]
    pub stage: String,
    pub rss_mb: usize,
    pub vsz_mb: usize,
}

pub fn get_process_memory() -> MemorySnapshot {
    let stage = String::new();
    
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            let mut rss_kb = 0;
            let mut vsz_kb = 0;
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        rss_kb = val.parse().unwrap_or(0);
                    }
                }
                if line.starts_with("VmSize:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        vsz_kb = val.parse().unwrap_or(0);
                    }
                }
            }
            return MemorySnapshot {
                stage,
                rss_mb: rss_kb / 1024,
                vsz_mb: vsz_kb / 1024,
            };
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("ps")
            .args(&["-o", "rss=,vsz=", "-p", &std::process::id().to_string()])
            .output()
        {
            if let Ok(stdout) = String::from_utf8(output.stdout) {
                let parts: Vec<&str> = stdout.split_whitespace().collect();
                if parts.len() >= 2 {
                    let rss_kb: usize = parts[0].parse().unwrap_or(0);
                    let vsz_kb: usize = parts[1].parse().unwrap_or(0);
                    return MemorySnapshot {
                        stage,
                        rss_mb: rss_kb / 1024,
                        vsz_mb: vsz_kb / 1024,
                    };
                }
            }
        }
    }

    MemorySnapshot {
        stage,
        rss_mb: 0,
        vsz_mb: 0,
    }
}

pub fn get_node_type_breakdown(graph: &impl Graph) -> HashMap<String, usize> {
    let mut breakdown = HashMap::new();
    let all_nodes = graph.get_all_nodes();
    
    for (node_type, _) in all_nodes {
        let type_name = format!("{:?}", node_type);
        *breakdown.entry(type_name).or_insert(0) += 1;
    }
    
    breakdown
}

pub fn sample_body_sizes(graph: &impl Graph) -> BodySizeStats {
    let all_nodes = graph.get_all_nodes();
    if all_nodes.is_empty() {
        return BodySizeStats {
            avg_bytes: 0,
            max_bytes: 0,
            total_bytes: 0,
        };
    }

    let sample_size = (all_nodes.len() / 10).max(1).min(100);
    let step = all_nodes.len().max(1) / sample_size;
    
    let mut total_sampled: u64 = 0;
    let mut max_bytes: usize = 0;
    let mut count = 0;

    for i in (0..all_nodes.len()).step_by(step.max(1)) {
        if let Some((_, nd)) = all_nodes.get(i) {
            let body_len = nd.body.len();
            total_sampled += body_len as u64;
            max_bytes = max_bytes.max(body_len);
            count += 1;
        }
    }

    let avg_bytes = if count > 0 {
        (total_sampled / count as u64) as usize
    } else {
        0
    };

    let total_estimated = avg_bytes * all_nodes.len();

    BodySizeStats {
        avg_bytes,
        max_bytes,
        total_bytes: total_estimated,
    }
}

#[derive(Clone, Debug)]
pub struct BodySizeStats {
    pub avg_bytes: usize,
    #[allow(dead_code)]
    pub max_bytes: usize,
    pub total_bytes: usize,
}
