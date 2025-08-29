use crate::types::Metric;
use shared::Result;
use std::fs;
use std::path::Path;

pub fn sanitize_repo(u: &str) -> String {
    u.chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect()
}

pub fn pct(cov: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        (cov as f64) / (total as f64) * 100.0
    }
}

pub fn metric(total: &serde_json::Value, key: &str) -> Option<Metric> {
    let m = total.get(key)?;
    Some(Metric {
        total: m.get("total")?.as_u64()?,
        covered: m.get("covered")?.as_u64()?,
        pct: m.get("pct")?.as_f64()?,
    })
}

pub fn parse_summary_or_final(
    repo_path: &Path,
) -> Result<(Option<Metric>, Option<Metric>, Option<Metric>, Option<Metric>)> {
    use shared::Error;
    let summary_path = repo_path.join("coverage/coverage-summary.json");
    println!("Checking for coverage summary at {}", summary_path.display());
    if summary_path.exists() {
        println!("Found coverage summary at {}", summary_path.display());
        let v: serde_json::Value = serde_json::from_slice(&fs::read(&summary_path)?)?;
        let total = v
            .get("total")
            .cloned()
            .ok_or_else(|| Error::Custom("missing total".into()))?;
        return Ok((
            metric(&total, "lines"),
            metric(&total, "branches"),
            metric(&total, "functions"),
            metric(&total, "statements"),
        ));
    } else {
        println!("No coverage summary found at {}", summary_path.display());
    }
    let final_path = repo_path.join("coverage/coverage-final.json");
    println!("Checking for coverage final at {}", final_path.display());
    if !final_path.exists() {
        println!("No coverage final found at {}", final_path.display());
        return Ok((None, None, None, None));
    }
    println!("Found coverage final at {}", final_path.display());
    let v: serde_json::Value = serde_json::from_slice(&fs::read(&final_path)?)?;
    let mut line_total = 0;
    let mut line_cov = 0;
    let mut branch_total = 0;
    let mut branch_cov = 0;
    let mut fn_total = 0;
    let mut fn_cov = 0;
    let mut stmt_total = 0;
    let mut stmt_cov = 0;
    if let Some(obj) = v.as_object() {
        for (_file, data) in obj.iter() {
            if let Some(s) = data.get("s").and_then(|x| x.as_object()) {
                stmt_total += s.len() as u64;
                stmt_cov += s.values().filter(|v| v.as_i64().unwrap_or(0) > 0).count() as u64;
            }
            if let Some(fm) = data.get("f").and_then(|x| x.as_object()) {
                fn_total += fm.len() as u64;
                fn_cov += fm.values().filter(|v| v.as_i64().unwrap_or(0) > 0).count() as u64;
            }
            if let Some(bm) = data.get("b").and_then(|x| x.as_object()) {
                for arr in bm.values() {
                    if let Some(a) = arr.as_array() {
                        branch_total += a.len() as u64;
                        branch_cov += a
                            .iter()
                            .filter(|v| v.as_i64().unwrap_or(0) > 0)
                            .count() as u64;
                    }
                }
            }
            if let Some(lines_obj) = data.get("l").and_then(|x| x.as_object()) {
                line_total += lines_obj.len() as u64;
                line_cov += lines_obj
                    .values()
                    .filter(|v| v.as_i64().unwrap_or(0) > 0)
                    .count() as u64;
            }
        }
    }
    let lines_m = if line_total > 0 { 
        Some(Metric { total: line_total, covered: line_cov, pct: pct(line_cov, line_total) })
    } else {
        None
    };

    let branches_m = if branch_total > 0 {
        Some(Metric { total: branch_total, covered: branch_cov, pct: pct(branch_cov, branch_total) })
    } else {
        None
    };
    let functions_m = if fn_total > 0 {
        Some(Metric { total: fn_total, covered: fn_cov, pct: pct(fn_cov, fn_total) })
    } else {
        None
    };
    let statements_m = if stmt_total > 0 {
        Some(Metric { total: stmt_total, covered: stmt_cov, pct: pct(stmt_cov, stmt_total) })
    } else {
        None
    };

    Ok((lines_m, branches_m, functions_m, statements_m))
}

pub fn has_any_files_with_ext(repo_path: &Path, exts: &[&str]) -> Result<bool> {
    let mut stack = vec![repo_path.to_path_buf()];
    while let Some(p) = stack.pop() {
        for entry in fs::read_dir(p)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if ["node_modules", "coverage"].contains(&name) || name.starts_with('.') {
                    continue;
                }
                stack.push(path);
            } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if exts.contains(&ext) {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}
