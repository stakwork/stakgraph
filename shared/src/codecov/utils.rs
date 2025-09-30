use crate::Result;
use crate::codecov::Metric;
use std::collections::{HashMap, HashSet};
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
) -> Result<(
    Option<Metric>,
    Option<Metric>,
    Option<Metric>,
    Option<Metric>,
)> {
    use crate::Error;
    let summary_path = repo_path.join("coverage/coverage-summary.json");
    if summary_path.exists() {
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
    }

    let final_path = repo_path.join("coverage/coverage-final.json");
    if !final_path.exists() {
        return Ok((None, None, None, None));
    }
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
        for (_file, data) in obj {
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
                        branch_cov +=
                            a.iter().filter(|v| v.as_i64().unwrap_or(0) > 0).count() as u64;
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
        Some(Metric {
            total: line_total,
            covered: line_cov,
            pct: pct(line_cov, line_total),
        })
    } else {
        None
    };
    let branches_m = if branch_total > 0 {
        Some(Metric {
            total: branch_total,
            covered: branch_cov,
            pct: pct(branch_cov, branch_total),
        })
    } else {
        None
    };
    let functions_m = if fn_total > 0 {
        Some(Metric {
            total: fn_total,
            covered: fn_cov,
            pct: pct(fn_cov, fn_total),
        })
    } else {
        None
    };
    let statements_m = if stmt_total > 0 {
        Some(Metric {
            total: stmt_total,
            covered: stmt_cov,
            pct: pct(stmt_cov, stmt_total),
        })
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

pub fn collect_uncovered_lines(repo_path: &Path) -> Result<HashMap<String, Vec<u32>>> {
    let mut result = HashMap::new();
    let final_path = repo_path.join("coverage/coverage-final.json");
    if !final_path.exists() {
        return Ok(result);
    }
    let v: serde_json::Value = serde_json::from_slice(&fs::read(&final_path)?)?;
    let obj = match v.as_object() {
        Some(o) => o,
        None => return Ok(result),
    };
    for (file, data) in obj {
        let mut lines: Vec<u32> = Vec::new();
        if let Some(l_map) = data.get("l").and_then(|x| x.as_object()) {
            let mut tmp: Vec<u32> = l_map
                .iter()
                .filter_map(|(k, v)| {
                    if v.as_i64().unwrap_or(0) == 0 {
                        k.parse::<u32>().ok()
                    } else {
                        None
                    }
                })
                .collect();
            lines.append(&mut tmp);
        } else if let (Some(stmt_map), Some(stmt_hits)) = (data.get("statementMap"), data.get("s"))
        {
            if let (Some(stmt_map_obj), Some(stmt_hits_obj)) =
                (stmt_map.as_object(), stmt_hits.as_object())
            {
                let mut set: HashSet<u32> = HashSet::new();
                for (id, loc_val) in stmt_map_obj {
                    if stmt_hits_obj.get(id).and_then(|v| v.as_i64()).unwrap_or(0) == 0 {
                        let start_line = loc_val
                            .get("start")
                            .and_then(|s| s.get("line"))
                            .and_then(|l| l.as_u64())
                            .unwrap_or(0) as u32;
                        let end_line = loc_val
                            .get("end")
                            .and_then(|e| e.get("line"))
                            .and_then(|l| l.as_u64())
                            .unwrap_or(start_line as u64)
                            as u32;
                        if start_line > 0 {
                            for ln in start_line..=end_line {
                                set.insert(ln);
                            }
                        }
                    }
                }
                if !set.is_empty() {
                    lines.extend(set.into_iter());
                }
            }
        }
        if !lines.is_empty() {
            lines.sort_unstable();
            result.insert(file.clone(), lines);
        }
    }
    Ok(result)
}

fn compress_line_ranges(lines: &[u32]) -> Vec<(u32, u32)> {
    if lines.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = lines[0];
    let mut prev = lines[0];
    for &ln in &lines[1..] {
        if ln == prev + 1 {
            prev = ln;
            continue;
        }
        ranges.push((start, prev));
        start = ln;
        prev = ln;
    }
    ranges.push((start, prev));
    ranges
}

pub fn augment_and_copy_summary(repo_path: &Path, src: &Path, dest: &Path) -> Result<()> {
    use std::io::Write;
    if !src.exists() {
        return Ok(());
    }
    let mut summary: serde_json::Value = serde_json::from_slice(&fs::read(src)?)?;
    let uncovered = collect_uncovered_lines(repo_path)?;
    if let Some(obj) = summary.as_object_mut() {
        for (file, lines) in uncovered {
            if let Some(file_entry) = obj.get_mut(&file) {
                if let Some(map) = file_entry.as_object_mut() {
                    let ranges = compress_line_ranges(&lines);
                    let json_ranges = serde_json::Value::Array(
                        ranges
                            .into_iter()
                            .map(|(s, e)| {
                                serde_json::Value::Array(vec![
                                    serde_json::Value::from(s),
                                    serde_json::Value::from(e),
                                ])
                            })
                            .collect(),
                    );
                    map.insert("uncoveredLineRanges".into(), json_ranges);
                }
            }
        }
    }
    let mut f = fs::File::create(dest)?;
    let bytes = serde_json::to_vec_pretty(&summary)?;
    f.write_all(&bytes)?;
    Ok(())
}
