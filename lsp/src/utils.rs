use futures_util::io::AsyncReadExt;
use shared::error::{Context, Error, Result};
use std::process::Stdio;

use crate::Language;

pub async fn run(cmd: &str, args: &[&str]) -> Result<String> {
    let output = async_process::Command::new(cmd)
        .args(args)
        .kill_on_drop(true)
        .output() // or "output"
        .await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(Error::Custom(format!("{} failed: {}", cmd, stderr)));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

pub async fn run_res_in_dir(cmd: &str, args: &[&str], dir: &str) -> Result<String> {
    let res = async_process::Command::new(cmd)
        .args(args)
        .current_dir(dir)
        .kill_on_drop(true)
        .output() // or "output"
        .await?;
    if !res.status.success() {
        let err = String::from_utf8_lossy(&res.stderr).to_string();
        return Err(Error::Custom(format!("err : {}", err)));
    }
    Ok(String::from_utf8_lossy(&res.stdout).to_string())
}

use flate2::read::GzDecoder;
use std::fs::File;
use tar::Archive;
pub fn untar(path: &str, dest: &str) -> Result<()> {
    let tar_gz = File::open(path)?;
    let tar = GzDecoder::new(tar_gz);
    let mut archive = Archive::new(tar);
    archive.unpack(dest)?;
    Ok(())
}

pub async fn get_lsp_version(l: Language) -> Result<String> {
    let child = async_process::Command::new(l.lsp_exec())
        .args([l.version_arg().as_str()])
        .stdout(Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;
    let mut stdout = child.stdout.context("Failed to capture stdout")?;
    let mut buf = Vec::new();
    stdout.read_to_end(&mut buf).await?;
    Ok(String::from_utf8_lossy(&buf).to_string())
}

pub fn remove_dir(path: &str) -> Result<()> {
    if std::path::Path::new(path).exists() {
        std::fs::remove_dir_all(path)?;
    }
    Ok(())
}
