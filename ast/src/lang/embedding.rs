use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use shared::{Error, Result};
use std::sync::Arc;
use tokio::sync::{Mutex, OnceCell};

pub const EMBEDDING_DIM: usize = 384;
pub const DEFAULT_CHUNK_SIZE: usize = 400;

static EMBEDDER: OnceCell<Arc<Mutex<TextEmbedding>>> = OnceCell::const_new();

pub async fn get_embedder() -> Result<Arc<Mutex<TextEmbedding>>> {
    if let Some(embedder) = EMBEDDER.get() {
        return Ok(embedder.clone());
    }

    let options = InitOptions::new(EmbeddingModel::BGESmallENV15).with_max_length(512);
    let model = TextEmbedding::try_new(options)
        .map_err(|e| Error::Custom(format!("failed to load embedding model: {e}")))?;
    let embedder = Arc::new(Mutex::new(model));
    let _ = EMBEDDER.set(embedder.clone());
    Ok(embedder)
}
fn weighted_pooling(embeddings: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    let dim = embeddings[0].len();
    let mut result = vec![0.0; dim];
    let mut total_weight = 0.0;
    for (embed, &weight) in embeddings.iter().zip(weights.iter()) {
        total_weight += weight;
        for (i, &val) in embed.iter().enumerate() {
            result[i] += val * weight;
        }
    }
    for val in &mut result {
        *val /= total_weight;
    }
    result
}

fn normalize(vec: &[f32]) -> Vec<f32> {
    let mag = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
    if mag == 0.0 {
        vec.to_vec()
    } else {
        vec.iter().map(|v| v / mag).collect()
    }
}

pub fn chunk_code(code: &str, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    let mut current_len = 0;
    for line in code.lines() {
        if current_len + line.len() > chunk_size {
            if current_len > 0 {
                chunks.push(current_chunk.join("\n"));
                current_chunk.clear();
                current_len = 0;
            }
            if line.len() > chunk_size {
                for chunk in line.as_bytes().chunks(chunk_size) {
                    chunks.push(String::from_utf8_lossy(chunk).to_string());
                }
            } else {
                current_chunk.push(line.to_string());
                current_len = line.len();
            }
        } else {
            current_chunk.push(line.to_string());
            current_len += line.len();
        }
    }
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join("\n"));
    }
    chunks
}

pub async fn vectorize_query(query: &str) -> Result<Vec<f32>> {
    let embedder = get_embedder().await?;
    let mut embedder = embedder.lock().await;
    let vecs = embedder.embed(vec![query], None)?;
    Ok(vecs
        .into_iter()
        .next()
        .unwrap_or_else(|| vec![0.0; EMBEDDING_DIM]))
}

pub async fn vectorize_code_document(code: &str) -> Result<Vec<f32>> {
    let embedder = get_embedder().await?;
    let mut embedder = embedder.lock().await;

    if code.len() < DEFAULT_CHUNK_SIZE {
        let vecs = embedder.embed(vec![code], None)?;
        let embedding = vecs
            .into_iter()
            .next()
            .unwrap_or_else(|| vec![0.0; EMBEDDING_DIM]);
        return Ok(embedding);
    }

    let chunks = chunk_code(code, DEFAULT_CHUNK_SIZE);
    let all_embeddings = embedder.embed(chunks.clone(), None)?;

    let mut weights = vec![1.0; all_embeddings.len()];
    if !weights.is_empty() {
        weights[0] = 1.2;
    }
    let pooled = weighted_pooling(&all_embeddings, &weights);
    Ok(normalize(&pooled))
}
/*
TODO:
- [X] Search with Rust
- [] Text embedding with fastembed in TS
- [] Explore embedding models best suited for code
*/
