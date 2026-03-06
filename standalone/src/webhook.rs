use hmac::{Hmac, Mac};
use reqwest::Client;
use sha2::Sha256;
use std::time::Duration as StdDuration;
use tokio::time::{sleep, Duration};
use tracing::{debug, info};
use url::Url;

type HmacSha256 = Hmac<Sha256>;

pub async fn validate_callback_url_async(raw: &str) -> Result<Url, shared::Error> {
    let url = Url::parse(raw)
        .map_err(|e| shared::Error::validation(format!("Invalid callback_url: {e}")))?;
    if url.scheme() != "https" {
        let allow_insecure = std::env::var("ALLOW_INSECURE_WEBHOOKS")
            .ok()
            .map(|v| v == "true")
            .unwrap_or(true);
        if !allow_insecure {
            return Err(shared::Error::validation("Callback_url must use https"));
        }
    }
    if url.cannot_be_a_base() || url.host_str().is_none() {
        return Err(shared::Error::validation("CallbackUrl must be absolute"));
    }

    Ok(url)
}

fn hmac_signature_hex(secret: &[u8], body: &[u8]) -> Result<String, shared::Error> {
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|e| shared::Error::dependency(format!("failed to initialize HMAC: {e}")))?;
    mac.update(body);
    let result = mac.finalize().into_bytes();
    Ok(hex::encode(result))
}

pub async fn send_with_retries<T: serde::Serialize + ?Sized + std::fmt::Debug>(
    client: &Client,
    request_id: &str,
    url: &Url,
    payload: &T,
) -> Result<(), shared::Error> {
    let secret = std::env::var("WEBHOOK_SECRET")
        .map_err(|_| shared::Error::dependency("WEBHOOK_SECRET not set"))?;
    let max_retries: u32 = match std::env::var("WEBHOOK_MAX_RETRIES") {
        Ok(v) => v.parse().unwrap_or(3),
        Err(_) => 3,
    };
    let timeout_ms: u64 = match std::env::var("WEBHOOK_TIMEOUT_MS") {
        Ok(v) => v.parse().unwrap_or(8000),
        Err(_) => 8000,
    };

    let body_bytes = serde_json::to_vec(payload)
        .map_err(|e| shared::Error::internal(format!("serialize payload: {e}")))?;
    let payload_value: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();
    let payload_status = payload_value
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let payload_progress = payload_value
        .get("progress")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let has_error = payload_value
        .get("error")
        .map(|v| !v.is_null())
        .unwrap_or(false);

    info!(
        "STAKGRAPH WEBHOOK request_id={} status={} progress={} has_error={}",
        request_id, payload_status, payload_progress, has_error
    );
    debug!("STAKGRAPH WEBHOOK PAYLOAD {:?}", payload);

    let sig = hmac_signature_hex(secret.as_bytes(), &body_bytes)?;
    let sig_header = format!("sha256={}", sig);

    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let req = client
            .post(url.clone())
            .timeout(StdDuration::from_millis(timeout_ms))
            .header("Content-Type", "application/json")
            .header("X-Signature", &sig_header)
            .header("Idempotency-Key", request_id)
            .header("X-Request-Id", request_id)
            .body(body_bytes.clone());

        let result = req.send().await;

        match result {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return Ok(());
                }
                if status.as_u16() == 429 || status.is_server_error() {
                    if attempt <= max_retries {
                        sleep(Duration::from_millis(2500)).await;
                        continue;
                    }
                    return Err(shared::Error::dependency(format!(
                        "webhook failed with status {} after {} attempts",
                        status, attempt
                    )));
                } else {
                    return Err(shared::Error::dependency(format!(
                        "webhook failed with status {}",
                        status
                    )));
                }
            }
            Err(e) => {
                if attempt <= max_retries {
                    sleep(Duration::from_millis(2500)).await;
                    continue;
                }
                return Err(shared::Error::dependency(format!(
                    "webhook error after {} attempts: {}",
                    attempt, e
                )));
            }
        }
    }
}
