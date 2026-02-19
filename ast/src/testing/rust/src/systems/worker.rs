use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::Mutex;

pub async fn start_worker() {
    let (tx, mut rx) = mpsc::channel(32);
    let counter = Arc::new(Mutex::new(0));

    let counter_clone = counter.clone();
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let mut lock = counter_clone.lock().await;
            *lock += 1;
            println!("Processed: {}", msg);
        }
    });

    tx.send("task 1").await.unwrap();
    tx.send("task 2").await.unwrap();

    tokio::time::sleep(Duration::from_millis(100)).await;
}
