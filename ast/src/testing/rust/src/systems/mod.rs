pub mod worker;

pub async fn run_system() {
    worker::start_worker().await;
}
