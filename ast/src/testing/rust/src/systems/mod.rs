// @ast node: Function "run_system"
pub mod worker;

pub async fn run_system() {
    worker::start_worker().await;
}
