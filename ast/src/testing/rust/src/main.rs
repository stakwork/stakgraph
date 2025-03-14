mod db;
mod model;
mod routes;

use anyhow::Result;
use hyper::Server;
use routes::create_router;
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<()> {
    db::init_db().await?;

    let app = create_router();

    let port = 5002;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    println!("Listening on http://{}", addr);

    Server::bind(&addr).serve(app.into_make_service()).await?;

    Ok(())
}
