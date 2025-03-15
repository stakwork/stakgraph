mod db;
mod model;
mod routes;

use anyhow::Result;
use std::net::SocketAddr;

const AXUM_PORT: u16 = 5002;
const ACTIX_PORT: u16 = 5004;
const ROCKET_PORT: u16 = 5006;
const ADDRESS: [u8; 4] = [0, 0, 0, 0];

#[tokio::main]
async fn main() -> Result<()> {
    println!("Initializing database...");
    db::init_db().await?;

    println!("Starting servers...");
    println!("Axum server on http://localhost:{}", AXUM_PORT);
    println!("Actix server on http://localhost:{}", ACTIX_PORT);
    println!("Rocket server on http://localhost:{}", ROCKET_PORT);

    // Run all three servers concurrently using select!
    tokio::select! {
        res = start_axum_server() => {
            if let Err(e) = res {
                eprintln!("Axum server error: {}", e);
            }
        },
        res = start_actix_server() => {
            if let Err(e) = res {
                eprintln!("Actix server error: {}", e);
            }
        },
        res = start_rocket_server() => {
            if let Err(e) = res {
                eprintln!("Rocket server error: {}", e);
            }
        },
    }

    Ok(())
}

async fn start_axum_server() -> Result<()> {
    let app = routes::axum_routes::create_router();
    let addr = SocketAddr::from((ADDRESS, AXUM_PORT));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .map_err(|e| anyhow::anyhow!("Axum server error: {}", e))?;

    Ok(())
}

async fn start_actix_server() -> Result<()> {
    // Using move to ensure closure captures nothing by reference
    actix_web::HttpServer::new(move || {
        actix_web::App::new().configure(routes::actix_routes::config)
    })
    .bind(format!("0.0.0.0:{}", ACTIX_PORT))?
    .run()
    .await
    .map_err(|e| anyhow::anyhow!("Actix server error: {}", e))?;

    Ok(())
}

async fn start_rocket_server() -> Result<()> {
    let figment = rocket::Config::figment()
        .merge(("port", ROCKET_PORT))
        .merge(("address", "0.0.0.0"));

    let rocket = routes::rocket_routes::create_rocket().configure(figment);

    rocket
        .launch()
        .await
        .map_err(|e| anyhow::anyhow!("Rocket server error: {:?}", e))?;

    Ok(())
}
