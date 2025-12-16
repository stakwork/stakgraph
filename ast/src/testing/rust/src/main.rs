mod db;
mod routes;
mod traits;
mod types;
mod macros;

use crate::db::init_db;
use crate::routes::{
    actix_routes::config, axum_routes::create_router, rocket_routes::create_rocket,
};

use anyhow::Result;
use std::net::SocketAddr;

#[derive(Copy, Clone)]
enum PORT {
    Axum = 5002,
    Actix = 5004,
    Rocket = 5006,
}

//should be DM and NOT Class 'cause there is no implementation
enum CRATES {
    STANDALONE,
    AST,
    LSP,
    SKILL,
}

impl CRATES {
    fn as_str(&self) -> &'static str {
        match self {
            CRATES::STANDALONE => "standalone",
            CRATES::AST => "ast",
            CRATES::LSP => "lsp",
            CRATES::SKILL => "skill",
        }
    }
}

impl PORT {
    fn as_u16(&self) -> u16 {
        *self as u16
    }
}

const ADDRESS: [u8; 4] = [0, 0, 0, 0];

#[tokio::main]
async fn main() -> Result<()> {
    println!("Initializing database...");
    init_db().await?;

    println!("Starting servers...");
    println!("Axum server on http://localhost:{}", PORT::Axum.as_u16());
    println!("Actix server on http://localhost:{}", PORT::Actix.as_u16());
    println!(
        "Rocket server on http://localhost:{}",
        PORT::Rocket.as_u16()
    );

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
    let app = create_router();
    let addr = SocketAddr::from((ADDRESS, AXUM_PORT));

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .map_err(|e| anyhow::anyhow!("Axum server error: {}", e))?;

    Ok(())
}

async fn start_actix_server() -> Result<()> {
    // Using move to ensure closure captures nothing by reference
    actix_web::HttpServer::new(move || actix_web::App::new().configure(config))
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

    let rocket = create_rocket().configure(figment);

    rocket
        .launch()
        .await
        .map_err(|e| anyhow::anyhow!("Rocket server error: {:?}", e))?;

    Ok(())
}

//should be DM and Class Because of implementation of Display
enum LANGUAGE {
    Rust,
    Python,
    JavaScript,
}

impl std::fmt::Display for LANGUAGE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LANGUAGE::Rust => write!(f, "Rust"),
            LANGUAGE::Python => write!(f, "Python"),
            LANGUAGE::JavaScript => write!(f, "JavaScript"),
        }
    }
}
