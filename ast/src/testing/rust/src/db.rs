use crate::model::Person;
use anyhow::{Context, Result};
use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};

pub struct Database {
    pool: Pool<Sqlite>,
}

static mut DB_INSTANCE: Option<Database> = None;

pub async fn get_db() -> &'static Database {
    unsafe { DB_INSTANCE.as_ref().expect("Database not initialized") }
}

pub async fn init_db() -> Result<()> {
    let database_url = "sqlite::memory:";
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await
        .context("failed to connect to database")?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL
    )"#,
    )
    .execute(&pool)
    .await
    .context("failed to create table")?;

    unsafe {
        DB_INSTANCE = Some(Database { pool });
    }

    Ok(())
}

impl Database {
    pub async fn new_person(&self, person: Person) -> Result<Person> {
        let id = sqlx::query("INSERT INTO people (name, email) VALUES (?, ?)")
            .bind(&person.name)
            .bind(&person.email)
            .execute(&self.pool)
            .await?
            .last_insert_rowid();

        Ok(Person {
            id: Some(id as i32),
            name: person.name,
            email: person.email,
        })
    }

    pub async fn get_person_by_id(&self, id: u32) -> Result<Person> {
        let person = sqlx::query_as::<_, Person>("SELECT id, name, email FROM people WHERE id = ?")
            .bind(id as i32)
            .fetch_one(&self.pool)
            .await
            .context("Person not found")?;

        Ok(person)
    }
}
