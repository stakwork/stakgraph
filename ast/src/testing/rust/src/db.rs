use serde::{Deserialize, Serialize};
use shared::{Context, Result};
use sqlx::FromRow;
use sqlx::{sqlite::SqlitePoolOptions, Pool, Sqlite};
use std::sync::OnceLock;

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct Person {
    #[serde(skip_deserializing)]
    pub id: Option<i32>,
    pub name: String,
    pub email: String,
}

#[derive(Clone)]
pub struct Database {
    pool: Pool<Sqlite>,
}

static DB_INSTANCE: OnceLock<Database> = OnceLock::new();

#[inline]
#[allow(dead_code)]
fn internal_helper() -> String {
    "helper".to_string()
}

#[cfg(feature = "advanced")]
pub fn advanced_feature() {
    println!("Advanced feature enabled");
}

#[inline(always)]
#[must_use]
#[deprecated(since = "1.0.0", note = "use new_helper instead")]
pub fn multi_attribute_function() -> i32 {
    42
}

async fn get_db() -> &'static Database {
    DB_INSTANCE.get().expect("Database not initialized")
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

    let db = Database { pool };

    if DB_INSTANCE.get().is_none() {
        if let Err(_) = DB_INSTANCE.set(db) {
            return Err(anyhow::anyhow!("Database already initialized"));
        }
    }

    Ok(())
}

impl Database {
    async fn new_person_impl(&self, person: Person) -> Result<Person> {
        let id = sqlx::query("INSERT INTO people (name, email) VALUES (?, ?)")
            .bind(&person.name)
            .bind(&person.email)
            .execute(&self.pool)
            .await?
            .last_insert_rowid();

        let result: Person = Person {
            id: Some(id as i32),
            name: person.name,
            email: person.email,
        };

        Ok(result)
    }

    async fn get_person_by_id_impl(&self, id: u32) -> Result<Person> {
        let person: Person =
            sqlx::query_as::<_, Person>("SELECT id, name, email FROM people WHERE id = ?")
                .bind(id as i32)
                .fetch_one(&self.pool)
                .await
                .context("Person not found")?;

        Ok(person)
    }

    pub async fn new_person(person: Person) -> Result<Person> {
        get_db().await.new_person_impl(person).await
    }
    pub async fn get_person_by_id(id: u32) -> Result<Person> {
        let result: Result<Person> = get_db().await.get_person_by_id_impl(id).await;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_person_creation() {
        let person = Person {
            id: None,
            name: "Alice".to_string(),
            email: "alice@example.com".to_string(),
        };
        assert_eq!(person.name, "Alice");
        assert_eq!(person.email, "alice@example.com");
    }

    #[tokio::test]
    async fn test_init_db() {
        let result = init_db().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_and_get_person() {
        init_db().await.unwrap();
        
        let person = Person {
            id: None,
            name: "Bob".to_string(),
            email: "bob@example.com".to_string(),
        };
        
        let created = Database::new_person(person).await.unwrap();
        assert!(created.id.is_some());
        
        let retrieved = Database::get_person_by_id(created.id.unwrap() as u32).await.unwrap();
        assert_eq!(retrieved.name, "Bob");
    }

    #[test]
    #[ignore]
    fn test_slow_operation() {
        std::thread::sleep(std::time::Duration::from_secs(5));
        assert!(true);
    }
}

pub unsafe fn raw_pointer_access(ptr: *const i32) -> i32 {
    *ptr
}

pub const fn buffer_size() -> usize {
    1024
}

pub const fn max_connections() -> usize {
    100
}

pub struct Wrapper<T: Clone> {
    pub value: T,
}

impl<T: Clone> Wrapper<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
    
    pub fn get(&self) -> T {
        self.value.clone()
    }
}
