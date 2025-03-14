use serde::{Deserialize, Serialize};
use sqlx::FromRow;

#[derive(Debug, Serialize, Deserialize, FromRow)]
pub struct Person {
    #[serde(skip_deserializing)]
    pub id: Option<i32>,
    pub name: String,
    pub email: String,
}
