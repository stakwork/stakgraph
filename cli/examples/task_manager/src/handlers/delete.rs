use crate::db::query::db_read;
use crate::db::write::db_delete;

pub fn delete_task(id: u64) -> String {
    if db_read(id).is_none() {
        return String::from("not found");
    }
    db_delete(id);
    format!("deleted task {}", id)
}
