use crate::models::task::Task;
use crate::db::query::db_read;

pub fn get_task(id: u64) -> Option<Task> {
    db_read(id)
}
