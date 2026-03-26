pub enum Priority {
    Low,
    Medium,
    High,
}

pub enum TaskStatus {
    Pending,
    InProgress,
    Done,
}

pub struct Task {
    pub id: u64,
    pub title: String,
    pub description: String,
    pub priority: Priority,
    pub status: TaskStatus,
}
