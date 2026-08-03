use std::sync::Arc;

// @ast node: DataModel "Pool"
// @ast node: DataModel "DataStore"
// @ast node: DataModel "Repo"
// @ast node: DataModel "ArcRepo"
// @ast node: Class "Pool"
// @ast edge: Operand -> Function "count" "method_chain.rs" [operand=Pool]
// @ast edge: Operand -> Function "new" "method_chain.rs" [operand=Pool]
// @ast node: Class "DataStore"
// @ast edge: Operand -> Function "count" "method_chain.rs" [operand=DataStore]
// @ast edge: Operand -> Function "new" "method_chain.rs" [operand=DataStore]
// @ast node: Class "Repo"
// @ast edge: Operand -> Function "total" "method_chain.rs"
// @ast edge: Operand -> Function "new" "method_chain.rs" [operand=Repo]
// @ast node: Class "ArcRepo"
// @ast edge: Operand -> Function "arc_total" "method_chain.rs"
// @ast node: Function "count" [operand=Pool]
// @ast node: Function "new" [operand=Pool]
// @ast node: Function "count" [operand=DataStore]
// @ast edge: Calls -> Function "count" "method_chain.rs" [operand=Pool]
// @ast node: Function "new" [operand=DataStore]
// @ast edge: Calls -> Function "new" "method_chain.rs" [operand=Pool]
// @ast node: Function "total"
// @ast edge: Calls -> Function "count" "method_chain.rs" [operand=DataStore]
// @ast node: Function "new" [operand=Repo]
// @ast edge: Calls -> Function "new" "method_chain.rs" [operand=DataStore]
// @ast node: Function "arc_total"
// @ast edge: Calls -> Function "count" "method_chain.rs" [operand=DataStore]
// @ast node: Function "pipeline"
// @ast edge: Calls -> Function "new" "method_chain.rs" [operand=Repo]
// @ast edge: Calls -> Function "total" "method_chain.rs"
// @ast node: Function "make_data_store"
// @ast node: Function "factory_pipeline"
// @ast edge: Calls -> Function "make_data_store" "method_chain.rs"
// @ast edge: Calls -> Function "count" "method_chain.rs" [operand=DataStore]

/// Leaf type — Pool::count is the first "count" defined in this file.
pub struct Pool;

impl Pool {
    pub fn new() -> Self {
        Self
    }

    pub fn count(&self) -> usize {
        0
    }
}

/// DataStore wraps a Pool field.
pub struct DataStore {
    pool: Pool,
}

impl DataStore {
    pub fn new() -> Self {
        Self { pool: Pool::new() }
    }

    /// Calls Pool::count via 2-level field_expression: self.pool.count()
    /// The outer field_expression value is another field_expression (not a bare identifier),
    /// so the existing tree-sitter query cannot capture the receiver type.
    /// The hybrid resolver resolves this correctly via struct_fields lookup.
    pub fn count(&self) -> usize {
        self.pool.count()
    }
}

/// Repo holds a DataStore field.
pub struct Repo {
    data: DataStore,
}

impl Repo {
    pub fn new() -> Self {
        Self {
            data: DataStore::new(),
        }
    }

    /// Calls DataStore::count via self.data.count() — requires hybrid resolver
    /// because "count" is ambiguous in this file (Pool::count comes first in graph order).
    pub fn total(&self) -> usize {
        self.data.count()
    }
}

/// ArcRepo holds an Arc<DataStore> field — tests transparent-wrapper stripping.
pub struct ArcRepo {
    inner: Arc<DataStore>,
}

impl ArcRepo {
    /// Calls DataStore::count via self.inner.count() where inner: Arc<DataStore>.
    /// strip_rust_type unwraps Arc<DataStore> → DataStore, enabling correct dispatch.
    pub fn arc_total(&self) -> usize {
        self.inner.count()
    }
}

/// Free function: tests constructor binding (let repo = Repo::new()) followed by
/// a method call on the bound variable (repo.total()).
pub fn pipeline() -> usize {
    let repo = Repo::new();
    repo.total()
}

/// Factory function: return type seeds fn_returns so that callers can type-bind
/// the result variable without an explicit type annotation.
pub fn make_data_store() -> DataStore {
    DataStore::new()
}

/// Tests the fn_returns path: let store = make_data_store() binds store → DataStore,
/// then store.count() resolves to DataStore::count via struct_fields dispatch.
pub fn factory_pipeline() -> usize {
    let store = make_data_store();
    store.count()
}
