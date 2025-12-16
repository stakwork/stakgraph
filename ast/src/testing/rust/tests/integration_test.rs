use anyhow::Result;
use rust::db::{init_db, Database, Person};
use rust::types::{process, Cache, ApiResult};
use rust::traits::{Greeter, Greet, SimpleProcessor, Processor};

#[tokio::test]
async fn integration_test_database_full_flow() -> Result<()> {
    init_db().await?;
    
    let person = Person {
        id: None,
        name: "Integration Test User".to_string(),
        email: "integration@test.com".to_string(),
    };
    
    let created = Database::new_person(person).await?;
    assert!(created.id.is_some(), "Created person should have ID");
    
    let retrieved = Database::get_person_by_id(created.id.unwrap() as u32).await?;
    assert_eq!(retrieved.name, "Integration Test User");
    assert_eq!(retrieved.email, "integration@test.com");
    
    Ok(())
}

#[tokio::test]
async fn integration_test_multiple_database_operations() {
    init_db().await.unwrap();
    
    let people = vec![
        Person { id: None, name: "User A".to_string(), email: "a@test.com".to_string() },
        Person { id: None, name: "User B".to_string(), email: "b@test.com".to_string() },
    ];
    
    let mut ids = Vec::new();
    for person in people {
        let created = Database::new_person(person).await.unwrap();
        ids.push(created.id.unwrap());
    }
    
    for id in ids {
        let retrieved = Database::get_person_by_id(id as u32).await;
        assert!(retrieved.is_ok());
    }
}

#[tokio::test]
async fn integration_test_database_with_serialization() -> Result<()> {
    init_db().await?;
    
    let person = Person {
        id: None,
        name: "Serializable User".to_string(),
        email: "serial@test.com".to_string(),
    };
    
    let created = Database::new_person(person).await?;
    let json = process(&created)?;
    
    assert!(json.contains("Serializable User"));
    assert!(json.contains("serial@test.com"));
    
    Ok(())
}

#[test]
fn integration_test_types_with_traits() {
    let greeter = Greeter {
        name: "Integration".to_string(),
    };
    
    let greeting = greeter.greet();
    assert_eq!(greeting, "Hello, Integration!");
}

#[test]
fn integration_test_cache_with_processor() {
    let mut cache: Cache<i32, String> = Cache::new();
    let processor = SimpleProcessor;
    
    let items = vec![1, 2, 3];
    let processed = processor.batch_process(items);
    
    for (i, value) in processed.iter().enumerate() {
        cache.insert(*value, format!("Item {}", i));
    }
    
    assert_eq!(cache.get(&1), Some(&"Item 0".to_string()));
    assert_eq!(cache.get(&2), Some(&"Item 1".to_string()));
}

#[test]
fn integration_test_api_result_with_cache() {
    let mut cache: Cache<String, ApiResult<i32, String>> = Cache::new();
    
    cache.insert("success".to_string(), ApiResult::Success(100));
    cache.insert("failure".to_string(), ApiResult::Failure("error".to_string()));
    cache.insert("pending".to_string(), ApiResult::Pending);
    
    let success = cache.get(&"success".to_string()).unwrap();
    assert!(success.is_success());
    
    let failure = cache.get(&"failure".to_string()).unwrap();
    assert!(!failure.is_success());
}

#[tokio::test]
async fn integration_test_concurrent_database_access() {
    init_db().await.unwrap();
    
    let handles: Vec<_> = (0..5).map(|i| {
        tokio::spawn(async move {
            let person = Person {
                id: None,
                name: format!("Concurrent User {}", i),
                email: format!("user{}@test.com", i),
            };
            Database::new_person(person).await
        })
    }).collect();
    
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn integration_test_database_error_handling() {
    init_db().await.unwrap();
    
    let result = Database::get_person_by_id(999999).await;
    assert!(result.is_err(), "Should fail for non-existent ID");
}
