use anyhow::Result;
use rust::db::{init_db, Database, Person};

#[tokio::test]
async fn e2e_test_full_crud_workflow() -> Result<()> {
    init_db().await?;
    
    let person = Person {
        id: None,
        name: "E2E User".to_string(),
        email: "e2e@test.com".to_string(),
    };
    
    let created = Database::new_person(person).await?;
    let person_id = created.id.expect("Created person should have ID");
    
    let retrieved = Database::get_person_by_id(person_id as u32).await?;
    assert_eq!(retrieved.name, "E2E User");
    assert_eq!(retrieved.email, "e2e@test.com");
    assert_eq!(retrieved.id, Some(person_id));
    
    Ok(())
}

#[tokio::test]
async fn e2e_test_user_journey_create_multiple_and_retrieve() {
    init_db().await.unwrap();
    
    let users = vec![
        ("Alice", "alice@e2e.com"),
        ("Bob", "bob@e2e.com"),
        ("Charlie", "charlie@e2e.com"),
    ];
    
    let mut created_ids = Vec::new();
    
    for (name, email) in &users {
        let person = Person {
            id: None,
            name: name.to_string(),
            email: email.to_string(),
        };
        
        let created = Database::new_person(person).await.unwrap();
        created_ids.push(created.id.unwrap());
    }
    
    for (i, id) in created_ids.iter().enumerate() {
        let retrieved = Database::get_person_by_id(*id as u32).await.unwrap();
        assert_eq!(retrieved.name, users[i].0);
        assert_eq!(retrieved.email, users[i].1);
    }
}

#[tokio::test]
async fn e2e_test_error_scenario_not_found() {
    init_db().await.unwrap();
    
    let result = Database::get_person_by_id(99999).await;
    
    assert!(result.is_err(), "Should return error for non-existent user");
    let error = result.unwrap_err();
    assert!(error.to_string().contains("not found") || error.to_string().contains("Person"));
}

#[tokio::test]
async fn e2e_test_data_persistence_verification() -> Result<()> {
    init_db().await?;
    
    let test_data = vec![
        Person { id: None, name: "User1".to_string(), email: "u1@test.com".to_string() },
        Person { id: None, name: "User2".to_string(), email: "u2@test.com".to_string() },
    ];
    
    let mut ids = Vec::new();
    for person in test_data {
        let created = Database::new_person(person).await?;
        ids.push(created.id.unwrap());
    }
    
    for id in ids {
        let exists = Database::get_person_by_id(id as u32).await;
        assert!(exists.is_ok(), "Previously created person should exist");
    }
    
    Ok(())
}

#[tokio::test]
async fn e2e_test_sequential_operations() -> Result<()> {
    init_db().await?;
    
    let person1 = Person {
        id: None,
        name: "First".to_string(),
        email: "first@test.com".to_string(),
    };
    let created1 = Database::new_person(person1).await?;
    
    let retrieved1 = Database::get_person_by_id(created1.id.unwrap() as u32).await?;
    assert_eq!(retrieved1.name, "First");
    
    let person2 = Person {
        id: None,
        name: "Second".to_string(),
        email: "second@test.com".to_string(),
    };
    let created2 = Database::new_person(person2).await?;
    
    let retrieved2 = Database::get_person_by_id(created2.id.unwrap() as u32).await?;
    assert_eq!(retrieved2.name, "Second");
    
    let retrieved1_again = Database::get_person_by_id(created1.id.unwrap() as u32).await?;
    assert_eq!(retrieved1_again.name, "First");
    
    Ok(())
}

#[tokio::test]
async fn e2e_test_empty_database_query() {
    init_db().await.unwrap();
    
    let result = Database::get_person_by_id(1).await;
    assert!(result.is_err());
}

#[test]
fn e2e_test_system_configuration() {
    use rust::db::{buffer_size, max_connections};
    
    let buffer = buffer_size();
    let max_conn = max_connections();
    
    assert!(buffer > 0, "Buffer size should be positive");
    assert!(max_conn > 0, "Max connections should be positive");
    assert!(buffer >= 512, "Buffer should be at least 512 bytes");
}

#[tokio::test]
#[ignore]
async fn e2e_test_long_running_operation() {
    init_db().await.unwrap();
    
    for i in 0..100 {
        let person = Person {
            id: None,
            name: format!("User{}", i),
            email: format!("user{}@bulk.com", i),
        };
        Database::new_person(person).await.unwrap();
    }
}
