use serde_json::Value;
use shared::error::Result;
use tokio::process::Command;

#[derive(Debug, serde::Deserialize)]
struct NodesResponse {
    items: Vec<NodeItem>,
    total_returned: usize,
    total_count: usize,
    total_pages: usize,
    current_page: usize,
}

#[derive(Debug, serde::Deserialize)]
struct NodeItem {
    node_type: String,
    name: String,
    file: String,
    ref_id: String,
    weight: usize,
    test_count: usize,
    covered: bool,
    body_length: usize,
    line_count: usize,
    start: usize,
    end: usize,
    meta: Value,
}

const BASE_URL: &str = "http://localhost:7799/tests/nodes";

async fn curl_endpoint(query_params: &str) -> Result<NodesResponse> {
    let url = format!("{}?{}", BASE_URL, query_params);
    let output = Command::new("curl")
        .arg("-s")
        .arg(&url)
        .output()
        .await
        .map_err(|e| shared::error::Error::Custom(format!("Failed to execute curl: {}", e)))?;

    if !output.status.success() {
        return Err(shared::error::Error::Custom(format!(
            "Curl failed with status: {}",
            output.status
        )));
    }

    let response_text = String::from_utf8(output.stdout)
        .map_err(|e| shared::error::Error::Custom(format!("Invalid UTF-8 response: {}", e)))?;

    serde_json::from_str(&response_text)
        .map_err(|e| shared::error::Error::Custom(format!("Failed to parse JSON: {}", e)))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_basic_node_counts() -> Result<()> {
    let functions = curl_endpoint("node_type=Function&limit=1").await?;
    assert_eq!(functions.total_count, 38, "Expected 38 Function nodes");

    let endpoints = curl_endpoint("node_type=Endpoint&limit=1").await?;
    assert_eq!(endpoints.total_count, 21, "Expected 21 Endpoint nodes");

    let unit_tests = curl_endpoint("node_type=UnitTest&limit=1").await?;
    assert_eq!(unit_tests.total_count, 23, "Expected 23 UnitTest nodes");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_multiple_node_types() -> Result<()> {
    let combined = curl_endpoint("node_type=Function,Endpoint&limit=1").await?;
    assert_eq!(combined.total_count, 59, "Expected 59 combined Function+Endpoint nodes");

    let triple = curl_endpoint("node_type=Function,Endpoint,UnitTest&limit=1").await?;
    assert_eq!(triple.total_count, 82, "Expected 82 combined Function+Endpoint+UnitTest nodes");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_basic_pagination() -> Result<()> {
    let page1 = curl_endpoint("node_type=Function&offset=0&limit=5").await?;
    assert_eq!(page1.items.len(), 5, "First page should have 5 items");
    assert_eq!(page1.total_returned, 5, "total_returned should be 5");
    assert_eq!(page1.current_page, 1, "Should be page 1");
    assert_eq!(page1.total_pages, 8, "Should have 8 total pages (38/5 = 7.6 -> 8)");

    let page2 = curl_endpoint("node_type=Function&offset=5&limit=5").await?;
    assert_eq!(page2.items.len(), 5, "Second page should have 5 items");
    assert_eq!(page2.current_page, 2, "Should be page 2");
    assert_eq!(page2.total_count, 38, "Total count should be consistent");

    let first_page1_name = &page1.items[0].name;
    let first_page2_name = &page2.items[0].name;
    assert_ne!(first_page1_name, first_page2_name, "Different pages should have different items");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pagination_edge_cases() -> Result<()> {
    let last_page = curl_endpoint("node_type=Function&offset=35&limit=5").await?;
    assert_eq!(last_page.items.len(), 3, "Last page should have 3 remaining items (38-35=3)");
    assert_eq!(last_page.current_page, 8, "Should be page 8");

    let beyond_end = curl_endpoint("node_type=Function&offset=50&limit=5").await?;
    assert_eq!(beyond_end.items.len(), 0, "Beyond end should return empty items");
    assert_eq!(beyond_end.total_count, 38, "Total count should still be correct");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_search_functionality() -> Result<()> {
    let search_use = curl_endpoint("node_type=Function&search=use&limit=20").await?;
    assert_eq!(search_use.total_count, 9, "Should find exactly 9 functions containing 'use'");
    
    for item in &search_use.items {
        assert!(item.name.to_lowercase().contains("use"), 
            "All results should contain 'use' in name: {}", item.name);
        assert!(item.file.starts_with("src/testing/nextjs/"), "File should be in nextjs test directory");
        assert!(item.line_count > 0, "Should have positive line count");
        assert_eq!(item.line_count, item.end - item.start + 1, "Line count should match end - start + 1");
    }

    let search_nonexistent = curl_endpoint("node_type=Function&search=zzznever&limit=10").await?;
    assert_eq!(search_nonexistent.total_count, 0, "Should find no results for non-existent search");
    assert_eq!(search_nonexistent.items.len(), 0, "Should return empty items");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sorting_options() -> Result<()> {
    let by_test_count = curl_endpoint("node_type=Function&sort_by_test_count=true&limit=10").await?;
    let first_test_count = by_test_count.items[0].test_count;
    let last_test_count = by_test_count.items.last().unwrap().test_count;
    assert!(first_test_count >= last_test_count, "Should be sorted by test count descending");

    let by_body_length = curl_endpoint("node_type=Function&body_length=true&limit=10").await?;
    let first_body_length = by_body_length.items[0].body_length;
    let last_body_length = by_body_length.items.last().unwrap().body_length;
    assert!(first_body_length >= last_body_length, "Should be sorted by body length descending");

    let by_line_count = curl_endpoint("node_type=Function&line_count=true&limit=10").await?;
    let first_line_count = by_line_count.items[0].line_count;
    let last_line_count = by_line_count.items.last().unwrap().line_count;
    assert!(first_line_count >= last_line_count, "Should be sorted by line count descending");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_coverage_filtering() -> Result<()> {
    let all_functions = curl_endpoint("node_type=Function&limit=1").await?;
    let tested = curl_endpoint("node_type=Function&coverage_filter=tested&limit=20").await?;
    let untested = curl_endpoint("node_type=Function&coverage_filter=untested&limit=20").await?;

    assert_eq!(tested.total_count, 38, "All 38 functions should be considered 'tested'");
    assert_eq!(untested.total_count, 38, "All 38 functions should be considered 'untested'");
    assert_eq!(all_functions.total_count, 38, "Should have 38 total functions");

    for item in &tested.items {
        assert_eq!(item.covered, item.test_count > 0, "Covered status should match test_count > 0");
        assert!(item.file.contains("nextjs/"), "Should be from nextjs test data");
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_large_pagination() -> Result<()> {
    let large_limit = curl_endpoint("node_type=Function,Endpoint,UnitTest&limit=50").await?;
    assert!(large_limit.items.len() <= 50, "Should not exceed requested limit");
    assert_eq!(large_limit.total_count, 82, "Total should be consistent for combined types");

    let all_items = curl_endpoint("node_type=Function,Endpoint,UnitTest&limit=100").await?;
    assert_eq!(all_items.items.len(), 82, "Should return all 82 items when limit exceeds total");
    assert_eq!(all_items.total_count, 82, "Total should match item count when all returned");
    assert_eq!(all_items.current_page, 1, "Should be page 1 when all items fit");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_search_with_pagination() -> Result<()> {
    let search_results = curl_endpoint("node_type=Function&search=use&limit=5").await?;
    if search_results.total_count > 5 {
        let page1 = curl_endpoint("node_type=Function&search=use&offset=0&limit=3").await?;
        let page2 = curl_endpoint("node_type=Function&search=use&offset=3&limit=3").await?;

        assert_eq!(page1.total_count, page2.total_count, "Search total should be consistent");
        assert!(page1.items.len() <= 3, "Page 1 should respect limit");
        
        let unique_names: std::collections::HashSet<_> = page1.items.iter()
            .chain(page2.items.iter())
            .map(|item| &item.name)
            .collect();
        let total_items = page1.items.len() + page2.items.len();
        assert_eq!(unique_names.len(), total_items, "Pages should not have duplicate items");
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_combined_filters() -> Result<()> {
    let combined = curl_endpoint("node_type=Function,Endpoint&search=use&sort_by_test_count=true&limit=10").await?;
    
    for item in &combined.items {
        assert!(item.node_type == "Function" || item.node_type == "Endpoint", 
            "Should only return Function or Endpoint types");
        assert!(item.name.to_lowercase().contains("use"), 
            "Should contain search term 'use': {}", item.name);
    }

    if combined.items.len() > 1 {
        for i in 0..combined.items.len()-1 {
            assert!(combined.items[i].test_count >= combined.items[i+1].test_count, 
                "Should be sorted by test count descending");
        }
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pagination_boundaries() -> Result<()> {
    let functions = curl_endpoint("node_type=Function&limit=1").await?;
    let total_functions = functions.total_count;

    let last_valid_offset = total_functions - 1;
    let last_item = curl_endpoint(&format!("node_type=Function&offset={}&limit=1", last_valid_offset)).await?;
    assert_eq!(last_item.items.len(), 1, "Should return 1 item at last valid offset");

    let beyond_end = curl_endpoint(&format!("node_type=Function&offset={}&limit=1", total_functions)).await?;
    assert_eq!(beyond_end.items.len(), 0, "Should return 0 items beyond valid range");
    assert_eq!(beyond_end.total_count, total_functions, "Total count should remain consistent");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_specific_known_nodes() -> Result<()> {
    let search_cn = curl_endpoint("node_type=Function&search=cn&limit=10").await?;
    assert_eq!(search_cn.total_count, 1, "Should find exactly 1 'cn' function");
    
    let cn_function = &search_cn.items[0];
    assert_eq!(cn_function.name, "cn", "Function name should be 'cn'");
    assert!(cn_function.file.ends_with("lib/utils.ts"), "Should be in utils.ts");
    assert_eq!(cn_function.line_count, 3, "cn function should have 3 lines");
    assert_eq!(cn_function.body_length, 78, "cn function should have specific body length");
    assert_eq!(cn_function.start, 5, "Should start at line 5");
    assert_eq!(cn_function.end, 7, "Should end at line 7");
    assert!(cn_function.test_count > 0, "cn function should be tested");
    assert!(cn_function.covered, "cn function should be covered");
   
    assert!(cn_function.meta.get("node_key").is_some(), "Should have node_key in meta");
    assert!(cn_function.meta.get("interface").is_some(), "Should have interface in meta");
    assert!(cn_function.meta.get("Data_Bank").is_some(), "Should have Data_Bank in meta");

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_property_relationships() -> Result<()> {
    let functions = curl_endpoint("node_type=Function&limit=20").await?;
    
    for item in &functions.items {
        assert_eq!(item.line_count, item.end - item.start + 1, 
            "Line count should equal end - start + 1 for {}", item.name);
        
        assert_eq!(item.covered, item.test_count > 0, 
            "Covered status should match test_count > 0 for {}", item.name);

        assert!(item.file.starts_with("src/testing/nextjs/"), 
            "File should be in nextjs directory: {}", item.file);

        if item.test_count > 0 {
            assert!(item.weight > 0, "Weight should be positive for tested function: {}", item.name);
        }

        assert!(item.body_length > 0, "Body length should be positive for {}", item.name);
 
        assert!(item.meta.is_object(), "Meta should be an object");
        let meta_obj = item.meta.as_object().unwrap();
        assert!(meta_obj.contains_key("node_key"), "Should have node_key in meta");
        assert!(meta_obj.contains_key("Data_Bank"), "Should have Data_Bank in meta");
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_endpoint_properties() -> Result<()> {
    let endpoints = curl_endpoint("node_type=Endpoint&limit=10").await?;
    assert_eq!(endpoints.total_count, 21, "Should have exactly 21 endpoints");
    
    for item in &endpoints.items {
        assert_eq!(item.node_type, "Endpoint", "Node type should be Endpoint");
        assert!(item.file.contains("/api/"), "Endpoint should be in api directory: {}", item.file);
        assert!(item.file.ends_with("/route.ts"), "Endpoint should be in route.ts file: {}", item.file);
        assert!(item.weight > 0, "Endpoint weight should be positive: {}", item.name);
        assert!(item.body_length > 10, "Endpoint should have substantial body: {}", item.name);
        
        let meta_obj = item.meta.as_object().unwrap();
        assert!(meta_obj.contains_key("node_key"), "Endpoint should have node_key");
        if let Some(interface) = meta_obj.get("interface") {
            assert!(interface.as_str().unwrap().contains("export"), "Interface should contain export");
        }
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_sorting_precision() -> Result<()> {
    let by_test_count = curl_endpoint("node_type=Function&sort_by_test_count=true&limit=5").await?;
    
    let test_counts: Vec<usize> = by_test_count.items.iter().map(|i| i.test_count).collect();
    assert_eq!(test_counts, vec![3, 2, 2, 1, 1], "Should be sorted by test count descending");
    
    let top_tested = &by_test_count.items[0];
    assert_eq!(top_tested.name, "useActions", "Most tested function should be useActions");
    assert_eq!(top_tested.test_count, 3, "useActions should have exactly 3 tests");
    assert!(top_tested.covered, "Most tested function should be covered");
    
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_consistency() -> Result<()> {
    let request1 = curl_endpoint("node_type=Function,Endpoint&offset=10&limit=5").await?;
    let request2 = curl_endpoint("node_type=Function,Endpoint&offset=10&limit=5").await?;

    assert_eq!(request1.total_count, request2.total_count, "Total count should be consistent");
    assert_eq!(request1.items.len(), request2.items.len(), "Item count should be consistent");
    assert_eq!(request1.current_page, request2.current_page, "Page number should be consistent");

    for (item1, item2) in request1.items.iter().zip(request2.items.iter()) {
        assert_eq!(item1.ref_id, item2.ref_id, "Same items should have same ref_id");
        assert_eq!(item1.name, item2.name, "Same items should have same name");
        assert_eq!(item1.file, item2.file, "Same items should have same file");
        assert_eq!(item1.weight, item2.weight, "Same items should have same weight");
        assert_eq!(item1.covered, item2.covered, "Same items should have same coverage status");
    }

    Ok(())
}