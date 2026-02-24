import XCTest
@testable import SphinxTestApp

final class APIIntegrationTests: XCTestCase {
    
    var mockAPI: API!
    
    override func setUp() {
        super.setUp()
        mockAPI = API()
    }
    
    override func tearDown() {
        mockAPI = nil
        super.tearDown()
    }
    
    func testFetchUsersIntegration() {
        let expectation = XCTestExpectation(description: "Fetch users")
        
        mockAPI.fetchUsers { result in
            switch result {
            case .success(let users):
                XCTAssertGreaterThan(users.count, 0, "Should return users")
            case .failure:
                XCTFail("Should not fail")
            }
            expectation.fulfill()
        }
        
        wait(for: [expectation], timeout: 5.0)
    }
}
