import XCTest
@testable import SphinxTestApp
// @ast node: Class "APIIntegrationTests"
// @ast edge: Operand -> Function "setUp" "APIIntegrationTests.swift"
// @ast edge: Operand -> Function "tearDown" "APIIntegrationTests.swift"
// @ast node: Function "setUp"
// @ast node: Function "tearDown"
// @ast node: UnitTest "testFetchUsersIntegration"
// @ast node: Import "import-imports-srctestingswiftlegacyapptestsapiintegrationtestsswift-0"

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
