import XCTest
@testable import SphinxTestApp
// @ast node: Class "APIIntegrationTests"
// @ast edge: Operand -> Function "setUp" "APIIntegrationTests.swift"
// @ast edge: Operand -> Function "tearDown" "APIIntegrationTests.swift"
// @ast node: Function "setUp"
// @ast node: Function "tearDown"
// @ast node: IntegrationTest "testFetchUsersIntegration"
// @ast node: IntegrationTest "testGetPeopleList"
// @ast edge: Calls -> Function "getPeopleList" "API.swift"
// @ast node: IntegrationTest "testUpdateProfile"
// @ast edge: Calls -> Function "updatePeopleProfileWith" "API.swift"
// @ast edge: Calls -> Function "createRequest" "API.swift"
// @ast node: Import "import-imports-srctestingswiftlegacyappintegrationtestsapiintegrationtestsswift-0"

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

    func testGetPeopleList() {
        let expectation = XCTestExpectation(description: "Get people list")

        mockAPI.getPeopleList { result in
            XCTAssertNotNil(result)
            expectation.fulfill()
        }

        wait(for: [expectation], timeout: 5.0)
    }

    func testUpdateProfile() {
        let request = mockAPI.createRequest(method: "PUT", path: "/person")
        XCTAssertNotNil(request)
        mockAPI.updatePeopleProfileWith(id: 1, name: "Updated") { result in
            XCTAssertNotNil(result)
        }
    }
}
