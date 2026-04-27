import XCTest
@testable import ModernApp
// @ast node: Class "ProfileTests"
// @ast edge: Operand -> Function "setUp" "ProfileTests.swift"
// @ast node: Function "setUp"
// @ast node: UnitTest "testFetchProfile"
// @ast node: UnitTest "testStatusUpdate"
// @ast node: Import "import-imports-srctestingswiftmodernapptestsmodernapptestsprofiletestsswift-0"

final class ProfileTests: XCTestCase {
    var service: ProfileService!
    var mockClient: APIClient<Profile>!
    
    override func setUp() async throws {
        mockClient = APIClient()
        service = ProfileService(apiClient: mockClient)
    }
    
    func testFetchProfile() async throws {
        let profile = try await service.fetchProfile(id: "test_user")
        XCTAssertEqual(profile.username, "Test User")
    }
    
    func testStatusUpdate() async {
        await service.updateStatus(status: .online)
        // Assert state change
    }
}
