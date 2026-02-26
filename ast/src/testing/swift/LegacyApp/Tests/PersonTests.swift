import XCTest
@testable import SphinxTestApp

final class PersonTests: XCTestCase {
    
    func testPersonInitialization() {
        let person = Person(name: "Alice", email: "alice@example.com")
        XCTAssertEqual(person.name, "Alice")
        XCTAssertEqual(person.email, "alice@example.com")
    }
    
    func testPersonDescription() {
        let person = Person(name: "Bob", email: "bob@example.com")
        let description = "\(person)"
        XCTAssertTrue(description.contains("Bob"))
    }
}
