import Foundation
// @ast node: Class "Profile"
// @ast node: Class "Status"
// @ast node: Var "id"
// @ast node: Var "username"
// @ast node: Var "bio"
// @ast node: Var "avatarURL"
// @ast node: Var "status"
// @ast node: Import "import-imports-srctestingswiftmodernappsourcesfeaturesprofileprofilemodelswift-0"

struct Profile: Codable, Identifiable {
    let id: String
    var username: String
    var bio: String?
    var avatarURL: URL?
    
    enum Status: String, Codable {
        case online, offline, away, busy
    }
    
    var status: Status
}
