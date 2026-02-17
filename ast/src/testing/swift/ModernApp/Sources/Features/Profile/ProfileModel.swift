import Foundation

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
