import Foundation

enum Endpoint {
    case userProfile(id: String)
    case chatMessages(roomId: String)
    case updateStatus
    
    var path: String {
        switch self {
        case .userProfile(let id):
            return "/api/v1/users/\(id)"
        case .chatMessages(let roomId):
            return "/api/v1/chat/\(roomId)/messages"
        case .updateStatus:
            return "/api/v1/user/status"
        }
    }
    
    var url: URL? {
        var components = URLComponents()
        components.scheme = "https"
        components.host = "api.example.com"
        components.path = path
        return components.url
    }
}
