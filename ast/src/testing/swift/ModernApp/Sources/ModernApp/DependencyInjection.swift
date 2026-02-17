import Foundation
import Combine

class DependencyContainer: ObservableObject {
    let profileService: ProfileService
    let chatManager: ChatManager
    let apiClient: APIClient<Any>
    
    init() {
        self.apiClient = APIClient()
        self.profileService = ProfileService(apiClient: apiClient)
        self.chatManager = ChatManager()
    }
}
