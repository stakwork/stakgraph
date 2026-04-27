import Foundation
import Combine
// @ast node: Class "DependencyContainer"
// @ast node: Import "import-imports-srctestingswiftmodernappsourcesmodernappdependencyinjectionswift-0"

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
