import Foundation
// @ast node: Class "ProfileService"
// @ast edge: Operand -> Function "fetchProfile" "ProfileService.swift"
// @ast edge: Operand -> Function "updateStatus" "ProfileService.swift"
// @ast node: Function "fetchProfile"
// @ast edge: Calls -> Function "fetch" "APIClient.swift"
// @ast node: Function "updateStatus"
// @ast node: Import "import-imports-srctestingswiftmodernappsourcesfeaturesprofileprofileserviceswift-0"

actor ProfileService {
    private let apiClient: APIClient<Profile>
    private var cache: [String: Profile] = [:]
    
    init(apiClient: APIClient<Profile>) {
        self.apiClient = apiClient
    }
    
    func fetchProfile(id: String) async throws -> Profile {
        if let cached = cache[id] {
            return cached
        }
        
        let profile = try await apiClient.fetch(endpoint: .userProfile(id: id))
        cache[id] = profile
        return profile
    }
    
    func updateStatus(status: Profile.Status) async {
        // Mock network call
        try? await Task.sleep(nanoseconds: 1_000_000_000)
        print("Status updated to \(status)")
    }
}
