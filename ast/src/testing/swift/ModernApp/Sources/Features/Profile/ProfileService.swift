import Foundation

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
