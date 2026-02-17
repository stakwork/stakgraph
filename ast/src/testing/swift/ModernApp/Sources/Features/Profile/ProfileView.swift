import SwiftUI

struct ProfileView: View {
    @State private var profile: Profile?
    @State private var isLoading = false
    @EnvironmentObject var container: DependencyContainer
    
    var body: some View {
        VStack {
            if isLoading {
                ProgressView()
            } else if let profile = profile {
                Text(profile.username)
                    .font(.title)
                Text(profile.bio ?? "No bio")
            } else {
                Text("No Profile Loaded")
            }
        }
        .task {
            await loadProfile()
        }
    }
    
    func loadProfile() async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            profile = try await container.profileService.fetchProfile(id: "me")
        } catch {
            print("Error loading profile: \(error)")
        }
    }
}
