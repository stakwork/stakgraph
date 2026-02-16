import SwiftUI

@main
struct ModernApp: App {
    @StateObject private var dependencyContainer = DependencyContainer()
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(dependencyContainer)
        }
    }
}

struct ContentView: View {
    var body: some View {
        TabView {
            ProfileView()
                .tabItem {
                    Label("Profile", systemImage: "person.circle")
                }
            
            ChatView()
                .tabItem {
                    Label("Chat", systemImage: "message.circle")
                }
        }
    }
}
