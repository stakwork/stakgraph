import SwiftUI
// @ast node: Class "ModernApp"
// @ast node: Class "ContentView"
// @ast node: Import "import-imports-srctestingswiftmodernappsourcesmodernappappswift-0"

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
