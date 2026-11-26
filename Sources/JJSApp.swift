import SwiftUI

@main
struct JJSApp: App {
    @StateObject private var store = ChatStore()

    var body: some Scene {
        WindowGroup {
            TabView {
                HomeView()
                    .tabItem { Label("Home", systemImage: "house") }
                ModelsView()
                    .tabItem { Label("Modellen", systemImage: "square.stack.3d.up") }
                ChatView()
                    .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
                ProjectsView()
                    .tabItem { Label("Projecten", systemImage: "shippingbox") }
            }
            .preferredColorScheme(.dark)
            .environmentObject(store)
            .alert("Fout", isPresented: Binding(
                get: { store.error != nil },
                set: { if !$0 { store.error = nil } }
            )) {
                Button("OK") { store.error = nil }
            } message: {
                Text(store.error ?? "")
            }
        }
    }
}
