import SwiftUI
import UIKit

private struct ProjectExport: Identifiable {
    let id = UUID()
    let url: URL
}

struct ProjectsView: View {
    @EnvironmentObject private var store: ChatStore
    @State private var name = "MyProject"
    @State private var export: ProjectExport?
    @State private var generating = false

    var body: some View {
        NavigationStack {
            Form {
                Section("Nieuw SwiftUI-project") {
                    TextField("Projectnaam", text: $name)
                    Button("Maak ZIP op deze iPhone") {
                        generating = true
                        let projectName = name
                        Task {
                            do {
                                let url = try await Task.detached {
                                    try ProjectGenerator.create(name: projectName)
                                }.value
                                export = ProjectExport(url: url)
                            } catch {
                                store.error = error.localizedDescription
                            }
                            generating = false
                        }
                    }
                    if generating { ProgressView() }
                }
            }
            .navigationTitle("Projecten")
            .sheet(item: $export) { item in
                ActivitySheet(items: [item.url])
            }
        }
    }
}

private enum ProjectGenerator {
    static func create(name: String) throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("JJS-\(UUID().uuidString)", isDirectory: true)
        let folder = base.appendingPathComponent("MyProject", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let literal = String(reflecting: name)
        let source = """
        import SwiftUI

        @main
        struct MyProjectApp: App {
            var body: some Scene {
                WindowGroup { ContentView() }
            }
        }

        struct ContentView: View {
            var body: some View {
                Text(\(literal))
                    .font(.largeTitle)
                    .padding()
            }
        }
        """
        try Data(source.utf8).write(to: folder.appendingPathComponent("ContentView.swift"))
        let readme = "# \(name)\n\nOpen ContentView.swift in a SwiftUI iOS project.\n"
        try Data(readme.utf8).write(to: folder.appendingPathComponent("README.md"))

        let target = base.appendingPathComponent("MyProject.zip")
        let coordinator = NSFileCoordinator()
        var coordinationError: NSError?
        var copyError: Error?
        coordinator.coordinate(readingItemAt: folder, options: .forUploading,
                               error: &coordinationError) { compressedURL in
            do { try FileManager.default.copyItem(at: compressedURL, to: target) }
            catch { copyError = error }
        }
        if let coordinationError { throw coordinationError }
        if let copyError { throw copyError }
        return target
    }
}

private struct ActivitySheet: UIViewControllerRepresentable {
    let items: [Any]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
