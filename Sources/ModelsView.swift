import SwiftUI
import UniformTypeIdentifiers

struct ModelsView: View {
    @EnvironmentObject private var store: ChatStore
    @State private var importing = false

    var body: some View {
        NavigationStack {
            List {
                ForEach(store.models) { model in
                    Button {
                        Task {
                            do { try await store.selectModel(model) }
                            catch { store.error = error.localizedDescription }
                        }
                    } label: {
                        HStack {
                            VStack(alignment: .leading) {
                                Text(model.name)
                                    .foregroundStyle(.primary)
                                Text(ByteCountFormatter.string(fromByteCount: model.byteCount,
                                                               countStyle: .file))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if store.selectedModelID == model.id {
                                Image(systemName: "checkmark.circle.fill")
                            }
                        }
                    }
                    .swipeActions {
                        Button("Verwijder", role: .destructive) {
                            Task { await store.removeModel(model) }
                        }
                    }
                }
            }
            .navigationTitle("GGUF-modellen")
            .toolbar {
                Button { importing = true } label: { Image(systemName: "plus") }
            }
            .fileImporter(isPresented: $importing,
                          allowedContentTypes: [UTType(filenameExtension: "gguf") ?? .data]) { result in
                switch result {
                case .success(let url): Task { await store.importModel(url) }
                case .failure(let error): store.error = error.localizedDescription
                }
            }
        }
    }
}
