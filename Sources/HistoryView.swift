import SwiftUI
import UniformTypeIdentifiers

struct HistoryView: View {
    @EnvironmentObject private var store: ChatStore
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @State private var matches: [ChatMessage] = []
    @State private var importing = false
    @State private var exportedURL: URL?

    var body: some View {
        List {
            if query.isEmpty {
                ForEach(store.chats) { chat in
                    Button {
                        Task { await store.openChat(chat); dismiss() }
                    } label: {
                        VStack(alignment: .leading) {
                            Text(chat.title).foregroundStyle(.primary)
                            Text(chat.updated, style: .date)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .swipeActions {
                        Button("Verwijder", role: .destructive) {
                            Task { await store.deleteChat(chat) }
                        }
                    }
                }
            } else {
                ForEach(matches) { message in
                    Button {
                        if let chat = store.chats.first(where: { $0.id == message.chatID }) {
                            Task { await store.openChat(chat); dismiss() }
                        }
                    } label: {
                        Text(message.content)
                            .lineLimit(3)
                            .foregroundStyle(.primary)
                    }
                }
            }
        }
        .searchable(text: $query)
        .onChange(of: query) { _, newValue in
            Task { matches = newValue.isEmpty ? [] : await store.search(newValue) }
        }
        .navigationTitle("Geschiedenis")
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button("Importeer") { importing = true }
                Button("Exporteer") {
                    Task { exportedURL = await store.exportHistory() }
                }
            }
        }
        .safeAreaInset(edge: .bottom) {
            if let exportedURL {
                ShareLink(item: exportedURL) {
                    Label("Deel lokale JSON-export", systemImage: "square.and.arrow.up")
                        .padding()
                        .frame(maxWidth: .infinity)
                }
                .background(.regularMaterial)
            }
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.json]) { result in
            switch result {
            case .success(let url): Task { await store.importHistory(url) }
            case .failure(let error): store.error = error.localizedDescription
            }
        }
    }
}
