import SwiftUI
import UniformTypeIdentifiers

struct ChatView: View {
    @EnvironmentObject private var store: ChatStore
    @State private var importing = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 12) {
                            ForEach(store.messages) { message in
                                HStack {
                                    if message.role == .user { Spacer(minLength: 36) }
                                    Text(message.content)
                                        .padding(12)
                                        .background(message.role == .user ? Color.blue : Color.gray.opacity(0.25))
                                        .clipShape(RoundedRectangle(cornerRadius: 14))
                                        .textSelection(.enabled)
                                    if message.role == .assistant { Spacer(minLength: 36) }
                                }
                                .id(message.id)
                            }
                            if !store.liveResponse.isEmpty {
                                Text(store.liveResponse)
                                    .padding(12)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Color.gray.opacity(0.25))
                                    .clipShape(RoundedRectangle(cornerRadius: 14))
                                    .textSelection(.enabled)
                                    .id("live")
                            }
                        }
                        .padding()
                    }
                    .onChange(of: store.messages.count) { _, _ in
                        if let last = store.messages.last { proxy.scrollTo(last.id) }
                    }
                    .onChange(of: store.liveResponse) { _, _ in
                        proxy.scrollTo("live")
                    }
                }
                Divider()
                HStack(alignment: .bottom) {
                    Button { importing = true } label: { Image(systemName: "paperclip") }
                        .padding(8)
                    TextField("Bericht aan het gekozen model", text: $store.draft, axis: .vertical)
                        .lineLimit(1...8)
                    Button {
                        Task { await store.sendDraft() }
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                    }
                    .padding(8)
                }
                .padding(.horizontal, 8)
            }
            .navigationTitle("Chat")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    NavigationLink("Geschiedenis") { HistoryView() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button { Task { await store.newChat() } } label: {
                        Image(systemName: "square.and.pencil")
                    }
                }
            }
            .fileImporter(isPresented: $importing, allowedContentTypes: [.item]) { result in
                switch result {
                case .success(let url): Task { await store.attach(url) }
                case .failure(let error): store.error = error.localizedDescription
                }
            }
        }
    }
}
