import Foundation
import Combine

@MainActor
final class ChatStore: ObservableObject {
    @Published private(set) var models: [ImportedModel] = []
    @Published private(set) var chats: [ChatRecord] = []
    @Published private(set) var messages: [ChatMessage] = []
    @Published private(set) var selectedModelID: UUID?
    @Published private(set) var activeChatID: UUID?
    @Published private(set) var liveResponse = ""
    @Published private(set) var isGenerating = false
    @Published private(set) var storage: PhoneStorage?
    @Published var draft = ""
    @Published var error: String?

    private let engine = LocalInferenceEngine()
    private let attachmentReader = AttachmentReader()
    private var modelRepository: ModelRepository?
    private var database: ChatDatabase?
    private var queue: [String] = []

    init() { Task { await initialize() } }

    private func initialize() async {
        do {
            let repository = try ModelRepository()
            let database = try ChatDatabase()
            modelRepository = repository
            self.database = database
            models = try await repository.models()
            storage = try await repository.storage()
            chats = try await database.chats()
            if let saved = UserDefaults.standard.string(forKey: "selectedModelID"),
               let id = UUID(uuidString: saved),
               let model = models.first(where: { $0.id == id }) {
                try await selectModel(model)
            }
            if let first = chats.first { await openChat(first) }
        } catch { self.error = error.localizedDescription }
    }

    func refreshStorage() async {
        do { storage = try await modelRepository?.storage() }
        catch { self.error = error.localizedDescription }
    }

    func importModel(_ url: URL) async {
        do {
            guard let repository = modelRepository else { return }
            let model = try await repository.importModel(from: url)
            models = try await repository.models()
            storage = try await repository.storage()
            try await selectModel(model)
        } catch { self.error = error.localizedDescription }
    }

    func selectModel(_ model: ImportedModel) async throws {
        try await engine.selectModel(at: model.fileURL)
        selectedModelID = model.id
        UserDefaults.standard.set(model.id.uuidString, forKey: "selectedModelID")
    }

    func removeModel(_ model: ImportedModel) async {
        do {
            try await modelRepository?.remove(model)
            models = try await modelRepository?.models() ?? []
            if selectedModelID == model.id {
                selectedModelID = nil
                UserDefaults.standard.removeObject(forKey: "selectedModelID")
            }
            storage = try await modelRepository?.storage()
        } catch { self.error = error.localizedDescription }
    }

    func newChat() async {
        do {
            guard let database else { return }
            let chat = try await database.createChat(title: "Nieuw gesprek")
            chats = try await database.chats()
            activeChatID = chat.id
            messages = []
            liveResponse = ""
        } catch { self.error = error.localizedDescription }
    }

    func openChat(_ chat: ChatRecord) async {
        do {
            activeChatID = chat.id
            messages = try await database?.messages(in: chat.id) ?? []
            liveResponse = ""
        } catch { self.error = error.localizedDescription }
    }

    func deleteChat(_ chat: ChatRecord) async {
        do {
            try await database?.deleteChat(chat.id)
            chats = try await database?.chats() ?? []
            if activeChatID == chat.id {
                activeChatID = nil
                messages = []
                liveResponse = ""
            }
        } catch { self.error = error.localizedDescription }
    }

    func sendDraft() async {
        let text = draft
        draft = ""
        queue.append(text)
        if !isGenerating { await drainQueue() }
    }

    private func drainQueue() async {
        guard let database else { return }
        isGenerating = true
        defer { isGenerating = false }
        while !queue.isEmpty {
            let text = queue.removeFirst()
            do {
                let chatID: UUID
                if let activeChatID {
                    chatID = activeChatID
                } else {
                    let chat = try await database.createChat(title: String(text.prefix(60)))
                    chatID = chat.id
                    activeChatID = chat.id
                }
                let userMessage = try await database.append(chatID: chatID, role: .user, content: text)
                if activeChatID == chatID { messages.append(userMessage) }
                let conversation = try await database.messages(in: chatID)
                var finalResponse = ""
                for try await snapshot in await engine.stream(messages: conversation) {
                    finalResponse = snapshot
                    if activeChatID == chatID { liveResponse = snapshot }
                }
                let answer = try await database.append(chatID: chatID,
                                                       role: .assistant,
                                                       content: finalResponse)
                if activeChatID == chatID {
                    messages.append(answer)
                    liveResponse = ""
                }
                chats = try await database.chats()
            } catch {
                self.error = error.localizedDescription
                liveResponse = ""
            }
        }
    }

    func attach(_ url: URL) async {
        do {
            let extracted = try await attachmentReader.text(from: url)
            draft += (draft.isEmpty ? "" : "\n\n") + extracted
        } catch { self.error = error.localizedDescription }
    }

    func search(_ term: String) async -> [ChatMessage] {
        do { return try await database?.search(term) ?? [] }
        catch { self.error = error.localizedDescription; return [] }
    }

    func exportHistory() async -> URL? {
        do {
            guard let data = try await database?.exportJSON() else { return nil }
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("JJS-history.json")
            try data.write(to: url, options: .atomic)
            return url
        } catch { self.error = error.localizedDescription; return nil }
    }

    func importHistory(_ url: URL) async {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            try await database?.importJSON(Data(contentsOf: url))
            chats = try await database?.chats() ?? []
        } catch { self.error = error.localizedDescription }
    }
}
