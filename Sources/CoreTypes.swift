import Foundation

enum MessageRole: String, Codable, Sendable {
    case user
    case assistant
}

struct ChatRecord: Identifiable, Codable, Sendable {
    let id: UUID
    var title: String
    var updated: Date
}

struct ChatMessage: Identifiable, Codable, Sendable {
    let id: UUID
    let chatID: UUID
    let sequence: Int
    let role: MessageRole
    var content: String
    let created: Date
}

struct ImportedModel: Identifiable, Sendable {
    let id: UUID
    let name: String
    let fileURL: URL
    let byteCount: Int64
}

struct PhoneStorage: Sendable {
    let availableBytes: Int64?
    let totalBytes: Int64?
}
