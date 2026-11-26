import Foundation
import SQLite3

actor ChatDatabase {
    enum DatabaseError: LocalizedError {
        case sqlite(String)

        var errorDescription: String? {
            if case .sqlite(let description) = self { return description }
            return nil
        }
    }

    private struct Archive: Codable {
        let chats: [ChatRecord]
        let messages: [ChatMessage]
    }

    private let db: OpaquePointer
    private let transient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    init() throws {
        let support = try FileManager.default.url(for: .applicationSupportDirectory,
                                                   in: .userDomainMask,
                                                   appropriateFor: nil,
                                                   create: true)
        let folder = support.appendingPathComponent("JJS", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let path = folder.appendingPathComponent("jjs.sqlite").path
        var connection: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX
        guard sqlite3_open_v2(path, &connection, flags, nil) == SQLITE_OK,
              let connection else {
            let message = connection.map { String(cString: sqlite3_errmsg($0)) } ?? "SQLite openen mislukt"
            if let connection { sqlite3_close(connection) }
            throw DatabaseError.sqlite(message)
        }
        db = connection
        let schema = """
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS chats (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          updated REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
          id TEXT PRIMARY KEY,
          chat TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
          seq INTEGER NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created REAL NOT NULL,
          UNIQUE(chat, seq)
        );
        CREATE INDEX IF NOT EXISTS messages_chat_seq ON messages(chat, seq);
        """
        var errorPointer: UnsafeMutablePointer<CChar>?
        guard sqlite3_exec(connection, schema, nil, nil, &errorPointer) == SQLITE_OK else {
            let message = errorPointer.map { String(cString: $0) } ?? "SQLite schema mislukt"
            sqlite3_free(errorPointer)
            sqlite3_close(connection)
            throw DatabaseError.sqlite(message)
        }
    }

    deinit { sqlite3_close(db) }

    func createChat(title: String) throws -> ChatRecord {
        let chat = ChatRecord(id: UUID(), title: title, updated: Date())
        try saveChat(chat)
        return chat
    }

    func saveChat(_ chat: ChatRecord) throws {
        let statement = try prepare("""
            INSERT INTO chats(id,title,updated) VALUES(?,?,?)
            ON CONFLICT(id) DO UPDATE SET title=excluded.title,updated=excluded.updated
            """)
        defer { sqlite3_finalize(statement) }
        bind(chat.id.uuidString, at: 1, in: statement)
        bind(chat.title, at: 2, in: statement)
        sqlite3_bind_double(statement, 3, chat.updated.timeIntervalSince1970)
        try stepDone(statement)
    }

    func chats() throws -> [ChatRecord] {
        let statement = try prepare("SELECT id,title,updated FROM chats ORDER BY updated DESC")
        defer { sqlite3_finalize(statement) }
        var result: [ChatRecord] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            guard let id = UUID(uuidString: columnText(statement, 0)) else { continue }
            result.append(ChatRecord(id: id,
                                     title: columnText(statement, 1),
                                     updated: Date(timeIntervalSince1970: sqlite3_column_double(statement, 2))))
        }
        return result
    }

    func append(chatID: UUID, role: MessageRole, content: String) throws -> ChatMessage {
        let sequenceStatement = try prepare("SELECT COALESCE(MAX(seq),-1)+1 FROM messages WHERE chat=?")
        bind(chatID.uuidString, at: 1, in: sequenceStatement)
        let sequence = sqlite3_step(sequenceStatement) == SQLITE_ROW
            ? Int(sqlite3_column_int64(sequenceStatement, 0)) : 0
        sqlite3_finalize(sequenceStatement)
        let message = ChatMessage(id: UUID(), chatID: chatID, sequence: sequence,
                                  role: role, content: content, created: Date())
        try saveMessage(message)
        return message
    }

    func saveMessage(_ message: ChatMessage) throws {
        let statement = try prepare("""
            INSERT INTO messages(id,chat,seq,role,content,created) VALUES(?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET content=excluded.content
            """)
        defer { sqlite3_finalize(statement) }
        bind(message.id.uuidString, at: 1, in: statement)
        bind(message.chatID.uuidString, at: 2, in: statement)
        sqlite3_bind_int64(statement, 3, Int64(message.sequence))
        bind(message.role.rawValue, at: 4, in: statement)
        bind(message.content, at: 5, in: statement)
        sqlite3_bind_double(statement, 6, message.created.timeIntervalSince1970)
        try stepDone(statement)
        let update = try prepare("UPDATE chats SET updated=? WHERE id=?")
        defer { sqlite3_finalize(update) }
        sqlite3_bind_double(update, 1, message.created.timeIntervalSince1970)
        bind(message.chatID.uuidString, at: 2, in: update)
        try stepDone(update)
    }

    func messages(in chatID: UUID) throws -> [ChatMessage] {
        let statement = try prepare("""
            SELECT id,seq,role,content,created FROM messages WHERE chat=? ORDER BY seq
            """)
        defer { sqlite3_finalize(statement) }
        bind(chatID.uuidString, at: 1, in: statement)
        var result: [ChatMessage] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            guard let id = UUID(uuidString: columnText(statement, 0)),
                  let role = MessageRole(rawValue: columnText(statement, 2)) else { continue }
            result.append(ChatMessage(id: id, chatID: chatID,
                                      sequence: Int(sqlite3_column_int64(statement, 1)),
                                      role: role, content: columnText(statement, 3),
                                      created: Date(timeIntervalSince1970: sqlite3_column_double(statement, 4))))
        }
        return result
    }

    func search(_ text: String) throws -> [ChatMessage] {
        let statement = try prepare("""
            SELECT id,chat,seq,role,content,created FROM messages
            WHERE instr(lower(content),lower(?)) > 0 ORDER BY created DESC
            """)
        defer { sqlite3_finalize(statement) }
        bind(text, at: 1, in: statement)
        var result: [ChatMessage] = []
        while sqlite3_step(statement) == SQLITE_ROW {
            guard let id = UUID(uuidString: columnText(statement, 0)),
                  let chatID = UUID(uuidString: columnText(statement, 1)),
                  let role = MessageRole(rawValue: columnText(statement, 3)) else { continue }
            result.append(ChatMessage(id: id, chatID: chatID,
                                      sequence: Int(sqlite3_column_int64(statement, 2)),
                                      role: role, content: columnText(statement, 4),
                                      created: Date(timeIntervalSince1970: sqlite3_column_double(statement, 5))))
        }
        return result
    }

    func deleteChat(_ id: UUID) throws {
        let statement = try prepare("DELETE FROM chats WHERE id=?")
        defer { sqlite3_finalize(statement) }
        bind(id.uuidString, at: 1, in: statement)
        try stepDone(statement)
    }

    func exportJSON() throws -> Data {
        let allChats = try chats()
        var allMessages: [ChatMessage] = []
        for chat in allChats { allMessages += try messages(in: chat.id) }
        return try JSONEncoder().encode(Archive(chats: allChats, messages: allMessages))
    }

    func importJSON(_ data: Data) throws {
        let archive = try JSONDecoder().decode(Archive.self, from: data)
        try execute("BEGIN IMMEDIATE")
        do {
            for chat in archive.chats { try saveChat(chat) }
            for message in archive.messages { try saveMessage(message) }
            try execute("COMMIT")
        } catch {
            try? execute("ROLLBACK")
            throw error
        }
    }

    private func prepare(_ sql: String) throws -> OpaquePointer {
        var statement: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &statement, nil) == SQLITE_OK,
              let statement else { throw DatabaseError.sqlite(String(cString: sqlite3_errmsg(db))) }
        return statement
    }

    private func bind(_ value: String, at position: Int32, in statement: OpaquePointer) {
        sqlite3_bind_text(statement, position, value, -1, transient)
    }

    private func columnText(_ statement: OpaquePointer, _ position: Int32) -> String {
        guard let pointer = sqlite3_column_text(statement, position) else { return "" }
        return String(cString: pointer)
    }

    private func stepDone(_ statement: OpaquePointer) throws {
        guard sqlite3_step(statement) == SQLITE_DONE else {
            throw DatabaseError.sqlite(String(cString: sqlite3_errmsg(db)))
        }
    }

    private func execute(_ sql: String) throws {
        guard sqlite3_exec(db, sql, nil, nil, nil) == SQLITE_OK else {
            throw DatabaseError.sqlite(String(cString: sqlite3_errmsg(db)))
        }
    }
}
