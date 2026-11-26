import Foundation

actor ModelRepository {
    enum RepositoryError: LocalizedError {
        case notGGUF
        case inaccessible

        var errorDescription: String? {
            switch self {
            case .notGGUF: return "Het gekozen bestand heeft geen GGUF-header."
            case .inaccessible: return "Het gekozen bestand is niet toegankelijk."
            }
        }
    }

    private let root: URL

    init() throws {
        let documents = try FileManager.default.url(for: .documentDirectory,
                                                    in: .userDomainMask,
                                                    appropriateFor: nil,
                                                    create: true)
        root = documents.appendingPathComponent("Models", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    func importModel(from source: URL) throws -> ImportedModel {
        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }

        guard FileManager.default.isReadableFile(atPath: source.path) else {
            throw RepositoryError.inaccessible
        }
        let handle = try FileHandle(forReadingFrom: source)
        defer { try? handle.close() }
        guard try handle.read(upToCount: 4) == Data("GGUF".utf8) else {
            throw RepositoryError.notGGUF
        }

        let id = UUID()
        let folder = root.appendingPathComponent(id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let target = folder.appendingPathComponent(source.lastPathComponent)
        do {
            try FileManager.default.copyItem(at: source, to: target)
            let attributes = try FileManager.default.attributesOfItem(atPath: target.path)
            return ImportedModel(id: id,
                                 name: target.lastPathComponent,
                                 fileURL: target,
                                 byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0)
        } catch {
            try? FileManager.default.removeItem(at: folder)
            throw error
        }
    }

    func models() throws -> [ImportedModel] {
        let folders = try FileManager.default.contentsOfDirectory(at: root,
                                                                    includingPropertiesForKeys: nil)
        return try folders.compactMap { folder in
            guard let id = UUID(uuidString: folder.lastPathComponent),
                  let file = try FileManager.default.contentsOfDirectory(at: folder,
                                                                           includingPropertiesForKeys: nil).first else {
                return nil
            }
            let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
            return ImportedModel(id: id,
                                 name: file.lastPathComponent,
                                 fileURL: file,
                                 byteCount: (attributes[.size] as? NSNumber)?.int64Value ?? 0)
        }.sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    func remove(_ model: ImportedModel) throws {
        try FileManager.default.removeItem(at: root.appendingPathComponent(model.id.uuidString))
    }

    func storage() throws -> PhoneStorage {
        let values = try root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey,
                                                        .volumeTotalCapacityKey])
        return PhoneStorage(availableBytes: values.volumeAvailableCapacityForImportantUsage.map(Int64.init),
                            totalBytes: values.volumeTotalCapacity.map(Int64.init))
    }
}
