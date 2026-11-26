import Foundation

/// Reads the GGUF container without treating a valid header as an executable model.
/// Tensor bytes stay in the mapped file and are never copied into an array here.
struct GGUFFile {
    struct Tensor {
        let name: String
        let dimensions: [UInt64]
        let storageType: UInt32
        let offset: UInt64
    }

    indirect enum Value {
        case unsigned(UInt64)
        case signed(Int64)
        case floating(Double)
        case boolean(Bool)
        case text(String)
        case array([Value])
    }

    enum FileError: LocalizedError {
        case invalid(String)
        var errorDescription: String? {
            if case .invalid(let message) = self { return message }
            return nil
        }
    }

    let version: UInt32
    let metadata: [String: Value]
    let tensors: [String: Tensor]
    let tensorDataOffset: Int
    private let bytes: Data

    init(url: URL) throws {
        let bytes = try Data(contentsOf: url, options: .mappedIfSafe)
        var reader = Reader(bytes: bytes)
        guard try reader.read(4) == Data("GGUF".utf8) else {
            throw FileError.invalid("Dit bestand is geen GGUF.")
        }
        let version = try reader.uint32()
        guard version == 2 || version == 3 else {
            throw FileError.invalid("Onbekende GGUF-versie \(version).")
        }
        let tensorCount = try reader.uint64()
        let metadataCount = try reader.uint64()
        guard tensorCount <= UInt64(reader.remaining / 24),
              metadataCount <= UInt64(reader.remaining / 13) else {
            throw FileError.invalid("De aantallen in de GGUF-header zijn ongeldig.")
        }

        var metadata: [String: Value] = [:]
        for _ in 0..<metadataCount {
            try Task.checkCancellation()
            let key = try reader.string()
            guard !key.isEmpty, metadata[key] == nil else {
                throw FileError.invalid("GGUF bevat een lege of dubbele metadatasleutel.")
            }
            let type = try reader.uint32()
            metadata[key] = try reader.value(type: type)
        }

        let alignment: UInt64
        if case .unsigned(let number)? = metadata["general.alignment"] {
            alignment = number
        } else {
            alignment = 32
        }
        guard alignment > 0, alignment.nonzeroBitCount == 1 else {
            throw FileError.invalid("GGUF-uitlijning is ongeldig.")
        }

        var tensors: [String: Tensor] = [:]
        for _ in 0..<tensorCount {
            try Task.checkCancellation()
            let name = try reader.string()
            let dimensionsCount = try reader.uint32()
            guard !name.isEmpty, tensors[name] == nil,
                  (1...4).contains(dimensionsCount) else {
                throw FileError.invalid("Een GGUF-tensor heeft een ongeldige naam of vorm.")
            }
            var dimensions: [UInt64] = []
            for _ in 0..<dimensionsCount {
                let dimension = try reader.uint64()
                guard dimension > 0 else {
                    throw FileError.invalid("Een GGUF-tensordimensie is nul.")
                }
                dimensions.append(dimension)
            }
            let storageType = try reader.uint32()
            let offset = try reader.uint64()
            guard offset.isMultiple(of: alignment) else {
                throw FileError.invalid("Een GGUF-tensor heeft een ongeldige uitlijning.")
            }
            tensors[name] = Tensor(name: name, dimensions: dimensions,
                                   storageType: storageType, offset: offset)
        }

        let headerEnd = UInt64(reader.position)
        let padding = (alignment - headerEnd % alignment) % alignment
        let (start, overflow) = headerEnd.addingReportingOverflow(padding)
        guard !overflow, let dataOffset = Int(exactly: start), dataOffset <= bytes.count else {
            throw FileError.invalid("GGUF-tensordata ontbreekt.")
        }
        let sortedTensors = tensors.values.sorted { $0.offset < $1.offset }
        for (index, tensor) in sortedTensors.enumerated() {
            guard tensor.offset < UInt64(bytes.count - dataOffset) else {
                throw FileError.invalid("GGUF-tensor \(tensor.name) valt buiten het bestand.")
            }
            if index + 1 < sortedTensors.count,
               sortedTensors[index + 1].offset <= tensor.offset {
                throw FileError.invalid("GGUF-tensoren hebben overlappende offsets.")
            }
        }

        self.version = version
        self.metadata = metadata
        self.tensors = tensors
        self.tensorDataOffset = dataOffset
        self.bytes = bytes
    }

    func withTensorBytes<T>(_ tensor: Tensor,
                            _ body: (UnsafeRawBufferPointer) throws -> T) throws -> T {
        guard let registered = tensors[tensor.name],
              registered.offset == tensor.offset,
              let offset = Int(exactly: tensor.offset),
              offset <= bytes.count - tensorDataOffset else {
            throw FileError.invalid("Tensor valt buiten het GGUF-bestand.")
        }
        let next = tensors.values
            .filter { $0.offset > tensor.offset }
            .map(\.offset)
            .min()
            .flatMap(Int.init(exactly:)) ?? bytes.count - tensorDataOffset
        guard next >= offset, next <= bytes.count - tensorDataOffset else {
            throw FileError.invalid("Tensorgrens valt buiten het GGUF-bestand.")
        }
        return try bytes.withUnsafeBytes { buffer in
            let slice = UnsafeRawBufferPointer(rebasing: buffer[(tensorDataOffset + offset)..<(tensorDataOffset + next)])
            return try body(slice)
        }
    }
}

private struct Reader {
    let bytes: Data
    var position = 0
    var remaining: Int { bytes.count - position }

    mutating func read(_ count: Int) throws -> Data {
        guard count >= 0, count <= remaining else {
            throw GGUFFile.FileError.invalid("GGUF-bestand is afgebroken.")
        }
        defer { position += count }
        return bytes.subdata(in: position..<(position + count))
    }

    mutating func integer<T: FixedWidthInteger>(_ type: T.Type) throws -> T {
        let data = try read(MemoryLayout<T>.size)
        return data.withUnsafeBytes { T(littleEndian: $0.loadUnaligned(as: T.self)) }
    }

    mutating func uint32() throws -> UInt32 { try integer(UInt32.self) }
    mutating func uint64() throws -> UInt64 { try integer(UInt64.self) }

    mutating func string() throws -> String {
        let length = try uint64()
        guard let count = Int(exactly: length), count <= remaining,
              let text = String(data: try read(count), encoding: .utf8) else {
            throw GGUFFile.FileError.invalid("GGUF bevat ongeldige UTF-8-tekst.")
        }
        return text
    }

    mutating func value(type: UInt32, insideArray: Bool = false) throws -> GGUFFile.Value {
        switch type {
        case 0: return .unsigned(UInt64(try integer(UInt8.self)))
        case 1: return .signed(Int64(try integer(Int8.self)))
        case 2: return .unsigned(UInt64(try integer(UInt16.self)))
        case 3: return .signed(Int64(try integer(Int16.self)))
        case 4: return .unsigned(UInt64(try uint32()))
        case 5: return .signed(Int64(try integer(Int32.self)))
        case 6: return .floating(Double(Float(bitPattern: try uint32())))
        case 7:
            let raw = try integer(UInt8.self)
            guard raw <= 1 else { throw GGUFFile.FileError.invalid("GGUF-boolean is ongeldig.") }
            return .boolean(raw == 1)
        case 8: return .text(try string())
        case 9:
            guard !insideArray else {
                throw GGUFFile.FileError.invalid("Geneste GGUF-arrays zijn ongeldig.")
            }
            let elementType = try uint32()
            let count = try uint64()
            guard count <= UInt64(remaining) else {
                throw GGUFFile.FileError.invalid("GGUF-array is afgebroken.")
            }
            var items: [GGUFFile.Value] = []
            for _ in 0..<count {
                try Task.checkCancellation()
                items.append(try value(type: elementType, insideArray: true))
            }
            return .array(items)
        case 10: return .unsigned(try uint64())
        case 11: return .signed(try integer(Int64.self))
        case 12: return .floating(Double(bitPattern: try uint64()))
        default: throw GGUFFile.FileError.invalid("Onbekend GGUF-metadatatype \(type).")
        }
    }
}
