import Foundation

/// Decodes GGML rows directly from the GGUF tensor storage.
/// The tensor format is taken from the file, not from a model-name whitelist.
struct GGMLTensor {
    enum DecodeError: LocalizedError {
        case invalidShape
        case unsupportedStorage(UInt32)
        case truncated

        var errorDescription: String? {
            switch self {
            case .invalidShape: return "GGUF-tensor heeft een ongeldige vorm."
            case .unsupportedStorage(let type): return "GGML-opslagtype \(type) is nog niet geïmplementeerd."
            case .truncated: return "GGUF-tensordata is afgebroken."
            }
        }
    }

    let file: GGUFFile
    let descriptor: GGUFFile.Tensor

    var columns: Int? { descriptor.dimensions.first.flatMap(Int.init(exactly:)) }
    var rows: Int? {
        let dimensions = descriptor.dimensions.dropFirst()
        guard !dimensions.isEmpty else { return 1 }
        return dimensions.reduce(1) { product, dimension in
            guard let count = Int(exactly: dimension),
                  product <= Int.max / count else { return 0 }
            return product * count
        }
    }

    func row(_ index: Int) throws -> [Float] {
        guard let columns, let rows, rows > 0, index >= 0, index < rows,
              let layout = Self.layout(for: descriptor.storageType),
              columns.isMultiple(of: layout.elements) else {
            if Self.layout(for: descriptor.storageType) == nil {
                throw DecodeError.unsupportedStorage(descriptor.storageType)
            }
            throw DecodeError.invalidShape
        }
        let blocks = columns / layout.elements
        let (rowSize, overflow) = blocks.multipliedReportingOverflow(by: layout.bytes)
        guard !overflow else { throw DecodeError.invalidShape }
        let (position, positionOverflow) = index.multipliedReportingOverflow(by: rowSize)
        guard !positionOverflow else { throw DecodeError.invalidShape }

        return try file.withTensorBytes(descriptor) { storage in
            guard position <= storage.count, rowSize <= storage.count - position else {
                throw DecodeError.truncated
            }
            let source = storage.baseAddress!.advanced(by: position)
            var values = [Float](repeating: 0, count: columns)
            switch descriptor.storageType {
            case 0:
                for column in 0..<columns {
                    let bits = UInt32(littleEndian: source.loadUnaligned(fromByteOffset: 4 * column, as: UInt32.self))
                    values[column] = Float(bitPattern: bits)
                }
            case 1:
                for column in 0..<columns {
                    let bits = UInt16(littleEndian: source.loadUnaligned(fromByteOffset: 2 * column, as: UInt16.self))
                    values[column] = Float(Float16(bitPattern: bits))
                }
            case 2:
                for block in 0..<blocks {
                    let start = source.advanced(by: block * 18)
                    let bits = UInt16(littleEndian: start.loadUnaligned(as: UInt16.self))
                    let scale = Float(Float16(bitPattern: bits))
                    for offset in 0..<16 {
                        let packed = start.load(fromByteOffset: offset + 2, as: UInt8.self)
                        values[block * 32 + offset] = scale * Float(Int(packed & 15) - 8)
                        values[block * 32 + offset + 16] = scale * Float(Int(packed >> 4) - 8)
                    }
                }
            case 3:
                for block in 0..<blocks {
                    let start = source.advanced(by: block * 20)
                    let scaleBits = UInt16(littleEndian: start.loadUnaligned(as: UInt16.self))
                    let minimumBits = UInt16(littleEndian: start.loadUnaligned(fromByteOffset: 2, as: UInt16.self))
                    let scale = Float(Float16(bitPattern: scaleBits))
                    let minimum = Float(Float16(bitPattern: minimumBits))
                    for offset in 0..<16 {
                        let packed = start.load(fromByteOffset: offset + 4, as: UInt8.self)
                        values[block * 32 + offset] = scale * Float(packed & 15) + minimum
                        values[block * 32 + offset + 16] = scale * Float(packed >> 4) + minimum
                    }
                }
            case 6, 7:
                let hasMinimum = descriptor.storageType == 7
                let blockSize = hasMinimum ? 24 : 22
                let packedOffset = hasMinimum ? 8 : 6
                for block in 0..<blocks {
                    let start = source.advanced(by: block * blockSize)
                    let scaleBits = UInt16(littleEndian: start.loadUnaligned(as: UInt16.self))
                    let scale = Float(Float16(bitPattern: scaleBits))
                    let minimum: Float
                    if hasMinimum {
                        let minimumBits = UInt16(littleEndian: start.loadUnaligned(fromByteOffset: 2, as: UInt16.self))
                        minimum = Float(Float16(bitPattern: minimumBits))
                    } else {
                        minimum = 0
                    }
                    let highBits = UInt32(littleEndian: start.loadUnaligned(fromByteOffset: hasMinimum ? 4 : 2, as: UInt32.self))
                    for offset in 0..<16 {
                        let packed = start.load(fromByteOffset: packedOffset + offset, as: UInt8.self)
                        let low = Int(packed & 15) | (Int((highBits >> offset) & 1) << 4)
                        let high = Int(packed >> 4) | (Int((highBits >> (offset + 16)) & 1) << 4)
                        values[block * 32 + offset] = scale * Float(hasMinimum ? low : low - 16) + minimum
                        values[block * 32 + offset + 16] = scale * Float(hasMinimum ? high : high - 16) + minimum
                    }
                }
            case 8:
                for block in 0..<blocks {
                    let start = source.advanced(by: block * 34)
                    let bits = UInt16(littleEndian: start.loadUnaligned(as: UInt16.self))
                    let scale = Float(Float16(bitPattern: bits))
                    for offset in 0..<32 {
                        let quantized = start.load(fromByteOffset: offset + 2, as: Int8.self)
                        values[block * 32 + offset] = scale * Float(quantized)
                    }
                }
            case 30:
                for column in 0..<columns {
                    let bits = UInt32(UInt16(littleEndian: source.loadUnaligned(fromByteOffset: 2 * column, as: UInt16.self))) << 16
                    values[column] = Float(bitPattern: bits)
                }
            default:
                throw DecodeError.unsupportedStorage(descriptor.storageType)
            }
            return values
        }
    }

    /// Matrix-vector product in GGML dimension order: [input, output].
    /// Decodes one row at a time so model weights remain file-backed.
    func multiply(_ vector: [Float]) throws -> [Float] {
        guard let columns, let rows, vector.count == columns, rows > 0 else {
            throw DecodeError.invalidShape
        }
        var output = [Float](repeating: 0, count: rows)
        for index in 0..<rows {
            try Task.checkCancellation()
            let weights = try row(index)
            var value: Float = 0
            for column in 0..<columns {
                value += weights[column] * vector[column]
            }
            output[index] = value
        }
        return output
    }

    private static func layout(for type: UInt32) -> (elements: Int, bytes: Int)? {
        switch type {
        case 0: return (1, 4)
        case 1, 30: return (1, 2)
        case 2: return (32, 18)
        case 3: return (32, 20)
        case 6: return (32, 22)
        case 7: return (32, 24)
        case 8: return (32, 34)
        default: return nil
        }
    }
}
