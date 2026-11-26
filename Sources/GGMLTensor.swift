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
            case 9:
                for block in 0..<blocks {
                    let start = source.advanced(by: block * 36)
                    let bits = UInt16(littleEndian: start.loadUnaligned(as: UInt16.self))
                    let scale = Float(Float16(bitPattern: bits))
                    for offset in 0..<32 {
                        values[block * 32 + offset] = scale
                            * Float(start.load(fromByteOffset: offset + 4, as: Int8.self))
                    }
                }
            case 10:
                for block in 0..<blocks {
                    let start = source.advanced(by: block * 84)
                    let dBits = UInt16(littleEndian: start.loadUnaligned(fromByteOffset: 80, as: UInt16.self))
                    let minBits = UInt16(littleEndian: start.loadUnaligned(fromByteOffset: 82, as: UInt16.self))
                    let d = Float(Float16(bitPattern: dBits))
                    let dmin = Float(Float16(bitPattern: minBits))
                    for half in 0..<2 {
                        for group in 0..<4 {
                            let shift = group * 2
                            for subblock in 0..<2 {
                                let scaleByte = start.load(fromByteOffset: half * 8 + group * 2 + subblock,
                                                           as: UInt8.self)
                                let scale = d * Float(scaleByte & 15)
                                let minimum = dmin * Float(scaleByte >> 4)
                                for offset in 0..<16 {
                                    let packed = start.load(fromByteOffset: 16 + half * 32
                                                            + subblock * 16 + offset, as: UInt8.self)
                                    let quantized = Float((packed >> shift) & 3)
                                    let index = block * 256 + half * 128 + group * 32
                                        + subblock * 16 + offset
                                    values[index] = scale * quantized - minimum
                                }
                            }
                        }
                    }
                }
            case 11:
                for block in 0..<blocks {
                    let start = source.advanced(by: block * 110)
                    let bits = UInt16(littleEndian: start.loadUnaligned(fromByteOffset: 108, as: UInt16.self))
                    let d = Float(Float16(bitPattern: bits))
                    for half in 0..<2 {
                        for group in 0..<4 {
                            let shift = group * 2
                            let highMask = UInt8(1 << (half * 4 + group))
                            for subblock in 0..<2 {
                                let scaleIndex = half * 8 + group * 2 + subblock
                                let scale = d * Float(Self.q3Scale(start, scaleIndex) - 32)
                                for offset in 0..<16 {
                                    let quantizedByte = start.load(fromByteOffset: 32 + half * 32
                                                                   + subblock * 16 + offset, as: UInt8.self)
                                    let highByte = start.load(fromByteOffset: subblock * 16 + offset,
                                                              as: UInt8.self)
                                    let quantized = Int((quantizedByte >> shift) & 3)
                                        - ((highByte & highMask) == 0 ? 4 : 0)
                                    let index = block * 256 + half * 128 + group * 32
                                        + subblock * 16 + offset
                                    values[index] = scale * Float(quantized)
                                }
                            }
                        }
                    }
                }
            case 12, 13:
                let fiveBit = descriptor.storageType == 13
                let blockSize = fiveBit ? 176 : 144
                let highOffset = 16
                let lowOffset = fiveBit ? 48 : 16
                for block in 0..<blocks {
                    let start = source.advanced(by: block * blockSize)
                    let dBits = UInt16(littleEndian: start.loadUnaligned(as: UInt16.self))
                    let minBits = UInt16(littleEndian: start.loadUnaligned(fromByteOffset: 2, as: UInt16.self))
                    let d = Float(Float16(bitPattern: dBits))
                    let dmin = Float(Float16(bitPattern: minBits))
                    for group in 0..<4 {
                        let (scale0, min0) = Self.scaleMin(start, group * 2)
                        let (scale1, min1) = Self.scaleMin(start, group * 2 + 1)
                        let lowScale = d * Float(scale0)
                        let highScale = d * Float(scale1)
                        let lowMinimum = dmin * Float(min0)
                        let highMinimum = dmin * Float(min1)
                        for offset in 0..<32 {
                            let byte = start.load(fromByteOffset: lowOffset + group * 32 + offset,
                                                  as: UInt8.self)
                            let highByte = fiveBit
                                ? start.load(fromByteOffset: highOffset + offset, as: UInt8.self)
                                : 0
                            let lowHighBit = fiveBit && (highByte & (1 << (group * 2))) != 0 ? 16 : 0
                            let upperHighBit = fiveBit && (highByte & (2 << (group * 2))) != 0 ? 16 : 0
                            values[block * 256 + group * 64 + offset] =
                                lowScale * Float(Int(byte & 15) + lowHighBit) - lowMinimum
                            values[block * 256 + group * 64 + offset + 32] =
                                highScale * Float(Int(byte >> 4) + upperHighBit) - highMinimum
                        }
                    }
                }
            case 14:
                for block in 0..<blocks {
                    let start = source.advanced(by: block * 210)
                    let bits = UInt16(littleEndian: start.loadUnaligned(fromByteOffset: 208, as: UInt16.self))
                    let d = Float(Float16(bitPattern: bits))
                    for half in 0..<2 {
                        for offset in 0..<32 {
                            let low0 = start.load(fromByteOffset: half * 64 + offset, as: UInt8.self)
                            let low1 = start.load(fromByteOffset: half * 64 + offset + 32, as: UInt8.self)
                            let high = start.load(fromByteOffset: 128 + half * 32 + offset, as: UInt8.self)
                            let scaleIndex = 192 + half * 8 + offset / 16
                            let s0 = Float(start.load(fromByteOffset: scaleIndex, as: Int8.self))
                            let s1 = Float(start.load(fromByteOffset: scaleIndex + 2, as: Int8.self))
                            let s2 = Float(start.load(fromByteOffset: scaleIndex + 4, as: Int8.self))
                            let s3 = Float(start.load(fromByteOffset: scaleIndex + 6, as: Int8.self))
                            let q0 = Int(low0 & 15) | (Int(high & 3) << 4)
                            let q1 = Int(low1 & 15) | (Int((high >> 2) & 3) << 4)
                            let q2 = Int(low0 >> 4) | (Int((high >> 4) & 3) << 4)
                            let q3 = Int(low1 >> 4) | (Int((high >> 6) & 3) << 4)
                            let index = block * 256 + half * 128 + offset
                            values[index] = d * s0 * Float(q0 - 32)
                            values[index + 32] = d * s1 * Float(q1 - 32)
                            values[index + 64] = d * s2 * Float(q2 - 32)
                            values[index + 96] = d * s3 * Float(q3 - 32)
                        }
                    }
                }
            case 15:
                for block in 0..<blocks {
                    let start = source.advanced(by: block * 292)
                    let bits = UInt32(littleEndian: start.loadUnaligned(as: UInt32.self))
                    let scale = Float(bitPattern: bits)
                    for offset in 0..<256 {
                        values[block * 256 + offset] = scale
                            * Float(start.load(fromByteOffset: offset + 4, as: Int8.self))
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
        case 9: return (32, 36)
        case 10: return (256, 84)
        case 11: return (256, 110)
        case 12: return (256, 144)
        case 13: return (256, 176)
        case 14: return (256, 210)
        case 15: return (256, 292)
        default: return nil
        }
    }

    private static func scaleMin(_ block: UnsafeRawPointer, _ index: Int) -> (Int, Int) {
        func scaleByte(_ offset: Int) -> UInt8 {
            block.load(fromByteOffset: 4 + offset, as: UInt8.self)
        }
        if index < 4 {
            return (Int(scaleByte(index) & 63), Int(scaleByte(index + 4) & 63))
        }
        let scale = Int(scaleByte(index + 4) & 15)
            | (Int(scaleByte(index - 4) >> 6) << 4)
        let minimum = Int(scaleByte(index + 4) >> 4)
            | (Int(scaleByte(index) >> 6) << 4)
        return (scale, minimum)
    }

    private static func q3Scale(_ block: UnsafeRawPointer, _ index: Int) -> Int {
        func byte(_ offset: Int) -> UInt8 {
            block.load(fromByteOffset: 96 + offset, as: UInt8.self)
        }
        switch index {
        case 0..<4:
            return Int(byte(index) & 15) | (Int(byte(8 + index) & 3) << 4)
        case 4..<8:
            return Int(byte(index) & 15) | (Int((byte(8 + index - 4) >> 2) & 3) << 4)
        case 8..<12:
            return Int(byte(index - 8) >> 4) | (Int((byte(8 + index - 8) >> 4) & 3) << 4)
        default:
            return Int(byte(index - 8) >> 4) | (Int((byte(8 + index - 12) >> 6) & 3) << 4)
        }
    }
}
