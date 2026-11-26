import Foundation

/// Uses tokenizer tables embedded in the selected GGUF file.
struct GGUFTokenizer {
    enum TokenizerError: LocalizedError {
        case missingVocabulary
        case cannotEncode(String)

        var errorDescription: String? {
            switch self {
            case .missingVocabulary: return "Het GGUF-bestand bevat geen tokenwoordenboek."
            case .cannotEncode(let text): return "Tekst kan niet worden omgezet naar modeltokens: \(text)"
            }
        }
    }

    let tokens: [String]
    let tokenIDs: [String: Int]
    let scores: [Double]
    let mergeRanks: [String: Int]
    let modelName: String
    let bosID: Int?
    let endIDs: Set<Int>

    private static let byteMap: [UInt8: String] = {
        var ordinary = Array(33...126) + Array(161...172) + Array(174...255)
        var map: [UInt8: String] = [:]
        for byte in ordinary { map[UInt8(byte)] = String(UnicodeScalar(byte)!) }
        var next = 256
        for byte in 0...255 where !ordinary.contains(byte) {
            map[UInt8(byte)] = String(UnicodeScalar(next)!)
            next += 1
        }
        return map
    }()

    private static let reverseByteMap: [UnicodeScalar: UInt8] = {
        var map: [UnicodeScalar: UInt8] = [:]
        for (byte, character) in byteMap { map[character.unicodeScalars.first!] = byte }
        return map
    }()

    init(metadata: [String: GGUFFile.Value]) throws {
        guard case .array(let values)? = metadata["tokenizer.ggml.tokens"] else {
            throw TokenizerError.missingVocabulary
        }
        let words = values.compactMap { value -> String? in
            if case .text(let word) = value { return word }
            return nil
        }
        guard words.count == values.count, !words.isEmpty else {
            throw TokenizerError.missingVocabulary
        }
        tokens = words
        tokenIDs = Dictionary(words.enumerated().map { ($0.element, $0.offset) },
                              uniquingKeysWith: { first, _ in first })
        if case .array(let rawScores)? = metadata["tokenizer.ggml.scores"] {
            scores = rawScores.map { value in
                if case .floating(let number) = value { return number }
                return -Double.infinity
            }
        } else {
            scores = []
        }
        if case .array(let rawMerges)? = metadata["tokenizer.ggml.merges"] {
            var ranks: [String: Int] = [:]
            for (index, value) in rawMerges.enumerated() {
                if case .text(let pair) = value { ranks[pair] = index }
            }
            mergeRanks = ranks
        } else {
            mergeRanks = [:]
        }
        if case .text(let name)? = metadata["tokenizer.ggml.model"] {
            modelName = name
        } else {
            modelName = ""
        }
        bosID = Self.integer(metadata["tokenizer.ggml.bos_token_id"])
        endIDs = Set(["tokenizer.ggml.eos_token_id",
                      "tokenizer.ggml.eot_token_id",
                      "tokenizer.ggml.eom_token_id"]
            .compactMap { Self.integer(metadata[$0]) }
            + ["<|eot_id|>", "<|im_end|>"]
                .compactMap { tokenIDs[$0] })
    }

    func id(for token: String) -> Int? { tokenIDs[token] }

    func encode(_ text: String) throws -> [Int] {
        if !mergeRanks.isEmpty { return try encodeByteBPE(text) }
        return try encodeSentencePiece(text)
    }

    func decode(_ ids: [Int]) -> String {
        if !mergeRanks.isEmpty {
            var bytes: [UInt8] = []
            for id in ids where tokens.indices.contains(id) {
                for scalar in tokens[id].unicodeScalars {
                    if let byte = Self.reverseByteMap[scalar] { bytes.append(byte) }
                }
            }
            return String(decoding: bytes, as: UTF8.self)
        }
        var bytes: [UInt8] = []
        for id in ids where tokens.indices.contains(id) {
            let token = tokens[id]
            if token.hasPrefix("<0x"), token.hasSuffix(">"),
               let byte = UInt8(token.dropFirst(3).dropLast(), radix: 16) {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: token.replacingOccurrences(of: "▁", with: " ").utf8)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    private func encodeByteBPE(_ text: String) throws -> [Int] {
        let pattern = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let regularExpression = try NSRegularExpression(pattern: pattern)
        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        var result: [Int] = []
        for match in regularExpression.matches(in: text, range: fullRange) {
            guard let range = Range(match.range, in: text) else { continue }
            var parts = Array(text[range].utf8).map { Self.byteMap[$0]! }
            while parts.count > 1 {
                var bestIndex: Int?
                var bestRank = Int.max
                for index in 0..<(parts.count - 1) {
                    let pair = parts[index] + " " + parts[index + 1]
                    if let rank = mergeRanks[pair], rank < bestRank {
                        bestRank = rank
                        bestIndex = index
                    }
                }
                guard let index = bestIndex else { break }
                parts[index] += parts[index + 1]
                parts.remove(at: index + 1)
            }
            for part in parts {
                guard let id = tokenIDs[part] else { throw TokenizerError.cannotEncode(part) }
                result.append(id)
            }
        }
        return result
    }

    private func encodeSentencePiece(_ text: String) throws -> [Int] {
        let normalized = "▁" + text.replacingOccurrences(of: " ", with: "▁")
        var parts = normalized.unicodeScalars.map(String.init)
        while parts.count > 1 {
            var bestIndex: Int?
            var bestScore = -Double.infinity
            for index in 0..<(parts.count - 1) {
                let candidate = parts[index] + parts[index + 1]
                if let id = tokenIDs[candidate] {
                    let score = scores.indices.contains(id) ? scores[id] : 0
                    if score > bestScore {
                        bestScore = score
                        bestIndex = index
                    }
                }
            }
            guard let index = bestIndex else { break }
            parts[index] += parts[index + 1]
            parts.remove(at: index + 1)
        }
        var result: [Int] = []
        for part in parts {
            if let id = tokenIDs[part] {
                result.append(id)
            } else {
                for byte in part.utf8 {
                    let spelling = String(format: "<0x%02X>", byte)
                    guard let id = tokenIDs[spelling] else {
                        throw TokenizerError.cannotEncode(part)
                    }
                    result.append(id)
                }
            }
        }
        return result
    }

    private static func integer(_ value: GGUFFile.Value?) -> Int? {
        switch value {
        case .unsigned(let number): return Int(exactly: number)
        case .signed(let number): return Int(exactly: number)
        default: return nil
        }
    }
}
