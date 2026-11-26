import Foundation

/// Executes decoder-only transformer tensors stored in the selected GGUF file.
/// It does not call a system language model, network service or other runtime.
actor LocalInferenceEngine {
    enum InferenceError: LocalizedError {
        case noModel
        case missingMetadata(String)
        case missingTensor(String)
        case invalidDimensions
        case contextExhausted

        var errorDescription: String? {
            switch self {
            case .noModel: return "Kies eerst een lokaal GGUF-model."
            case .missingMetadata(let key): return "GGUF-metadata ontbreekt: \(key)"
            case .missingTensor(let name): return "GGUF-tensor ontbreekt: \(name)"
            case .invalidDimensions: return "De tensordimensies komen niet overeen."
            case .contextExhausted: return "De contextgrootte van het GGUF-model is bereikt."
            }
        }
    }

    private struct LayerCache {
        var keys: [[Float]] = []
        var values: [[Float]] = []
    }

    private struct Loaded {
        let file: GGUFFile
        let tokenizer: GGUFTokenizer
        let architecture: String
        let embeddingSize: Int
        let layerCount: Int
        let heads: Int
        let kvHeads: Int
        let contextLength: Int
        let rmsEpsilon: Float
        let ropeBase: Double

        func tensor(_ name: String) throws -> GGMLTensor {
            guard let descriptor = file.tensors[name] else {
                throw InferenceError.missingTensor(name)
            }
            return GGMLTensor(file: file, descriptor: descriptor)
        }

        func optionalTensor(_ name: String) -> GGMLTensor? {
            file.tensors[name].map { GGMLTensor(file: file, descriptor: $0) }
        }
    }

    private var loaded: Loaded?

    func selectModel(at url: URL) throws {
        let file = try GGUFFile(url: url)
        let tokenizer = try GGUFTokenizer(metadata: file.metadata)
        guard case .text(let architecture)? = file.metadata["general.architecture"] else {
            throw InferenceError.missingMetadata("general.architecture")
        }
        func required(_ suffix: String) throws -> Int {
            let key = "\(architecture).\(suffix)"
            guard let number = Self.integer(file.metadata[key]), number > 0 else {
                throw InferenceError.missingMetadata(key)
            }
            return number
        }
        let embeddingSize = try required("embedding_length")
        let layerCount = try required("block_count")
        let heads = try required("attention.head_count")
        let kvHeads = Self.integer(file.metadata["\(architecture).attention.head_count_kv"]) ?? heads
        guard kvHeads > 0, heads.isMultiple(of: kvHeads),
              embeddingSize.isMultiple(of: heads) else {
            throw InferenceError.invalidDimensions
        }
        let contextLength = Self.integer(file.metadata["\(architecture).context_length"]) ?? Int.max
        let epsilon = Self.number(file.metadata["\(architecture).attention.layer_norm_rms_epsilon"]) ?? 1e-5
        let ropeBase = Self.number(file.metadata["\(architecture).rope.freq_base"]) ?? 10_000
        loaded = Loaded(file: file, tokenizer: tokenizer, architecture: architecture,
                        embeddingSize: embeddingSize, layerCount: layerCount,
                        heads: heads, kvHeads: kvHeads, contextLength: contextLength,
                        rmsEpsilon: Float(epsilon), ropeBase: ropeBase)
    }

    func stream(messages: [ChatMessage]) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await generate(messages: messages, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func generate(messages: [ChatMessage],
                          continuation: AsyncThrowingStream<String, Error>.Continuation) async throws {
        guard let model = loaded else { throw InferenceError.noModel }
        let prompt = try promptTokens(messages, tokenizer: model.tokenizer)
        guard !prompt.isEmpty else { return }
        var cache = [LayerCache](repeating: LayerCache(), count: model.layerCount)
        var logits: [Float] = []
        for (position, token) in prompt.enumerated() {
            try Task.checkCancellation()
            guard position < model.contextLength else { throw InferenceError.contextExhausted }
            logits = try forward(token: token, position: position, cache: &cache, model: model)
        }
        var outputTokens: [Int] = []
        var position = prompt.count
        while true {
            try Task.checkCancellation()
            guard position < model.contextLength else { throw InferenceError.contextExhausted }
            guard let next = logits.indices.max(by: { logits[$0] < logits[$1] }) else {
                throw InferenceError.invalidDimensions
            }
            if model.tokenizer.endIDs.contains(next) { return }
            outputTokens.append(next)
            continuation.yield(model.tokenizer.decode(outputTokens))
            logits = try forward(token: next, position: position, cache: &cache, model: model)
            position += 1
            await Task.yield()
        }
    }

    private func promptTokens(_ messages: [ChatMessage], tokenizer: GGUFTokenizer) throws -> [Int] {
        var result: [Int] = []
        if let start = tokenizer.id(for: "<|start_header_id|>"),
           let end = tokenizer.id(for: "<|end_header_id|>"),
           let messageEnd = tokenizer.id(for: "<|eot_id|>") {
            if let bos = tokenizer.bosID { result.append(bos) }
            for message in messages {
                result += [start]
                result += try tokenizer.encode(message.role.rawValue)
                result += [end]
                result += try tokenizer.encode("\n\n" + message.content)
                result += [messageEnd]
            }
            result += [start]
            result += try tokenizer.encode("assistant")
            result += [end]
            result += try tokenizer.encode("\n\n")
        } else if let start = tokenizer.id(for: "<|im_start|>"),
                  let end = tokenizer.id(for: "<|im_end|>") {
            for message in messages {
                result.append(start)
                result += try tokenizer.encode(message.role.rawValue + "\n" + message.content)
                result.append(end)
                result += try tokenizer.encode("\n")
            }
            result.append(start)
            result += try tokenizer.encode("assistant\n")
        } else {
            if let bos = tokenizer.bosID { result.append(bos) }
            for message in messages {
                let role = message.role == .user ? "User" : "Assistant"
                result += try tokenizer.encode("\n\(role): \(message.content)")
            }
            result += try tokenizer.encode("\nAssistant:")
        }
        return result
    }

    private func forward(token: Int, position: Int,
                         cache: inout [LayerCache], model: Loaded) throws -> [Float] {
        var hidden = try model.tensor("token_embd.weight").row(token)
        guard hidden.count == model.embeddingSize else { throw InferenceError.invalidDimensions }
        let headSize = model.embeddingSize / model.heads
        let kvSize = headSize * model.kvHeads

        for layer in 0..<model.layerCount {
            try Task.checkCancellation()
            let prefix = "blk.\(layer)."
            let attentionNorm = try model.tensor(prefix + "attn_norm.weight").row(0)
            let normalized = try Self.rmsNorm(hidden, weight: attentionNorm, epsilon: model.rmsEpsilon)
            var query = try model.tensor(prefix + "attn_q.weight").multiply(normalized)
            var key = try model.tensor(prefix + "attn_k.weight").multiply(normalized)
            let value = try model.tensor(prefix + "attn_v.weight").multiply(normalized)
            guard query.count == model.embeddingSize, key.count == kvSize,
                  value.count == kvSize else { throw InferenceError.invalidDimensions }
            if let bias = model.optionalTensor(prefix + "attn_q.bias") {
                query = try Self.add(query, bias.row(0))
            }
            if let bias = model.optionalTensor(prefix + "attn_k.bias") {
                key = try Self.add(key, bias.row(0))
            }
            Self.applyRoPE(&query, heads: model.heads, position: position, base: model.ropeBase)
            Self.applyRoPE(&key, heads: model.kvHeads, position: position, base: model.ropeBase)
            cache[layer].keys.append(key)
            cache[layer].values.append(value)

            var attended = [Float](repeating: 0, count: model.embeddingSize)
            for head in 0..<model.heads {
                let kvHead = head / (model.heads / model.kvHeads)
                var scores: [Float] = []
                scores.reserveCapacity(cache[layer].keys.count)
                for pastKey in cache[layer].keys {
                    var dot: Float = 0
                    for index in 0..<headSize {
                        dot += query[head * headSize + index] * pastKey[kvHead * headSize + index]
                    }
                    scores.append(dot / sqrt(Float(headSize)))
                }
                Self.softmax(&scores)
                for (past, probability) in scores.enumerated() {
                    for index in 0..<headSize {
                        attended[head * headSize + index] += probability
                            * cache[layer].values[past][kvHead * headSize + index]
                    }
                }
            }
            let projected = try model.tensor(prefix + "attn_output.weight").multiply(attended)
            hidden = try Self.add(hidden, projected)
            let feedNorm = try model.tensor(prefix + "ffn_norm.weight").row(0)
            let feedInput = try Self.rmsNorm(hidden, weight: feedNorm, epsilon: model.rmsEpsilon)
            let gate = try model.tensor(prefix + "ffn_gate.weight").multiply(feedInput)
            let up = try model.tensor(prefix + "ffn_up.weight").multiply(feedInput)
            guard gate.count == up.count else { throw InferenceError.invalidDimensions }
            let activated = zip(gate, up).map { pair -> Float in
                let (g, u) = pair
                return (g / (1 + Float(exp(Double(-g))))) * u
            }
            let down = try model.tensor(prefix + "ffn_down.weight").multiply(activated)
            hidden = try Self.add(hidden, down)
        }
        let outputNorm = try model.tensor("output_norm.weight").row(0)
        let outputInput = try Self.rmsNorm(hidden, weight: outputNorm, epsilon: model.rmsEpsilon)
        return try (model.optionalTensor("output.weight") ?? model.tensor("token_embd.weight"))
            .multiply(outputInput)
    }

    private static func rmsNorm(_ values: [Float], weight: [Float], epsilon: Float) throws -> [Float] {
        guard values.count == weight.count, !values.isEmpty else { throw InferenceError.invalidDimensions }
        let meanSquare = values.reduce(Float(0)) { $0 + $1 * $1 } / Float(values.count)
        let inverse = 1 / sqrt(meanSquare + epsilon)
        return zip(values, weight).map { $0.0 * inverse * $0.1 }
    }

    private static func add(_ left: [Float], _ right: [Float]) throws -> [Float] {
        guard left.count == right.count else { throw InferenceError.invalidDimensions }
        return zip(left, right).map { $0.0 + $0.1 }
    }

    private static func applyRoPE(_ values: inout [Float], heads: Int,
                                  position: Int, base: Double) {
        let width = values.count / heads
        guard width.isMultiple(of: 2) else { return }
        for head in 0..<heads {
            for index in 0..<(width / 2) {
                let theta = Double(position) / pow(base, Double(2 * index) / Double(width))
                let sine = Float(sin(theta))
                let cosine = Float(cos(theta))
                let first = head * width + index
                let second = first + width / 2
                let a = values[first]
                let b = values[second]
                values[first] = a * cosine - b * sine
                values[second] = a * sine + b * cosine
            }
        }
    }

    private static func softmax(_ values: inout [Float]) {
        guard let maximum = values.max() else { return }
        var sum: Float = 0
        for index in values.indices {
            values[index] = exp(values[index] - maximum)
            sum += values[index]
        }
        for index in values.indices { values[index] /= sum }
    }

    private static func integer(_ value: GGUFFile.Value?) -> Int? {
        switch value {
        case .unsigned(let number): return Int(exactly: number)
        case .signed(let number): return Int(exactly: number)
        default: return nil
        }
    }

    private static func number(_ value: GGUFFile.Value?) -> Double? {
        switch value {
        case .floating(let number): return number
        case .unsigned(let number): return Double(number)
        case .signed(let number): return Double(number)
        default: return nil
        }
    }
}
