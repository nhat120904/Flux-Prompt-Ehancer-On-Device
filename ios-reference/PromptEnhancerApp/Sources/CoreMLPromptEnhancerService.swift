import Foundation

actor CoreMLPromptEnhancerService {
    private let tokenizer: Tokenizer
    private let runner: CoreMLT5Runner

    init(tokenizer: Tokenizer, runner: CoreMLT5Runner) {
        self.tokenizer = tokenizer
        self.runner = runner
    }

    func warmup() async throws {
        try await runner.warmup()
    }

    func enhance(input: String, options: EnhanceOptions = EnhanceOptions()) async throws -> EnhanceResult {
        let start = Date()
        let inputIds = tokenizer.encode(input, maxSourceTokens: 128)
        let attentionMask = inputIds.map { $0 == tokenizer.padTokenId ? 0 : 1 }

        var bestText = ""
        var bestTokens = 0
        var finishReason = "max_tokens"

        for attempt in 0..<options.retries {
            if Int(Date().timeIntervalSince(start) * 1000) > options.timeoutMs {
                finishReason = "timeout"
                break
            }

            try Task.checkCancellation()
            let tokens = try await generateOnce(
                inputIds: inputIds,
                attentionMask: attentionMask,
                options: options,
                attempt: attempt,
                start: start
            )
            let text = tokenizer.decode(tokens).trimmingCharacters(in: .whitespacesAndNewlines)

            if text.count > bestText.count {
                bestText = text
                bestTokens = tokens.count
            }

            if isCompleteSentence(text) {
                finishReason = "eos"
                return EnhanceResult(
                    text: text,
                    latencyMs: Int(Date().timeIntervalSince(start) * 1000),
                    tokensGenerated: tokens.count,
                    finishReason: finishReason
                )
            }
        }

        if !bestText.isEmpty, !isCompleteSentence(bestText) {
            bestText += "."
        }

        return EnhanceResult(
            text: bestText,
            latencyMs: Int(Date().timeIntervalSince(start) * 1000),
            tokensGenerated: bestTokens,
            finishReason: finishReason
        )
    }

    private func generateOnce(
        inputIds: [Int],
        attentionMask: [Int],
        options: EnhanceOptions,
        attempt: Int,
        start: Date
    ) async throws -> [Int] {
        var output: [Int] = []
        var decoderIds: [Int] = [tokenizer.decoderStartTokenId]
        var seenTokens: [Int] = []

        let seed = options.seed ?? UInt64(Date().timeIntervalSince1970) &+ UInt64(attempt)
        var rng = SeededGenerator(seed: seed)

        let maxModelNewTokens = 63
        let cappedMaxNewTokens = min(options.maxNewTokens, maxModelNewTokens)
        let cappedMinNewTokens = min(options.minNewTokens, cappedMaxNewTokens)

        for step in 0..<cappedMaxNewTokens {
            if Int(Date().timeIntervalSince(start) * 1000) > options.timeoutMs {
                break
            }
            try Task.checkCancellation()

            let logits = try await runner.decodeNextTokenLogits(
                inputIds: inputIds.map(Int32.init),
                attentionMask: attentionMask.map(Int32.init),
                decoderInputIds: decoderIds.map(Int32.init)
            )

            let adjusted = applyLogitRules(
                logits: logits,
                generatedTokens: seenTokens,
                noRepeatNgramSize: options.noRepeatNgramSize,
                repetitionPenalty: options.repetitionPenalty
            )

            let next = sampleTopP(
                logits: adjusted,
                topP: options.topP,
                temperature: options.temperature,
                generator: &rng
            )

            decoderIds.append(next)
            if next == tokenizer.eosTokenId && step + 1 >= cappedMinNewTokens {
                break
            }

            output.append(next)
            seenTokens.append(next)
        }

        return output
    }

    private func isCompleteSentence(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.hasSuffix(".") || trimmed.hasSuffix("!") || trimmed.hasSuffix("?")
    }

    private func applyLogitRules(
        logits: [Float],
        generatedTokens: [Int],
        noRepeatNgramSize: Int,
        repetitionPenalty: Float
    ) -> [Float] {
        var adjusted = logits

        for token in generatedTokens where token >= 0 && token < adjusted.count {
            if adjusted[token] < 0 {
                adjusted[token] *= repetitionPenalty
            } else {
                adjusted[token] /= repetitionPenalty
            }
        }

        if noRepeatNgramSize > 1 && generatedTokens.count >= noRepeatNgramSize - 1 {
            let prefix = Array(generatedTokens.suffix(noRepeatNgramSize - 1))
            var blocked = Set<Int>()
            if generatedTokens.count >= noRepeatNgramSize {
                for i in 0...(generatedTokens.count - noRepeatNgramSize) {
                    let ngram = Array(generatedTokens[i..<(i + noRepeatNgramSize)])
                    if Array(ngram.dropLast()) == prefix {
                        blocked.insert(ngram.last!)
                    }
                }
            }
            for token in blocked where token >= 0 && token < adjusted.count {
                adjusted[token] = -.greatestFiniteMagnitude
            }
        }

        return adjusted
    }

    private func sampleTopP(
        logits: [Float],
        topP: Float,
        temperature: Float,
        generator: inout SeededGenerator
    ) -> Int {
        let safeTemperature = max(temperature, 1e-5)
        let safeTopP = min(max(topP, 0.01), 1.0)

        let finite = logits.enumerated().compactMap { idx, value -> (Int, Double)? in
            guard value.isFinite else { return nil }
            return (idx, Double(value / safeTemperature))
        }

        guard !finite.isEmpty else {
            return logits.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        }

        let maxLogit = finite.map(\.1).max() ?? 0.0
        let weighted = finite.map { (idx, scaled) -> (Int, Double) in
            (idx, Foundation.exp(scaled - maxLogit))
        }.sorted { $0.1 > $1.1 }

        let allMass = weighted.reduce(0.0) { $0 + $1.1 }
        guard allMass > 0 else {
            return weighted.first?.0 ?? 0
        }

        var candidates: [(Int, Double)] = []
        var cumulativeMass = 0.0
        for (idx, weight) in weighted {
            candidates.append((idx, weight))
            cumulativeMass += weight
            if (cumulativeMass / allMass) >= Double(safeTopP), candidates.count > 1 {
                break
            }
        }

        let norm = candidates.reduce(0.0) { $0 + $1.1 }
        guard norm > 0 else {
            return candidates.first?.0 ?? 0
        }

        let target = Double.random(in: 0..<norm, using: &generator)
        var sum = 0.0
        for (idx, p) in candidates {
            sum += p
            if sum >= target {
                return idx
            }
        }

        return candidates.last?.0 ?? 0
    }
}

struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed == 0 ? 0x123456789abcdef : seed
    }

    mutating func next() -> UInt64 {
        state = state &* 6364136223846793005 &+ 1
        return state
    }
}
