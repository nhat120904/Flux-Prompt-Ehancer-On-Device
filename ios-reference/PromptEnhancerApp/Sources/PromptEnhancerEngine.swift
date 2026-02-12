import Foundation

enum PromptEnhancerEngineError: LocalizedError {
    case tokenizerNotFound

    var errorDescription: String? {
        switch self {
        case .tokenizerNotFound:
            return "Tokenizer resource tokenizer_vocab.json not found in app bundle"
        }
    }
}

actor PromptEnhancerEngine {
    private var service: CoreMLPromptEnhancerService?

    func warmup() async throws {
        let svc = try buildServiceIfNeeded()
        try await svc.warmup()
    }

    func enhance(prompt: String, options: EnhanceOptions = EnhanceOptions()) async throws -> EnhanceResult {
        let svc = try buildServiceIfNeeded()
        return try await svc.enhance(input: prompt, options: options)
    }

    private func buildServiceIfNeeded() throws -> CoreMLPromptEnhancerService {
        if let service {
            return service
        }

        let tokenizerURL = try resolveTokenizerURL()
        let tokenizer = try SentencePieceTokenizer(vocabURL: tokenizerURL)
        let runner = LocalCoreMLT5Runner()
        let newService = CoreMLPromptEnhancerService(tokenizer: tokenizer, runner: runner)
        service = newService
        return newService
    }

    private func resolveTokenizerURL() throws -> URL {
        if let direct = Bundle.main.url(forResource: "tokenizer_vocab", withExtension: "json") {
            return direct
        }
        if let subdir = Bundle.main.url(
            forResource: "tokenizer_vocab",
            withExtension: "json",
            subdirectory: "AssetsData/Tokenizer"
        ) {
            return subdir
        }

        let candidates: [URL?] = [
            Bundle.main.resourceURL?.appendingPathComponent("AssetsData/Tokenizer/tokenizer_vocab.json"),
            Bundle.main.resourceURL?.appendingPathComponent("Tokenizer/tokenizer_vocab.json"),
            Bundle.main.resourceURL?.appendingPathComponent("tokenizer_vocab.json"),
        ]

        for candidate in candidates {
            if let url = candidate, FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        throw PromptEnhancerEngineError.tokenizerNotFound
    }
}
