import CoreML
import Foundation

enum LocalCoreMLRunnerError: LocalizedError {
    case modelNotFound(String)
    case invalidInput(String)
    case predictionOutputMissing(String)

    var errorDescription: String? {
        switch self {
        case let .modelNotFound(name):
            return "Model resource not found: \(name)"
        case let .invalidInput(reason):
            return "Invalid model input: \(reason)"
        case let .predictionOutputMissing(name):
            return "Model output missing: \(name)"
        }
    }
}

final class LocalCoreMLT5Runner: CoreMLT5Runner, @unchecked Sendable {
    private static let maxInputTokens = 128
    private static let maxDecoderTokens = 64

    private let queue = DispatchQueue(label: "com.varmeta.prompter.runner")
    private var decoderModel: MLModel?
    private var decoderOutputName: String?

    func warmup() async throws {
        let dummyInput = Array(repeating: Int32(0), count: Self.maxInputTokens)
        let dummyMask = Array(repeating: Int32(1), count: Self.maxInputTokens)
        let dummyDecoder = [Int32(0)]
        _ = try await decodeNextTokenLogits(
            inputIds: dummyInput,
            attentionMask: dummyMask,
            decoderInputIds: dummyDecoder
        )
    }

    func decodeNextTokenLogits(
        inputIds: [Int32],
        attentionMask: [Int32],
        decoderInputIds: [Int32]
    ) async throws -> [Float] {
        if inputIds.isEmpty || inputIds.count > Self.maxInputTokens {
            throw LocalCoreMLRunnerError.invalidInput("input_ids count must be 1...\(Self.maxInputTokens)")
        }
        if attentionMask.count != inputIds.count {
            throw LocalCoreMLRunnerError.invalidInput("attention_mask size must match input_ids")
        }
        if decoderInputIds.isEmpty || decoderInputIds.count > Self.maxDecoderTokens {
            throw LocalCoreMLRunnerError.invalidInput("decoder_input_ids count must be 1...\(Self.maxDecoderTokens)")
        }

        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    try self.ensureModelLoaded()

                    guard let model = self.decoderModel,
                          let outputName = self.decoderOutputName
                    else {
                        throw LocalCoreMLRunnerError.predictionOutputMissing("decoder model")
                    }

                    let inputArray = try Self.makeInt32Array(
                        values: inputIds,
                        fixedLength: Self.maxInputTokens,
                        padValue: 0
                    )
                    let maskArray = try Self.makeInt32Array(
                        values: attentionMask,
                        fixedLength: Self.maxInputTokens,
                        padValue: 0
                    )
                    let decoderArray = try Self.makeInt32Array(
                        values: decoderInputIds,
                        fixedLength: Self.maxDecoderTokens,
                        padValue: 0
                    )

                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        "input_ids": MLFeatureValue(multiArray: inputArray),
                        "attention_mask": MLFeatureValue(multiArray: maskArray),
                        "decoder_input_ids": MLFeatureValue(multiArray: decoderArray),
                    ])

                    let output = try model.prediction(from: provider)
                    guard let logitsValue = output.featureValue(for: outputName)?.multiArrayValue else {
                        throw LocalCoreMLRunnerError.predictionOutputMissing(outputName)
                    }

                    let logits = try Self.extractLogits(
                        from: logitsValue,
                        decoderTokenCount: decoderInputIds.count
                    )
                    continuation.resume(returning: logits)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func ensureModelLoaded() throws {
        if decoderModel != nil {
            return
        }

        let decoderURL = try locateModelPackage(named: "decoder_init")
        let compiledURL = try MLModel.compileModel(at: decoderURL)

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all

        let model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        guard let firstOutput = model.modelDescription.outputDescriptionsByName.keys.first else {
            throw LocalCoreMLRunnerError.predictionOutputMissing("<unknown>")
        }

        decoderModel = model
        decoderOutputName = firstOutput
    }

    private func locateModelPackage(named name: String) throws -> URL {
        if let direct = Bundle.main.url(forResource: name, withExtension: "mlpackage") {
            return direct
        }

        if let subdir = Bundle.main.url(
            forResource: name,
            withExtension: "mlpackage",
            subdirectory: "AssetsData/Model"
        ) {
            return subdir
        }

        let candidates: [URL?] = [
            Bundle.main.resourceURL?.appendingPathComponent("AssetsData/Model/\(name).mlpackage"),
            Bundle.main.resourceURL?.appendingPathComponent("Model/\(name).mlpackage"),
            Bundle.main.resourceURL?.appendingPathComponent("\(name).mlpackage"),
        ]

        for candidate in candidates {
            if let url = candidate, FileManager.default.fileExists(atPath: url.path) {
                return url
            }
        }

        throw LocalCoreMLRunnerError.modelNotFound("\(name).mlpackage")
    }

    private static func makeInt32Array(
        values: [Int32],
        fixedLength: Int,
        padValue: Int32
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: [1, NSNumber(value: fixedLength)], dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: fixedLength)
        for i in 0..<fixedLength {
            ptr[i] = i < values.count ? values[i] : padValue
        }
        return array
    }

    private static func extractLogits(
        from logits: MLMultiArray,
        decoderTokenCount: Int
    ) throws -> [Float] {
        let shape = logits.shape.map { $0.intValue }
        guard shape.count == 3 else {
            throw LocalCoreMLRunnerError.invalidInput("unexpected logits rank: \(shape)")
        }

        let steps = shape[1]
        let vocabSize = shape[2]
        let strides = logits.strides.map { $0.intValue }
        let stepIndex = min(max(decoderTokenCount - 1, 0), max(steps - 1, 0))

        var out = Array(repeating: Float(0), count: vocabSize)

        switch logits.dataType {
        case .float32:
            let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
            for token in 0..<vocabSize {
                let offset = stepIndex * strides[1] + token * strides[2]
                out[token] = ptr[offset]
            }
        case .float16:
            let ptr = logits.dataPointer.bindMemory(to: UInt16.self, capacity: logits.count)
            for token in 0..<vocabSize {
                let offset = stepIndex * strides[1] + token * strides[2]
                out[token] = Float(Float16(bitPattern: ptr[offset]))
            }
        default:
            for token in 0..<vocabSize {
                let index: [NSNumber] = [0, NSNumber(value: stepIndex), NSNumber(value: token)]
                out[token] = logits[index].floatValue
            }
        }

        return out
    }
}
