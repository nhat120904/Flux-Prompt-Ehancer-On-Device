import Foundation

protocol CoreMLT5Runner: Sendable {
    func warmup() async throws

    func decodeNextTokenLogits(
        inputIds: [Int32],
        attentionMask: [Int32],
        decoderInputIds: [Int32]
    ) async throws -> [Float]
}
