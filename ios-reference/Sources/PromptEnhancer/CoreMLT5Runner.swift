import Foundation

public protocol CoreMLT5Runner: Sendable {
    func warmup() async throws

    /// Returns vocab logits for the next token using full decoder_input_ids (non-KV-cache path).
    func decodeNextTokenLogits(
        inputIds: [Int32],
        attentionMask: [Int32],
        decoderInputIds: [Int32]
    ) async throws -> [Float]
}
