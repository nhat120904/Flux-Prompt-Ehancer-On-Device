import Foundation

public struct EnhanceResult: Sendable {
    public let text: String
    public let latencyMs: Int
    public let tokensGenerated: Int
    public let finishReason: String

    public init(text: String, latencyMs: Int, tokensGenerated: Int, finishReason: String) {
        self.text = text
        self.latencyMs = latencyMs
        self.tokensGenerated = tokensGenerated
        self.finishReason = finishReason
    }
}
