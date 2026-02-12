import Foundation

struct EnhanceResult: Sendable {
    let text: String
    let latencyMs: Int
    let tokensGenerated: Int
    let finishReason: String
}
