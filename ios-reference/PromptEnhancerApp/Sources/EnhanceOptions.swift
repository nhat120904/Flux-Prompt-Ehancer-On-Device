import Foundation

struct EnhanceOptions: Sendable {
    var topP: Float
    var temperature: Float
    var repetitionPenalty: Float
    var noRepeatNgramSize: Int
    var minNewTokens: Int
    var maxNewTokens: Int
    var retries: Int
    var seed: UInt64?
    var timeoutMs: Int

    init(
        topP: Float = 0.92,
        temperature: Float = 0.8,
        repetitionPenalty: Float = 1.2,
        noRepeatNgramSize: Int = 3,
        minNewTokens: Int = 60,
        maxNewTokens: Int = 63,
        retries: Int = 3,
        seed: UInt64? = nil,
        timeoutMs: Int = 8000
    ) {
        self.topP = topP
        self.temperature = temperature
        self.repetitionPenalty = repetitionPenalty
        self.noRepeatNgramSize = noRepeatNgramSize
        self.minNewTokens = minNewTokens
        self.maxNewTokens = maxNewTokens
        self.retries = retries
        self.seed = seed
        self.timeoutMs = timeoutMs
    }
}
