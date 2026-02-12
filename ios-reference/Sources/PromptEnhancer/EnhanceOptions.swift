import Foundation

public struct EnhanceOptions: Sendable {
    public var topP: Float
    public var temperature: Float
    public var repetitionPenalty: Float
    public var noRepeatNgramSize: Int
    public var minNewTokens: Int
    public var maxNewTokens: Int
    public var retries: Int
    public var seed: UInt64?
    public var timeoutMs: Int

    public init(
        topP: Float = 0.92,
        temperature: Float = 0.8,
        repetitionPenalty: Float = 1.2,
        noRepeatNgramSize: Int = 3,
        minNewTokens: Int = 60,
        maxNewTokens: Int = 180,
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
