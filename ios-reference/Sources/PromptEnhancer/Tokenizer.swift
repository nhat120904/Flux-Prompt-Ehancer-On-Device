import Foundation

public protocol Tokenizer: Sendable {
    var eosTokenId: Int { get }
    var padTokenId: Int { get }
    var decoderStartTokenId: Int { get }

    func encode(_ text: String, maxSourceTokens: Int) -> [Int]
    func decode(_ tokenIds: [Int]) -> String
}
