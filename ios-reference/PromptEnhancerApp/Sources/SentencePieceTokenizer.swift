import Foundation

private struct SentencePieceVocabEntry: Codable {
    let id: Int
    let piece: String
    let score: Float
}

private struct SentencePieceVocabPayload: Codable {
    let modelType: String
    let pieceSize: Int
    let unkId: Int
    let padId: Int
    let eosId: Int
    let decoderStartId: Int
    let extraIds: Int
    let entries: [SentencePieceVocabEntry]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case pieceSize = "piece_size"
        case unkId = "unk_id"
        case padId = "pad_id"
        case eosId = "eos_id"
        case decoderStartId = "decoder_start_id"
        case extraIds = "extra_ids"
        case entries
    }
}

enum SentencePieceTokenizerError: LocalizedError {
    case invalidVocab(String)

    var errorDescription: String? {
        switch self {
        case let .invalidVocab(reason):
            return "Invalid tokenizer vocabulary: \(reason)"
        }
    }
}

final class SentencePieceTokenizer: Tokenizer, @unchecked Sendable {
    private struct TrieNode {
        var children: [Character: Int] = [:]
        var tokenIds: [Int] = []
    }

    let eosTokenId: Int
    let padTokenId: Int
    let decoderStartTokenId: Int

    private let unkTokenId: Int
    private let extraIds: Int
    private let pieceById: [String]
    private let scoreById: [Float]
    private let trie: [TrieNode]

    init(vocabURL: URL) throws {
        let data = try Data(contentsOf: vocabURL)
        let payload = try JSONDecoder().decode(SentencePieceVocabPayload.self, from: data)

        guard payload.pieceSize > 0 else {
            throw SentencePieceTokenizerError.invalidVocab("piece_size must be > 0")
        }
        guard payload.entries.count == payload.pieceSize else {
            throw SentencePieceTokenizerError.invalidVocab("entries count mismatch")
        }

        var pieces = Array(repeating: "", count: payload.pieceSize)
        var scores = Array(repeating: Float(-100), count: payload.pieceSize)
        for entry in payload.entries {
            guard entry.id >= 0 && entry.id < payload.pieceSize else {
                throw SentencePieceTokenizerError.invalidVocab("entry id out of range: \(entry.id)")
            }
            pieces[entry.id] = entry.piece
            scores[entry.id] = entry.score
        }

        var trieNodes = [TrieNode()]
        for id in 0..<payload.pieceSize {
            let piece = pieces[id]
            if piece.hasPrefix("<") && piece.hasSuffix(">") {
                continue
            }
            var node = 0
            for ch in piece {
                if let next = trieNodes[node].children[ch] {
                    node = next
                } else {
                    let newIndex = trieNodes.count
                    trieNodes[node].children[ch] = newIndex
                    trieNodes.append(TrieNode())
                    node = newIndex
                }
            }
            trieNodes[node].tokenIds.append(id)
        }

        eosTokenId = payload.eosId
        padTokenId = payload.padId
        decoderStartTokenId = payload.decoderStartId
        unkTokenId = payload.unkId
        extraIds = payload.extraIds
        pieceById = pieces
        scoreById = scores
        trie = trieNodes
    }

    func encode(_ text: String, maxSourceTokens: Int) -> [Int] {
        guard maxSourceTokens > 0 else {
            return []
        }

        let normalized = normalizeForSentencePiece(text)
        var tokenIds: [Int]
        if normalized.isEmpty {
            tokenIds = []
        } else {
            tokenIds = encodeUnigram(normalized)
        }

        tokenIds.append(eosTokenId)

        if tokenIds.count > maxSourceTokens {
            tokenIds = Array(tokenIds.prefix(maxSourceTokens))
            if let last = tokenIds.last, last != eosTokenId {
                tokenIds[tokenIds.count - 1] = eosTokenId
            }
        }

        return tokenIds
    }

    func decode(_ tokenIds: [Int]) -> String {
        let extraIdStart = pieceById.count
        let extraIdEnd = extraIdStart + extraIds - 1

        var pieces: [String] = []
        pieces.reserveCapacity(tokenIds.count)

        for id in tokenIds {
            if id == eosTokenId || id == padTokenId || id == unkTokenId {
                continue
            }
            if id >= extraIdStart && id <= extraIdEnd {
                continue
            }
            if id >= 0 && id < pieceById.count {
                pieces.append(pieceById[id])
            }
        }

        let joined = pieces.joined()
        return joined.replacingOccurrences(of: "▁", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func encodeUnigram(_ normalized: String) -> [Int] {
        let chars = Array(normalized)
        let n = chars.count

        var bestScore = Array(repeating: -Float.greatestFiniteMagnitude, count: n + 1)
        var nextIndex = Array(repeating: -1, count: n + 1)
        var nextToken = Array(repeating: unkTokenId, count: n + 1)
        bestScore[n] = 0

        if n > 0 {
            for i in stride(from: n - 1, through: 0, by: -1) {
                var node = 0
                var j = i
                var found = false

                while j < n, let child = trie[node].children[chars[j]] {
                    node = child
                    j += 1

                    let tokenIds = trie[node].tokenIds
                    if !tokenIds.isEmpty {
                        found = true
                        for tokenId in tokenIds {
                            let candidate = scoreById[tokenId] + bestScore[j]
                            if candidate > bestScore[i] {
                                bestScore[i] = candidate
                                nextIndex[i] = j
                                nextToken[i] = tokenId
                            }
                        }
                    }
                }

                if !found {
                    bestScore[i] = -100 + bestScore[i + 1]
                    nextIndex[i] = i + 1
                    nextToken[i] = unkTokenId
                }
            }
        }

        var ids: [Int] = []
        ids.reserveCapacity(max(1, n / 2))

        var cursor = 0
        while cursor < n, nextIndex[cursor] != -1 {
            ids.append(nextToken[cursor])
            cursor = nextIndex[cursor]
        }

        return ids
    }

    private func normalizeForSentencePiece(_ text: String) -> String {
        var normalized = text.precomposedStringWithCompatibilityMapping

        if !normalized.isEmpty {
            var chars: [Character] = []
            chars.reserveCapacity(normalized.count)
            for ch in normalized {
                if ch.unicodeScalars.allSatisfy({ CharacterSet.whitespacesAndNewlines.contains($0) }) {
                    chars.append(" ")
                } else {
                    chars.append(ch)
                }
            }
            normalized = String(chars)
        }

        normalized = normalized.trimmingCharacters(in: .whitespaces)
        normalized = collapseSpaces(normalized)

        guard !normalized.isEmpty else {
            return ""
        }

        return "▁" + normalized.replacingOccurrences(of: " ", with: "▁")
    }

    private func collapseSpaces(_ text: String) -> String {
        guard !text.isEmpty else { return text }
        var out = ""
        out.reserveCapacity(text.count)

        var previousWasSpace = false
        for ch in text {
            if ch == " " {
                if !previousWasSpace {
                    out.append(ch)
                }
                previousWasSpace = true
            } else {
                out.append(ch)
                previousWasSpace = false
            }
        }
        return out
    }
}
