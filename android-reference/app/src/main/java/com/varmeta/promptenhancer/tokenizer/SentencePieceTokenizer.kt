package com.varmeta.promptenhancer.tokenizer

import org.json.JSONObject
import java.io.InputStream
import java.text.Normalizer

class SentencePieceTokenizer(vocabStream: InputStream) : Tokenizer {
    private data class TrieNode(
        val children: MutableMap<Char, Int> = mutableMapOf(),
        val tokenIds: MutableList<Int> = mutableListOf()
    )

    override val eosTokenId: Int
    override val padTokenId: Int
    override val decoderStartTokenId: Int

    private val unkTokenId: Int
    private val extraIds: Int
    private val pieceById: Array<String>
    private val scoreById: FloatArray
    private val trie: List<TrieNode>

    init {
        val payload = JSONObject(vocabStream.bufferedReader(Charsets.UTF_8).use { it.readText() })
        val pieceSize = payload.getInt("piece_size")
        require(pieceSize > 0) { "piece_size must be > 0" }

        val entries = payload.getJSONArray("entries")
        require(entries.length() == pieceSize) { "entries count mismatch" }

        val pieces = Array(pieceSize) { "" }
        val scores = FloatArray(pieceSize) { -100f }
        for (i in 0 until entries.length()) {
            val entry = entries.getJSONObject(i)
            val id = entry.getInt("id")
            require(id in 0 until pieceSize) { "entry id out of range: $id" }
            pieces[id] = entry.getString("piece")
            scores[id] = entry.getDouble("score").toFloat()
        }

        val trieNodes = mutableListOf(TrieNode())
        for (id in 0 until pieceSize) {
            val piece = pieces[id]
            if (piece.startsWith("<") && piece.endsWith(">")) {
                continue
            }
            var node = 0
            for (ch in piece) {
                node = trieNodes[node].children.getOrPut(ch) {
                    trieNodes.add(TrieNode())
                    trieNodes.lastIndex
                }
            }
            trieNodes[node].tokenIds.add(id)
        }

        eosTokenId = payload.getInt("eos_id")
        padTokenId = payload.getInt("pad_id")
        decoderStartTokenId = payload.getInt("decoder_start_id")
        unkTokenId = payload.getInt("unk_id")
        extraIds = payload.getInt("extra_ids")
        pieceById = pieces
        scoreById = scores
        trie = trieNodes
    }

    override fun encode(text: String, maxSourceTokens: Int): IntArray {
        if (maxSourceTokens <= 0) {
            return intArrayOf()
        }

        val normalized = normalizeForSentencePiece(text)
        val tokenIds = if (normalized.isEmpty()) {
            mutableListOf<Int>()
        } else {
            encodeUnigram(normalized).toMutableList()
        }

        tokenIds.add(eosTokenId)

        if (tokenIds.size > maxSourceTokens) {
            val truncated = tokenIds.take(maxSourceTokens).toMutableList()
            if (truncated.lastOrNull() != eosTokenId) {
                truncated[truncated.lastIndex] = eosTokenId
            }
            return truncated.toIntArray()
        }

        return tokenIds.toIntArray()
    }

    override fun decode(tokenIds: IntArray): String {
        val extraIdStart = pieceById.size
        val extraIdEnd = extraIdStart + extraIds - 1

        val pieces = ArrayList<String>(tokenIds.size)
        for (id in tokenIds) {
            if (id == eosTokenId || id == padTokenId || id == unkTokenId) {
                continue
            }
            if (id in extraIdStart..extraIdEnd) {
                continue
            }
            if (id in pieceById.indices) {
                pieces.add(pieceById[id])
            }
        }

        return pieces.joinToString("")
            .replace("▁", " ")
            .trim()
    }

    private fun encodeUnigram(normalized: String): IntArray {
        val chars = normalized.toCharArray()
        val n = chars.size

        val bestScore = FloatArray(n + 1) { Float.NEGATIVE_INFINITY }
        val nextIndex = IntArray(n + 1) { -1 }
        val nextToken = IntArray(n + 1) { unkTokenId }
        bestScore[n] = 0f

        if (n > 0) {
            for (i in n - 1 downTo 0) {
                var node = 0
                var j = i
                var found = false

                while (j < n) {
                    val child = trie[node].children[chars[j]] ?: break
                    node = child
                    j += 1

                    val tokenIds = trie[node].tokenIds
                    if (tokenIds.isNotEmpty()) {
                        found = true
                        for (tokenId in tokenIds) {
                            val candidate = scoreById[tokenId] + bestScore[j]
                            if (candidate > bestScore[i]) {
                                bestScore[i] = candidate
                                nextIndex[i] = j
                                nextToken[i] = tokenId
                            }
                        }
                    }
                }

                if (!found) {
                    bestScore[i] = -100f + bestScore[i + 1]
                    nextIndex[i] = i + 1
                    nextToken[i] = unkTokenId
                }
            }
        }

        val out = ArrayList<Int>(maxOf(1, n / 2 + 1))
        var cursor = 0
        while (cursor < n && nextIndex[cursor] != -1) {
            out.add(nextToken[cursor])
            cursor = nextIndex[cursor]
        }

        return out.toIntArray()
    }

    private fun normalizeForSentencePiece(text: String): String {
        var normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
        if (normalized.isNotEmpty()) {
            val chars = StringBuilder(normalized.length)
            for (ch in normalized) {
                chars.append(if (ch.isWhitespace()) ' ' else ch)
            }
            normalized = chars.toString()
        }

        normalized = collapseSpaces(normalized.trim())
        if (normalized.isEmpty()) {
            return ""
        }

        return "▁" + normalized.replace(" ", "▁")
    }

    private fun collapseSpaces(text: String): String {
        if (text.isEmpty()) {
            return text
        }

        val out = StringBuilder(text.length)
        var previousWasSpace = false
        for (ch in text) {
            if (ch == ' ') {
                if (!previousWasSpace) {
                    out.append(ch)
                }
                previousWasSpace = true
            } else {
                out.append(ch)
                previousWasSpace = false
            }
        }
        return out.toString()
    }
}
