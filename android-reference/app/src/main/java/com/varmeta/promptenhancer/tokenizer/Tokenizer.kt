package com.varmeta.promptenhancer.tokenizer

interface Tokenizer {
    val eosTokenId: Int
    val padTokenId: Int
    val decoderStartTokenId: Int

    fun encode(text: String, maxSourceTokens: Int): IntArray
    fun decode(tokenIds: IntArray): String
}
