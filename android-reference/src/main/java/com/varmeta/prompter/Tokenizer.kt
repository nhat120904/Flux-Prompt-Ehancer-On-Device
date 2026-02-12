package com.varmeta.prompter

interface Tokenizer {
    val eosTokenId: Int
    val padTokenId: Int
    val decoderStartTokenId: Int

    fun encode(text: String, maxSourceTokens: Int = 128): IntArray
    fun decode(tokenIds: IntArray): String
}
