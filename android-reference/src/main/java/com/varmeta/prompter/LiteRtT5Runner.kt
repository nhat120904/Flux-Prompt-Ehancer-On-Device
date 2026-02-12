package com.varmeta.prompter

interface LiteRtT5Runner {
    suspend fun warmup()

    /**
     * Returns logits for the next token (vocab-sized float array) using current decoder_input_ids.
     * v1 reference path uses non-cache decoder graph; replace with KV-cache aware implementation when available.
     */
    suspend fun decodeNextTokenLogits(
        inputIds: IntArray,
        attentionMask: IntArray,
        decoderInputIds: IntArray
    ): FloatArray
}
