package com.varmeta.promptenhancer.model

data class EnhanceOptions(
    val topP: Float = 0.92f,
    val temperature: Float = 0.8f,
    val repetitionPenalty: Float = 1.2f,
    val noRepeatNgramSize: Int = 3,
    val minNewTokens: Int = 60,
    val maxNewTokens: Int = 63,
    val retries: Int = 3,
    val seed: ULong? = null,
    val timeoutMs: Int = 8_000
)
