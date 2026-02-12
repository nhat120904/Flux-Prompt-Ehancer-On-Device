package com.varmeta.prompter

data class EnhanceOptions(
    val topP: Float = 0.92f,
    val temperature: Float = 0.8f,
    val repetitionPenalty: Float = 1.2f,
    val noRepeatNgramSize: Int = 3,
    val minNewTokens: Int = 60,
    val maxNewTokens: Int = 180,
    val retries: Int = 3,
    val seed: Long? = null,
    val timeoutMs: Long = 8_000L
)
