package com.varmeta.prompter

data class EnhanceResult(
    val text: String,
    val latencyMs: Long,
    val tokensGenerated: Int,
    val finishReason: String
)
