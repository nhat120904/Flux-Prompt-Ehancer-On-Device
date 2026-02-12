package com.varmeta.promptenhancer.model

data class EnhanceResult(
    val text: String,
    val latencyMs: Int,
    val tokensGenerated: Int,
    val finishReason: String
)
