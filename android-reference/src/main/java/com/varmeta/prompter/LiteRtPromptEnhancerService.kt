package com.varmeta.prompter

import kotlin.math.exp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import kotlin.coroutines.coroutineContext
import kotlin.random.Random

class LiteRtPromptEnhancerService(
    private val tokenizer: Tokenizer,
    private val runner: LiteRtT5Runner
) {
    suspend fun enhance(input: String, options: EnhanceOptions = EnhanceOptions()): EnhanceResult =
        withContext(Dispatchers.Default) {
            val start = System.currentTimeMillis()
            val inputIds = tokenizer.encode(input, maxSourceTokens = 128)
            val attentionMask = IntArray(inputIds.size) { 1 }

            var bestText = ""
            var bestTokenCount = 0
            var finishReason = "max_tokens"

            repeat(options.retries) { attempt ->
                coroutineContext.ensureActive()
                if ((System.currentTimeMillis() - start) > options.timeoutMs) {
                    finishReason = "timeout"
                    return@repeat
                }

                val generated = generateOnce(inputIds, attentionMask, options, attempt)
                val text = tokenizer.decode(generated.toIntArray()).trim()
                if (text.length > bestText.length) {
                    bestText = text
                    bestTokenCount = generated.size
                }

                if (isCompleteSentence(text)) {
                    finishReason = "eos"
                    return@withContext EnhanceResult(
                        text = text,
                        latencyMs = System.currentTimeMillis() - start,
                        tokensGenerated = generated.size,
                        finishReason = finishReason
                    )
                }
            }

            if (bestText.isNotEmpty() && !isCompleteSentence(bestText)) {
                bestText += "."
            }
            EnhanceResult(
                text = bestText,
                latencyMs = System.currentTimeMillis() - start,
                tokensGenerated = bestTokenCount,
                finishReason = finishReason
            )
        }

    suspend fun warmup() {
        runner.warmup()
    }

    private suspend fun generateOnce(
        inputIds: IntArray,
        attentionMask: IntArray,
        options: EnhanceOptions,
        attempt: Int
    ): MutableList<Int> {
        val output = mutableListOf<Int>()
        val decoder = mutableListOf(tokenizer.decoderStartTokenId)
        val seenTokens = mutableListOf<Int>()
        val rng = Random(options.seed?.plus(attempt) ?: System.currentTimeMillis())

        for (step in 0 until options.maxNewTokens) {
            coroutineContext.ensureActive()
            val logits = runner.decodeNextTokenLogits(
                inputIds = inputIds,
                attentionMask = attentionMask,
                decoderInputIds = decoder.toIntArray()
            )
            val adjusted = applyLogitRules(
                rawLogits = logits,
                generatedTokens = seenTokens,
                noRepeatNgramSize = options.noRepeatNgramSize,
                repetitionPenalty = options.repetitionPenalty
            )
            val nextTokenId = sampleTopP(
                logits = adjusted,
                topP = options.topP,
                temperature = options.temperature,
                random = rng
            )
            decoder.add(nextTokenId)
            if (nextTokenId == tokenizer.eosTokenId && step + 1 >= options.minNewTokens) {
                break
            }
            output.add(nextTokenId)
            seenTokens.add(nextTokenId)
        }
        return output
    }

    private fun isCompleteSentence(text: String): Boolean {
        val trimmed = text.trimEnd()
        return trimmed.endsWith(".") || trimmed.endsWith("!") || trimmed.endsWith("?")
    }

    private fun applyLogitRules(
        rawLogits: FloatArray,
        generatedTokens: List<Int>,
        noRepeatNgramSize: Int,
        repetitionPenalty: Float
    ): FloatArray {
        val adjusted = rawLogits.copyOf()

        for (token in generatedTokens) {
            if (token !in adjusted.indices) continue
            adjusted[token] = if (adjusted[token] < 0f) {
                adjusted[token] * repetitionPenalty
            } else {
                adjusted[token] / repetitionPenalty
            }
        }

        if (noRepeatNgramSize > 1 && generatedTokens.size >= noRepeatNgramSize - 1) {
            val prefix = generatedTokens.takeLast(noRepeatNgramSize - 1)
            val blocked = mutableSetOf<Int>()
            for (i in 0..generatedTokens.size - noRepeatNgramSize) {
                val ngram = generatedTokens.subList(i, i + noRepeatNgramSize)
                if (ngram.dropLast(1) == prefix) {
                    blocked.add(ngram.last())
                }
            }
            for (token in blocked) {
                if (token in adjusted.indices) {
                    adjusted[token] = Float.NEGATIVE_INFINITY
                }
            }
        }
        return adjusted
    }

    private fun sampleTopP(
        logits: FloatArray,
        topP: Float,
        temperature: Float,
        random: Random
    ): Int {
        val finite = logits.mapIndexedNotNull { idx, value ->
            if (value.isFinite()) {
                idx to (value / temperature)
            } else {
                null
            }
        }
        if (finite.isEmpty()) {
            return logits.indices.maxByOrNull { logits[it] } ?: 0
        }

        val maxLogit = finite.maxOf { it.second }
        val weighted = finite.map { (idx, scaled) ->
            idx to exp((scaled - maxLogit).toDouble())
        }.sortedByDescending { it.second }

        val allMass = weighted.sumOf { it.second }
        val candidates = mutableListOf<Pair<Int, Double>>()
        var cumulativeMass = 0.0
        for ((idx, weight) in weighted) {
            candidates.add(idx to weight)
            cumulativeMass += weight
            if ((cumulativeMass / allMass) >= topP && candidates.size > 1) {
                break
            }
        }

        val norm = candidates.sumOf { it.second }
        val target = random.nextDouble() * norm
        var cumulative = 0.0
        for ((idx, p) in candidates) {
            cumulative += p
            if (cumulative >= target) {
                return idx
            }
        }
        return candidates.last().first
    }
}
