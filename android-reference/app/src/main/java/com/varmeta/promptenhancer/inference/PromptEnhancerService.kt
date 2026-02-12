package com.varmeta.promptenhancer.inference

import android.os.SystemClock
import com.varmeta.promptenhancer.model.EnhanceOptions
import com.varmeta.promptenhancer.model.EnhanceResult
import com.varmeta.promptenhancer.tokenizer.Tokenizer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class PromptEnhancerService(
    private val tokenizer: Tokenizer,
    private val runner: TfliteT5Runner
) {

    fun enhance(input: String, options: EnhanceOptions = EnhanceOptions()): EnhanceResult {
        val start = SystemClock.elapsedRealtime()
        val inputIds = tokenizer.encode(input, runner.maxInputTokens())
        val attentionMask = IntArray(inputIds.size) { idx ->
            if (inputIds[idx] == tokenizer.padTokenId) 0 else 1
        }

        var bestText = ""
        var bestTokens = 0
        var finishReason = "max_tokens"

        for (attempt in 0 until options.retries) {
            if (elapsedMs(start) > options.timeoutMs) {
                finishReason = "timeout"
                break
            }

            val tokens = generateOnce(
                inputIds = inputIds,
                attentionMask = attentionMask,
                options = options,
                attempt = attempt,
                startMs = start
            )

            val text = tokenizer.decode(tokens.toIntArray()).trim()
            if (text.length > bestText.length) {
                bestText = text
                bestTokens = tokens.size
            }

            if (isCompleteSentence(text)) {
                finishReason = "eos"
                return EnhanceResult(
                    text = text,
                    latencyMs = elapsedMs(start),
                    tokensGenerated = tokens.size,
                    finishReason = finishReason
                )
            }
        }

        if (bestText.isNotEmpty() && !isCompleteSentence(bestText)) {
            bestText += "."
        }

        return EnhanceResult(
            text = bestText,
            latencyMs = elapsedMs(start),
            tokensGenerated = bestTokens,
            finishReason = finishReason
        )
    }

    private fun generateOnce(
        inputIds: IntArray,
        attentionMask: IntArray,
        options: EnhanceOptions,
        attempt: Int,
        startMs: Long
    ): MutableList<Int> {
        val output = mutableListOf<Int>()
        val decoderIds = mutableListOf(tokenizer.decoderStartTokenId)
        val seenTokens = mutableListOf<Int>()

        val seed = options.seed ?: (System.currentTimeMillis().toULong() + attempt.toULong())
        val rng = SeededGenerator(seed)

        val maxModelNewTokens = runner.maxDecoderTokens() - 1
        val cappedMaxNewTokens = min(options.maxNewTokens, maxModelNewTokens)
        val cappedMinNewTokens = min(options.minNewTokens, cappedMaxNewTokens)

        for (step in 0 until cappedMaxNewTokens) {
            if (elapsedMs(startMs) > options.timeoutMs) {
                break
            }

            val logits = runner.decodeNextTokenLogits(
                inputIds = inputIds,
                attentionMask = attentionMask,
                decoderInputIds = decoderIds.toIntArray()
            )

            val adjusted = applyLogitRules(
                logits = logits,
                generatedTokens = seenTokens,
                noRepeatNgramSize = options.noRepeatNgramSize,
                repetitionPenalty = options.repetitionPenalty
            )

            val next = sampleTopP(
                logits = adjusted,
                topP = options.topP,
                temperature = options.temperature,
                rng = rng
            )

            decoderIds.add(next)
            if (next == tokenizer.eosTokenId && step + 1 >= cappedMinNewTokens) {
                break
            }

            output.add(next)
            seenTokens.add(next)
        }

        return output
    }

    private fun isCompleteSentence(text: String): Boolean {
        val trimmed = text.trim()
        return trimmed.endsWith(".") || trimmed.endsWith("!") || trimmed.endsWith("?")
    }

    private fun applyLogitRules(
        logits: FloatArray,
        generatedTokens: List<Int>,
        noRepeatNgramSize: Int,
        repetitionPenalty: Float
    ): FloatArray {
        val adjusted = logits.copyOf()

        for (token in generatedTokens) {
            if (token !in adjusted.indices) {
                continue
            }
            adjusted[token] = if (adjusted[token] < 0f) {
                adjusted[token] * repetitionPenalty
            } else {
                adjusted[token] / repetitionPenalty
            }
        }

        if (noRepeatNgramSize > 1 && generatedTokens.size >= noRepeatNgramSize - 1) {
            val prefixStart = generatedTokens.size - (noRepeatNgramSize - 1)
            val blocked = HashSet<Int>()

            if (generatedTokens.size >= noRepeatNgramSize) {
                for (i in 0..(generatedTokens.size - noRepeatNgramSize)) {
                    var prefixMatch = true
                    for (j in 0 until noRepeatNgramSize - 1) {
                        if (generatedTokens[i + j] != generatedTokens[prefixStart + j]) {
                            prefixMatch = false
                            break
                        }
                    }
                    if (prefixMatch) {
                        blocked.add(generatedTokens[i + noRepeatNgramSize - 1])
                    }
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
        rng: SeededGenerator
    ): Int {
        val safeTemperature = max(temperature, 1e-5f)
        val safeTopP = min(max(topP, 0.01f), 1f)

        val finite = ArrayList<Pair<Int, Double>>(logits.size)
        for (i in logits.indices) {
            val value = logits[i]
            if (value.isFinite()) {
                finite.add(i to (value / safeTemperature).toDouble())
            }
        }

        if (finite.isEmpty()) {
            return logits.indices.maxByOrNull { logits[it] } ?: 0
        }

        val maxLogit = finite.maxOf { it.second }
        val weighted = finite.map { (idx, scaled) ->
            idx to exp(scaled - maxLogit)
        }.sortedByDescending { it.second }

        val allMass = weighted.sumOf { it.second }
        if (allMass <= 0.0) {
            return weighted.first().first
        }

        val candidates = ArrayList<Pair<Int, Double>>()
        var cumulativeMass = 0.0
        for ((idx, weight) in weighted) {
            candidates.add(idx to weight)
            cumulativeMass += weight
            if ((cumulativeMass / allMass) >= safeTopP.toDouble() && candidates.size > 1) {
                break
            }
        }

        val norm = candidates.sumOf { it.second }
        if (norm <= 0.0) {
            return candidates.first().first
        }

        val target = rng.nextDouble() * norm
        var sum = 0.0
        for ((idx, p) in candidates) {
            sum += p
            if (sum >= target) {
                return idx
            }
        }

        return candidates.last().first
    }

    private fun elapsedMs(start: Long): Int {
        return (SystemClock.elapsedRealtime() - start).toInt()
    }

    private class SeededGenerator(seed: ULong) {
        private var state: ULong = if (seed == 0UL) 0x123456789abcdefUL else seed

        fun nextDouble(): Double {
            state = state * 6364136223846793005UL + 1UL
            val bits = state shr 11
            return bits.toDouble() / (1UL shl 53).toDouble()
        }
    }
}
