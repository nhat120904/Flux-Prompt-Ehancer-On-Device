package com.varmeta.promptenhancer.inference

import android.content.Context
import com.varmeta.promptenhancer.model.EnhanceOptions
import com.varmeta.promptenhancer.model.EnhanceResult
import com.varmeta.promptenhancer.tokenizer.SentencePieceTokenizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.Closeable

class PromptEnhancerEngine(private val context: Context) : Closeable {
    private val lock = Any()

    @Volatile
    private var service: PromptEnhancerService? = null

    @Volatile
    private var runner: TfliteT5Runner? = null

    suspend fun enhance(prompt: String, options: EnhanceOptions = EnhanceOptions()): EnhanceResult {
        return withContext(Dispatchers.Default) {
            getOrCreateService().enhance(prompt, options)
        }
    }

    override fun close() {
        synchronized(lock) {
            runner?.close()
            runner = null
            service = null
        }
    }

    private fun getOrCreateService(): PromptEnhancerService {
        service?.let { return it }

        synchronized(lock) {
            service?.let { return it }

            val tokenizer = context.assets.open("tokenizer/tokenizer_vocab.json").use {
                SentencePieceTokenizer(it)
            }
            val newRunner = TfliteT5Runner(context)
            val newService = PromptEnhancerService(tokenizer, newRunner)

            runner = newRunner
            service = newService
            return newService
        }
    }
}
