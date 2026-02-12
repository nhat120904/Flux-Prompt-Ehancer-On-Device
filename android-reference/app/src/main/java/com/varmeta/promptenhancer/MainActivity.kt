package com.varmeta.promptenhancer

import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.varmeta.promptenhancer.databinding.ActivityMainBinding
import com.varmeta.promptenhancer.inference.PromptEnhancerEngine
import com.varmeta.promptenhancer.model.EnhanceOptions
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val engine: PromptEnhancerEngine by lazy { PromptEnhancerEngine(applicationContext) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.promptInput.setText("beautiful house with text 'hello'")
        binding.topPInput.setText("0.92")
        binding.temperatureInput.setText("0.8")
        binding.maxTokensInput.setText("63")

        binding.enhanceButton.setOnClickListener {
            runEnhance()
        }
    }

    override fun onDestroy() {
        engine.close()
        super.onDestroy()
    }

    private fun runEnhance() {
        val prompt = binding.promptInput.text?.toString()?.trim().orEmpty()
        if (prompt.isEmpty()) {
            binding.promptInputLayout.error = "Prompt is required"
            return
        }
        binding.promptInputLayout.error = null

        val options = readOptions()

        lifecycleScope.launch {
            setLoading(true)
            binding.statusText.text = "Running on-device inference..."
            binding.statsText.text = ""

            try {
                val result = engine.enhance(prompt, options)
                binding.outputText.text = if (result.text.isNotBlank()) result.text else "(empty output)"
                binding.statusText.text = "Done (${result.finishReason})"
                binding.statsText.text = "Latency: ${result.latencyMs} ms | Tokens: ${result.tokensGenerated}"
            } catch (t: Throwable) {
                binding.statusText.text = "Failed"
                val detail = buildString {
                    append(t.message ?: t.javaClass.simpleName)
                    val cause = t.cause
                    if (cause != null && cause.message != null) {
                        append("\nCaused by: ")
                        append(cause.message)
                    }
                }
                binding.statsText.text = detail
                binding.outputText.text = detail
                Toast.makeText(this@MainActivity, "Enhance failed", Toast.LENGTH_LONG).show()
            } finally {
                setLoading(false)
            }
        }
    }

    private fun readOptions(): EnhanceOptions {
        val topP = binding.topPInput.text?.toString()?.toFloatOrNull() ?: 0.92f
        val temperature = binding.temperatureInput.text?.toString()?.toFloatOrNull() ?: 0.8f
        val maxTokens = binding.maxTokensInput.text?.toString()?.toIntOrNull() ?: 63

        return EnhanceOptions(
            topP = topP,
            temperature = temperature,
            maxNewTokens = maxTokens.coerceIn(1, 63)
        )
    }

    private fun setLoading(loading: Boolean) {
        binding.enhanceButton.isEnabled = !loading
        binding.promptInput.isEnabled = !loading
        binding.topPInput.isEnabled = !loading
        binding.temperatureInput.isEnabled = !loading
        binding.maxTokensInput.isEnabled = !loading
        binding.progressIndicator.visibility = if (loading) View.VISIBLE else View.GONE
    }
}
